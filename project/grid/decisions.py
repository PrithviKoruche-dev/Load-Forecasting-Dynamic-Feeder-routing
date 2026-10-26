"""Decision layer: structured recommendations (gap item F6), operating modes (F9), emergency stop,
rollback and action history (F10).

The engine answers "what is the best safe configuration for these loads?". The Controller turns that
into operations on the (simulated) network, according to the operating mode:

    monitor    alerts only; no switching is proposed or applied
    recommend  a recommendation is proposed; it is applied only when the operator approves it
    auto       safe recommendations are applied immediately (simulated automatic operation)

Emergency stop halts automation at once: the mode drops to monitor, any pending recommendation is
cancelled and approvals are refused until the stop is cleared. It deliberately does NOT switch the
network back to normal on its own, because a blind change could itself overload a transformer; the
operator can roll back or reset explicitly.

Every event is kept in Controller.history (persisted to the database in Phase 3).
"""
import datetime as dt
from dataclasses import asdict, dataclass, field

from risk import RiskModel, forecast_errors
from core import config  # risk.py has already put project/ on sys.path

MODES = {"monitor": "Monitor only", "recommend": "Recommend (operator approves)", "auto": "Automatic"}


class DecisionError(Exception):
    """An operation the controller refuses (wrong mode, emergency stop, stale recommendation, ...)."""


@dataclass
class Recommendation:
    rec_id: str
    created_at: str                     # wall-clock time
    for_time: str                       # the (simulated) time the loads belong to
    context: str                        # scenario / settings the loads were produced under
    kind: str                           # reroute | restore | no_change | no_action | no_safe_option
    status: str                         # pending | applied | rejected | superseded | info
    summary: str
    threshold: float
    from_config: list
    target: list
    plan: list = field(default_factory=list)
    before: dict = None
    after: dict = None
    checks: list = field(default_factory=list)
    losses_before_kw: float = None
    losses_after_kw: float = None
    confidence: dict = None
    candidates: list = field(default_factory=list)
    binding: list = field(default_factory=list)
    engine_status: str = ""
    compute_s: float = None

    @property
    def actionable(self):
        return self.kind in ("reroute", "restore")

    def to_dict(self):
        return asdict(self)


class Controller:
    def __init__(self, engine, ops_cfg=None, error_by_tx=None, now=None):
        self.engine = engine
        self.risk = RiskModel(ops_cfg)
        ops = ops_cfg if ops_cfg is not None else config.operations()
        self.mode = ops["decisions"]["default_mode"]
        self.error_by_tx = forecast_errors() if error_by_tx is None else error_by_tx
        self.active = []                # action keys in force on the network
        self.pending = None
        self.last = None                # most recent recommendation of any kind
        self.history = []
        self.estopped = False
        self._undo = []                 # previous configurations, for rollback
        self._seq = 0
        self._now = now or (lambda: dt.datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------ events
    def _event(self, kind, message, sim_time=None, rec_id=None, **data):
        e = {"seq": len(self.history) + 1, "wall_time": self._now(), "sim_time": sim_time, "type": kind,
             "message": message, "rec_id": rec_id, "mode": self.mode, "configuration": list(self.active), **data}
        self.history.append(e)
        return e

    def set_mode(self, mode, sim_time=None):
        if mode not in MODES:
            raise DecisionError(f"unknown mode {mode!r}")
        if mode != "monitor" and self.estopped:
            raise DecisionError("Emergency stop is active: clear it before leaving monitor-only mode.")
        if mode == self.mode:
            return
        old, self.mode = self.mode, mode
        if mode != "recommend":
            self._supersede("mode changed")
        self._event("mode", f"Mode changed: {MODES[old]} -> {MODES[mode]}", sim_time)

    # ------------------------------------------------------------------ recommendations
    def _checks(self, m, threshold):
        if not m:
            return []
        e = self.engine
        out = [{"check": f"{t} loading", "value": f"{x:.1f}%", "limit": f"<= {threshold:g}% target, {e.hard_limit:g}% max",
                "ok": x <= threshold} for t, x in m["loading_by_tx"].items()]
        out += [{"check": "lowest bus voltage", "value": f"{m['min_voltage_pu']:.4f} pu ({m.get('min_voltage_bus', '')})",
                 "limit": f">= {e.v_min:g} pu", "ok": m["min_voltage_pu"] >= e.v_min},
                {"check": "highest bus voltage", "value": f"{m['max_voltage_pu']:.4f} pu", "limit": f"<= {e.v_max:g} pu",
                 "ok": m["max_voltage_pu"] <= e.v_max},
                {"check": "highest line loading", "value": f"{m['max_line_loading']:.1f}%", "limit": f"<= {e.line_limit:g}%",
                 "ok": m["max_line_loading"] <= e.line_limit},
                {"check": "radial, every load supplied", "value": "yes", "limit": "required", "ok": True}]
        return out

    def recommend(self, loads_kw, for_time, context="", threshold=None, use_mv=None):
        """Assess the loads and produce a Recommendation. In monitor mode, or while the emergency stop is
        active, nothing is recommended (returns None). In auto mode an actionable recommendation is applied."""
        if self.mode == "monitor" or self.estopped:
            return None
        e = self.engine
        thr = e.threshold if threshold is None else threshold
        r = e.evaluate_and_reroute(loads_kw, thr, use_mv)
        before = r["before"]
        if self.active and before is not None:          # what is actually in force now
            cur = e.evaluate_configuration(loads_kw, self.active)
            before = cur["metrics"] or before
        if r["status"] == "rerouted":
            target = r["target"]
            if sorted(target) == sorted(self.active):
                kind, summary = "no_change", "The network is already in the recommended configuration."
            else:
                kind = "reroute"
                summary = (f"Switch to: {'; '.join(r['action'])}. Peak loading {before['max_loading']:.1f}% -> "
                           f"{r['after']['max_loading']:.1f}%.")
            after = r["after"]
        elif r["status"] == "no_action":
            target, after = [], r["before"]
            if self.active:
                kind = "restore"
                summary = (f"Loads are below {thr:g}% in the normal configuration: return to it. "
                           f"Peak loading {after['max_loading']:.1f}%.")
            else:
                kind, summary = "no_action", f"No transformer above {thr:g}%: no action needed."
        else:
            target, after = list(self.active), r["after"]
            kind = "no_safe_option"
            summary = "No safe switching configuration brings every transformer under the threshold."
        self._seq += 1
        rec = Recommendation(
            rec_id=f"REC-{self._seq:04d}", created_at=self._now(), for_time=str(for_time), context=context,
            kind=kind, status="pending" if kind in ("reroute", "restore") else "info", summary=summary,
            threshold=thr, from_config=list(self.active), target=list(target),
            plan=e.switch_plan(self.active, target) if kind in ("reroute", "restore") else [],
            before=before, after=after,
            checks=self._checks(after, thr) if kind in ("reroute", "restore", "no_change") else [],
            losses_before_kw=(before or {}).get("losses_kw"), losses_after_kw=(after or {}).get("losses_kw"),
            confidence=self.risk.confidence((after or {}).get("loading_by_tx"), self.error_by_tx, thr)
            if kind in ("reroute", "restore", "no_change") else None,
            candidates=r["candidates"], binding=r["binding"], engine_status=r["status"], compute_s=r["compute_s"])
        self.last = rec
        self._supersede(f"replaced by {rec.rec_id}")          # a newer assessment replaces any older proposal
        if rec.actionable:
            self.pending = rec
            self._event("recommendation", f"{rec.rec_id}: {summary}", for_time, rec.rec_id, plan=rec.plan)
            if self.mode == "auto":
                self._apply(rec, for_time, how="auto")
        elif kind == "no_safe_option":
            self._event("no_safe_option", f"{rec.rec_id}: {summary} " + " ".join(rec.binding[:1]), for_time, rec.rec_id)
        return rec

    def _supersede(self, why):
        if self.pending is not None and self.pending.status == "pending":
            self.pending.status = "superseded"
            self._event("superseded", f"{self.pending.rec_id} cancelled: {why}", rec_id=self.pending.rec_id)
        self.pending = None

    def approve(self, rec_id, for_time=None, context=None):
        """Apply a pending recommendation. Refused if it is not the pending one, if the emergency stop is
        active, outside recommend mode, or if the loads it was computed for are no longer current."""
        if self.estopped:
            raise DecisionError("Emergency stop is active: no switching can be approved.")
        if self.mode != "recommend":
            raise DecisionError("Approval is only used in recommend mode.")
        rec = self.pending
        if rec is None or rec.rec_id != rec_id or rec.status != "pending":
            raise DecisionError(f"{rec_id} is not the pending recommendation.")
        if (for_time is not None and str(for_time) != rec.for_time) or (context is not None and context != rec.context):
            raise DecisionError(f"{rec_id} was computed for {rec.for_time} ({rec.context}); conditions have changed, "
                                "so get a new recommendation.")
        if rec.from_config != self.active:
            raise DecisionError(f"{rec_id} assumed a different switch configuration; get a new recommendation.")
        self._event("approved", f"{rec_id} approved by the operator", rec.for_time, rec_id)
        self._apply(rec, rec.for_time, how="operator")
        return rec

    def reject(self, rec_id, reason=""):
        rec = self.pending
        if rec is None or rec.rec_id != rec_id:
            raise DecisionError(f"{rec_id} is not the pending recommendation.")
        rec.status = "rejected"
        self.pending = None
        self._event("rejected", f"{rec_id} rejected by the operator" + (f": {reason}" if reason else ""), rec.for_time, rec_id)

    def _apply(self, rec, sim_time, how):
        self._undo.append(list(self.active))
        self.active = list(rec.target)
        rec.status = "applied"
        self.pending = None
        self._event("applied", f"{rec.rec_id} applied ({how}): " + (", ".join(s["switch_id"] + " " + s["operation"]
                    for s in rec.plan) or "no switch operations"), sim_time, rec.rec_id, plan=rec.plan)

    # ------------------------------------------------------------------ operator actions
    def set_configuration(self, keys, sim_time=None, reason="manual switching"):
        """Operator sets the switch configuration directly (manual override)."""
        keys = [self.engine.resolve(k).key for k in keys]
        if keys == self.active:
            return []
        ok, why = self.engine.check_radial(keys)
        if not ok:
            raise DecisionError(f"Refused: this configuration {why}.")
        plan = self.engine.switch_plan(self.active, keys)
        self._undo.append(list(self.active))
        self.active = keys
        self._supersede(reason)
        self._event("manual", f"{reason}: " + (", ".join(s["switch_id"] + " " + s["operation"] for s in plan)
                                               or "no change"), sim_time, plan=plan)
        return plan

    def reset_to_normal(self, sim_time=None):
        return self.set_configuration([], sim_time, reason="reset to normal configuration")

    def rollback(self, sim_time=None):
        """Undo the most recent change of configuration (applied recommendation or manual switching)."""
        if not self._undo:
            raise DecisionError("Nothing to roll back.")
        previous = self._undo.pop()
        plan = self.engine.switch_plan(self.active, previous)
        self.active = previous
        self._supersede("rollback")
        self._event("rollback", "Rolled back to the previous configuration: "
                    + (", ".join(s["switch_id"] + " " + s["operation"] for s in plan) or "no change"), sim_time, plan=plan)
        return plan

    def emergency_stop(self, sim_time=None):
        if self.estopped:
            return
        self.estopped = True
        self._supersede("emergency stop")
        old, self.mode = self.mode, "monitor"
        self._event("emergency_stop", f"EMERGENCY STOP: automation halted (was {MODES[old]}); "
                    "switches left as they are", sim_time)

    def clear_emergency_stop(self, sim_time=None):
        if not self.estopped:
            return
        self.estopped = False
        self._event("emergency_stop_cleared", "Emergency stop cleared; mode stays Monitor only until changed", sim_time)

    @property
    def can_rollback(self):
        return bool(self._undo)

    def snapshot(self):
        """Plain-data state (for the database in Phase 3)."""
        return {"mode": self.mode, "active": list(self.active), "estopped": self.estopped, "undo": [list(u) for u in self._undo],
                "pending": self.pending.to_dict() if self.pending else None, "seq": self._seq, "history": list(self.history)}
