"""Decision layer: structured recommendations (gap item F6), operating modes (F9), emergency stop,
rollback and action history (F10).

The engine answers "what is the best safe configuration for these loads?". The Controller turns that
into operations on the (simulated) network, according to the operating mode:

    monitor    alerts only; no switching is proposed or applied
    recommend  a recommendation is proposed; it is applied only when the operator approves it
    auto       safe recommendations are applied immediately (simulated automatic operation)

Recommendation kinds:
    reroute         a configuration that brings every transformer under the threshold
    restore         loads are back under the threshold in the normal configuration: return to it
    mitigate        no configuration reaches the threshold, but a better one than the present exists that
                    respects every hard limit (e.g. T4 122% -> 97%): offered as a mitigation, never as a fix
    no_change       the network is already in the recommended configuration
    no_action       nothing is overloaded
    no_safe_option  nothing reaches the threshold and nothing safer than the present configuration exists

Emergency stop halts automation at once: the mode drops to monitor, any pending recommendation is
cancelled and approvals are refused until the stop is cleared. It deliberately does NOT switch the
network back to normal on its own, because a blind change could itself overload a transformer; the
operator can roll back or reset explicitly.

Every event is kept in Controller.history. With a Store (Phase 3) every event, recommendation (with all
its candidates), executed switch operation and the controller's own state (mode, configuration, emergency
stop, undo stack) is written to the database, and a Controller created on the same database later picks
up exactly where the previous one stopped.
"""
import datetime as dt
import functools
import threading
from dataclasses import asdict, dataclass, field

from risk import RiskModel, forecast_errors
from core import config, ids  # risk.py has already put project/ on sys.path
from core.logging_setup import get_logger

log = get_logger("decisions")

MODES = {"monitor": "Monitor only", "recommend": "Recommend (operator approves)", "auto": "Automatic"}
ACTIONABLE = ("reroute", "restore", "mitigate")
# audit-log severity of each event type (no_safe_option and recommendations are refined per event)
SEVERITY = {"mode": "info", "recommendation": "info", "approved": "info", "applied": "info", "rejected": "info",
            "superseded": "info", "manual": "info", "rollback": "warning", "no_safe_option": "warning",
            "emergency_stop": "critical", "emergency_stop_cleared": "warning",
            "service": "info", "clock": "info"}       # background service and simulation clock (Phase 5)


def _locked(fn):
    """Serialise public controller operations: one controller is shared by every dashboard session."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


def _describe(v):
    """Short present-tense text for a limit violation from the engine."""
    unit = {"transformer": "%", "line": "% of its rating"}.get(v["constraint"], " pu")
    where = v["element"].upper() if v["constraint"] != "line" else f"line {v['element']}"
    if v["constraint"].startswith("voltage"):
        where = f"bus {v['element'].upper()}"
    limit = f"{v['limit']:g}%" if v["constraint"] in ("transformer", "line") else f"{v['limit']:g} pu"
    return f"{where} at {v['value']}{unit} (limit {limit})"


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
        return self.kind in ACTIONABLE

    def to_dict(self):
        return asdict(self)


class Controller:
    def __init__(self, engine, ops_cfg=None, error_by_tx=None, now=None, store=None, name="default"):
        """store: a storage.Store to persist to and restore from (None = in memory only).
        name: the controller's identity in the database (one network, one controller)."""
        self.engine = engine
        self.store, self.name = store, name
        self._lock = threading.RLock()
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
        self._rec_assets = {}           # rec_id -> asset IDs it concerns (for the audit log)
        if store is not None:
            self._restore()

    # ------------------------------------------------------------------ persistence
    def _restore(self):
        st = self.store.load_state(self.name)
        if st:
            self.mode, self.estopped, self._seq = st["mode"], st["estopped"], st["rec_seq"]
            valid = [k for k in st["active"] if k in self.engine.actions]
            if valid != st["active"] or not self.engine.check_radial(valid)[0]:
                log.warning("stored configuration %s is not valid for the current network; using normal", st["active"])
                valid = []
            self.active = valid
            self._undo = [u for u in st["undo"] if all(k in self.engine.actions for k in u)]
        self.history = list(reversed(self.store.events(controller=self.name, limit=500)))
        p = self.store.pending_recommendation(self.name)
        self.pending = Recommendation(**p) if p else None
        last = self.store.latest_recommendation(self.name)
        self.last = Recommendation(**last) if last else None
        if st:
            log.info("controller '%s' restored: mode %s, configuration %s, %d events, pending %s", self.name,
                     self.mode, self.active or "normal", len(self.history), self.pending.rec_id if self.pending else None)

    def _save_state(self):
        if self.store is not None:
            self.store.save_state(self.name, self.mode, self.active, self.estopped, self._undo, self._seq)

    def _save_rec(self, rec):
        if self.store is not None:
            self.store.save_recommendation(self.name, rec.to_dict())

    def _set_status(self, rec, status):
        rec.status = status
        if self.store is not None:
            self.store.update_recommendation_status(rec.rec_id, status)

    # ------------------------------------------------------------------ which assets an event concerns
    def _plan_assets(self, plan):
        out = set()
        for p in plan or []:
            if p.get("switch_id"):
                out.add(p["switch_id"])
            a = self.engine.actions.get(p.get("action"))
            for t in (a.source, a.target) if a else ():
                if t:
                    out.add(ids.transformer_id(t))
        return out

    def _assets_of(self, rec):
        over = {t for t, x in ((rec.before or {}).get("loading_by_tx") or {}).items() if x > rec.threshold}
        return sorted({ids.transformer_id(t) for t in over} | self._plan_assets(rec.plan))

    # ------------------------------------------------------------------ events
    def _event(self, kind, message, sim_time=None, rec_id=None, severity=None, assets=(), switched=None, how=None,
               **data):
        """Record an event. `switched` is a list of switch operations that were EXECUTED (they also go to the
        actions table); a recommendation's proposed plan goes in **data instead."""
        e = {"seq": len(self.history) + 1, "wall_time": self._now(), "sim_time": sim_time, "type": kind,
             "severity": severity or SEVERITY.get(kind, "info"), "message": message, "rec_id": rec_id,
             "mode": self.mode, "configuration": list(self.active), "assets": sorted(set(assets)), **data}
        if switched is not None:
            e["plan"] = switched
        if self.store is not None:
            e["seq"] = self.store.add_event(self.name, e, e["assets"], plan=switched, how=how)
        self.history.append(e)
        return e

    @_locked
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
        self._save_state()

    # ------------------------------------------------------------------ recommendations
    def _checks(self, m, threshold, mitigation=False):
        """Constraint checks on a target state. For a mitigation the threshold cannot be met, so each
        transformer is checked against the hard limit and the check says so."""
        if not m:
            return []
        e = self.engine
        if mitigation:
            out = [{"check": f"{t} loading", "value": f"{x:.1f}%",
                    "limit": f"<= {e.hard_limit:g}% max ({threshold:g}% target not reachable)" if x > threshold
                    else f"<= {threshold:g}% target", "ok": x <= e.hard_limit} for t, x in m["loading_by_tx"].items()]
        else:
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

    @_locked
    def recommend(self, loads_kw, for_time, context="", threshold=None, use_mv=None):
        """Assess the loads and produce a Recommendation. In monitor mode, or while the emergency stop is
        active, nothing is recommended (returns None). In auto mode an actionable recommendation is applied."""
        if self.mode == "monitor" or self.estopped:
            return None
        e = self.engine
        thr = e.threshold if threshold is None else threshold
        r = e.evaluate_and_reroute(loads_kw, thr, use_mv)
        before = r["before"]
        conf_limit = thr
        cur_viol = r.get("violations_before", [])
        if self.active and before is not None:          # what is actually in force now
            cur = e.evaluate_configuration(loads_kw, self.active)
            before = cur["metrics"] or before
            cur_viol = cur["violations"] if cur["metrics"] else cur_viol
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
            best = r.get("best_partial_target")
            if best is not None and before is not None:
                if sorted(best) != sorted(self.active):
                    cand = e.evaluate_configuration(loads_kw, best)
                    if cand["ok"] and self._safer(cand["metrics"], before):
                        kind, target, after, conf_limit = "mitigate", list(best), cand["metrics"], e.hard_limit
                        still = [t for t, x in after["loading_by_tx"].items() if x > thr]
                        how = ("switch to: " + "; ".join(e.resolve(k).label for k in best) if best
                               else "return to the normal configuration")
                        removes = "; ".join(_describe(v) for v in cur_viol)
                        summary = (f"Mitigation, not a full fix: {how}. " + (f"Removes the hard-limit violation{'s' if len(cur_viol) > 1 else ''}: {removes}. " if removes else "")
                                   + f"Peak loading {before['max_loading']:.1f}% -> "
                                   f"{after['max_loading']:.1f}% with every hard limit respected; "
                                   f"{', '.join(still)} remain{'s' if len(still) == 1 else ''} above {thr:g}%.")
                elif self.active:
                    summary += " The best available configuration (a mitigation) is already in force."
            normal = r["before"]
            if kind == "no_safe_option" and self.active and normal["max_loading"] < before["max_loading"] - e.min_improvement:
                summary += (f" The normal configuration would be less overloaded (peak {normal['max_loading']:.1f}% "
                            f"instead of {before['max_loading']:.1f}%): consider resetting the switches.")
        # numbers are unique across every controller sharing the database (sandbox and simulation service)
        self._seq = self.store.next_rec_seq() if self.store is not None else self._seq + 1
        rec = Recommendation(
            rec_id=f"REC-{self._seq:04d}", created_at=self._now(), for_time=str(for_time), context=context,
            kind=kind, status="pending" if kind in ACTIONABLE else "info", summary=summary,
            threshold=thr, from_config=list(self.active), target=list(target),
            plan=e.switch_plan(self.active, target) if kind in ACTIONABLE else [],
            before=before, after=after,
            checks=self._checks(after, thr, mitigation=kind == "mitigate")
            if kind in ACTIONABLE + ("no_change",) else [],
            losses_before_kw=(before or {}).get("losses_kw"), losses_after_kw=(after or {}).get("losses_kw"),
            confidence=self._confidence((after or {}).get("loading_by_tx"), conf_limit, kind)
            if kind in ACTIONABLE + ("no_change",) else None,
            candidates=r["candidates"], binding=r["binding"], engine_status=r["status"], compute_s=r["compute_s"])
        self.last = rec
        self._supersede(f"replaced by {rec.rec_id}")          # a newer assessment replaces any older proposal
        self._save_rec(rec)
        assets = self._rec_assets[rec.rec_id] = self._assets_of(rec)
        if rec.actionable:
            self.pending = rec
            self._event("recommendation", f"{rec.rec_id}: {summary}", for_time, rec.rec_id,
                        severity="warning" if kind == "mitigate" else "info", assets=assets, plan=rec.plan)
            if self.mode == "auto":
                self._apply(rec, for_time, how="auto")
        elif kind == "no_safe_option":
            unsafe = bool(r.get("violations_before")) or not self.engine.is_feasible(before) if before else True
            self._event("no_safe_option", f"{rec.rec_id}: {summary} " + " ".join(rec.binding[:1]), for_time, rec.rec_id,
                        severity="critical" if unsafe else "warning", assets=assets)
        self._save_state()
        return rec

    def _safer(self, new, current):
        """True if `new` (which respects every hard limit) is safer than `current`: either `current` breaks a
        hard limit, or `new` lowers the peak loading by at least the engine's minimum improvement."""
        e = self.engine
        return not e.is_feasible(current) or new["max_loading"] < current["max_loading"] - e.min_improvement

    def _confidence(self, loading, limit, kind):
        c = self.risk.confidence(loading, self.error_by_tx, limit)
        if c and kind == "mitigate":
            c["basis"] = (f"probability that every transformer stays at or below the {limit:g}% hard limit, given "
                          "the forecast's test-period error for each transformer (the threshold cannot be met)")
        return c

    def _supersede(self, why):
        if self.pending is not None and self.pending.status == "pending":
            self._set_status(self.pending, "superseded")
            self._event("superseded", f"{self.pending.rec_id} cancelled: {why}", rec_id=self.pending.rec_id,
                        assets=self._rec_assets.get(self.pending.rec_id, ()))
        self.pending = None

    @_locked
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
        self._event("approved", f"{rec_id} approved by the operator", rec.for_time, rec_id,
                    assets=self._rec_assets.get(rec_id) or self._assets_of(rec))
        self._apply(rec, rec.for_time, how="operator")
        return rec

    @_locked
    def reject(self, rec_id, reason=""):
        rec = self.pending
        if rec is None or rec.rec_id != rec_id:
            raise DecisionError(f"{rec_id} is not the pending recommendation.")
        self._set_status(rec, "rejected")
        self.pending = None
        self._event("rejected", f"{rec_id} rejected by the operator" + (f": {reason}" if reason else ""), rec.for_time,
                    rec_id, assets=self._rec_assets.get(rec_id) or self._assets_of(rec))

    def _apply(self, rec, sim_time, how):
        self._undo.append(list(self.active))
        self.active = list(rec.target)
        self._set_status(rec, "applied")
        self.pending = None
        self._event("applied", f"{rec.rec_id} applied ({how}): " + (", ".join(s["switch_id"] + " " + s["operation"]
                    for s in rec.plan) or "no switch operations"), sim_time, rec.rec_id,
                    assets=self._rec_assets.get(rec.rec_id) or self._assets_of(rec), switched=rec.plan, how=how)
        self._save_state()

    # ------------------------------------------------------------------ operator actions
    @_locked
    def set_configuration(self, keys, sim_time=None, reason="manual switching", how="manual"):
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
                                               or "no change"), sim_time, assets=self._plan_assets(plan),
                    switched=plan, how=how)
        self._save_state()
        return plan

    def reset_to_normal(self, sim_time=None):
        return self.set_configuration([], sim_time, reason="reset to normal configuration", how="reset")

    @_locked
    def rollback(self, sim_time=None):
        """Undo the most recent change of configuration (applied recommendation or manual switching)."""
        if not self._undo:
            raise DecisionError("Nothing to roll back.")
        previous = self._undo.pop()
        plan = self.engine.switch_plan(self.active, previous)
        self.active = previous
        self._supersede("rollback")
        self._event("rollback", "Rolled back to the previous configuration: "
                    + (", ".join(s["switch_id"] + " " + s["operation"] for s in plan) or "no change"), sim_time,
                    assets=self._plan_assets(plan), switched=plan, how="rollback")
        self._save_state()
        return plan

    @_locked
    def emergency_stop(self, sim_time=None):
        if self.estopped:
            return
        self.estopped = True
        self._supersede("emergency stop")
        old, self.mode = self.mode, "monitor"
        self._event("emergency_stop", f"EMERGENCY STOP: automation halted (was {MODES[old]}); "
                    "switches left as they are", sim_time)
        self._save_state()

    @_locked
    def clear_emergency_stop(self, sim_time=None):
        if not self.estopped:
            return
        self.estopped = False
        self._event("emergency_stop_cleared", "Emergency stop cleared; mode stays Monitor only until changed", sim_time)
        self._save_state()

    @_locked
    def cancel_pending(self, reason):
        """Cancel a pending proposal because its conditions changed (scenario, time jump, restart)."""
        self._supersede(reason)

    @_locked
    def note(self, kind, message, sim_time=None, severity=None):
        """Record an event that is not a decision (service start/stop, clock warnings)."""
        return self._event(kind, message, sim_time, severity=severity)

    @property
    def can_rollback(self):
        return bool(self._undo)

    @_locked
    def snapshot(self):
        """Plain-data state (for the database in Phase 3)."""
        return {"mode": self.mode, "active": list(self.active), "estopped": self.estopped, "undo": [list(u) for u in self._undo],
                "pending": self.pending.to_dict() if self.pending else None, "seq": self._seq, "history": list(self.history)}
