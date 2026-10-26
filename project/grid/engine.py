"""Grid engine: OpenDSS power flow, switching actions and the overload-rerouting search.

Everything about the network (assets, switches, the movable-load scheme, MV transfers) comes from
config/grid.yaml, and every limit from config/operations.yaml.

    engine = GridEngine()                          # or use the module-level functions below
    result = engine.evaluate_and_reroute({"T1": 420.0, ...})

The search (unchanged in its logic from the first version, so the published results still reproduce):
  1. Solve the base case. If no transformer is above the threshold and no limit is violated: no action.
  2. Greedy rounds: try every single switching action on top of those already chosen, keep the ones whose
     power flow satisfies every limit, and pick the one with the lowest peak transformer loading
     (lower losses breaks a tie). If no single action improves the peak, try every pair.
  3. Stop when the peak is at or below the threshold ("rerouted"), or when nothing improves it
     ("no_safe_option": the switches are restored and the binding constraint is named).

Every candidate is recorded with its outcome and, if rejected, the reason, for the operator's audit.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import opendssdirect as dss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config  # noqa: E402
from core.logging_setup import get_logger  # noqa: E402
from core.paths import project_path  # noqa: E402

logger = get_logger(__name__)


@dataclass(frozen=True)
class Action:
    """One switching action: a sequence of switch operations, executed in order (break-before-make)."""
    key: str                    # "T1>T2" (LV block move) or "MV1" (MV transfer)
    kind: str                   # "lv" or "mv"
    label: str
    ops: tuple                  # ((switch name, "open" | "close"), ...)
    group: str                  # at most one action per group can be active
    source: str = None          # LV: transformer whose block b moves
    target: str = None          # LV: transformer that receives it

    def revert_ops(self):
        return tuple((sw, "close" if op == "open" else "open") for sw, op in reversed(self.ops))


def _bus(name):
    return name.split(".")[0].lower()


class GridEngine:
    def __init__(self, grid_cfg=None, ops_cfg=None):
        g = grid_cfg if grid_cfg is not None else config.grid()
        o = ops_cfg if ops_cfg is not None else config.operations()
        self.grid, self.ops_cfg = g, o
        self.network = project_path(g["network_file"])
        self._lines = [ln.strip() for ln in Path(self.network).read_text().splitlines()]
        self.tx = list(g["transformers"])
        self.movable_fraction = g["load_split"]["movable_fraction"]
        self.load_names = {t: (x["fixed_load"], x["movable_load"]) for t, x in g["transformers"].items()}
        self.power_factor = {t: g["load_categories"][x["category"]]["power_factor"]
                             for t, x in g["transformers"].items()}
        lim, rr = o["limits"], o["rerouting"]
        self.threshold = lim["overload_threshold_pct"]
        self.hard_limit = lim["transformer_hard_limit_pct"]
        self.v_min, self.v_max = lim["voltage_min_pu"], lim["voltage_max_pu"]
        self.line_limit = lim["line_limit_pct"]
        self.min_improvement = rr["min_improvement_pct"]
        self.tie_tolerance = rr["tie_tolerance_pct"]
        self.use_mv = rr["use_mv_transfers"]
        self.switch_ids = {s: sw["id"] for s, sw in g["switches"].items()}
        self.normal_state = {s: sw["normal"] for s, sw in g["switches"].items()}

        self.actions = {}
        for t, m in config.move_options(g):
            a = Action(f"{t}>{m['to']}", "lv", f"move {t} block b -> {m['to']}",
                       ((m["home"], "open"), (m["alt"], "close")), group=t, source=t, target=m["to"])
            self.actions[a.key] = a
        for name, mv in (g.get("mv_transfers") or {}).items():
            a = Action(name, "mv", f"MV transfer {name}: {mv['description']}",
                       ((mv["open"], "open"), (mv["close"], "close")), group=f"mv:{mv['close']}")
            self.actions[a.key] = a
        self.default_move = {}                       # transformer -> its first LV option (old API: moves by name)
        for a in self.actions.values():
            if a.kind == "lv":
                self.default_move.setdefault(a.source, a.key)
        self._active = []                            # action keys currently applied in OpenDSS
        self._topology = None

    # ------------------------------------------------------------------ network and loads
    def load_network(self):
        """Send the .dss file to OpenDSS line by line (avoids Redirect path problems); normal switch state."""
        dss.Command("Clear")
        for line in self._lines:
            if line and not line.startswith("!") and line.lower() != "clear":
                dss.Command(line)
        for t, names in self.load_names.items():
            for ld in names:
                dss.Loads.Name(ld)
                dss.Loads.PF(self.power_factor[t])
        self._active = []
        if self._topology is None:
            self._topology = self._read_topology()

    def set_loads(self, kw_by_tx):
        """kw_by_tx: {"T1": 300.0, ...}. Splits each into block a (fixed) and block b (movable)."""
        for t, kw in kw_by_tx.items():
            fixed, movable = self.load_names[t]
            for load, share in ((fixed, 1 - self.movable_fraction), (movable, self.movable_fraction)):
                dss.Loads.Name(load)
                dss.Loads.kW(kw * share)

    @staticmethod
    def solve():
        dss.Solution.Solve()
        return dss.Solution.Converged()

    # ------------------------------------------------------------------ measurements
    @staticmethod
    def node_voltages_pu():
        """{bus: [per-unit magnitude of each phase node]}. AllBusMagPu has one value per NODE, in the order
        of AllNodeNames (an earlier version zipped it with AllBusNames, so only 11 kV buses were checked)."""
        out = {}
        for node, v in zip(dss.Circuit.AllNodeNames(), dss.Circuit.AllBusMagPu()):
            out.setdefault(_bus(node), []).append(v)
        return out

    def bus_voltages_pu(self):
        """{bus: lowest phase voltage in pu}."""
        return {b: min(v) for b, v in self.node_voltages_pu().items()}

    @staticmethod
    def transformer_power():
        """{T: {kw, kvar, kva, loading_pct}} at each transformer's primary terminal."""
        out = {}
        for name in dss.Transformers.AllNames():
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            p = dss.CktElement.Powers()[:6]          # terminal 1, 3 phases (kW, kvar pairs)
            kw, kvar = sum(p[0::2]), sum(p[1::2])
            kva = (kw ** 2 + kvar ** 2) ** 0.5
            dss.Transformers.Name(name)
            out[name.upper()] = {"kw": kw, "kvar": kvar, "kva": kva, "loading_pct": 100 * kva / dss.Transformers.kVA()}
        return out

    def transformer_loading(self):
        """Loading % = apparent power at primary terminal / kVA rating."""
        return {t: x["loading_pct"] for t, x in self.transformer_power().items()}

    @staticmethod
    def line_loading():
        """{LINE: % of normal ampacity} for every line that is not a switch."""
        out = {}
        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            if dss.Lines.IsSwitch():
                continue
            dss.Circuit.SetActiveElement(f"Line.{name}")
            i = max(dss.CktElement.CurrentsMagAng()[0:6:2])
            out[name.upper()] = 100 * i / dss.Lines.NormAmps()
        return out

    @staticmethod
    def losses_kw():
        return dss.Circuit.Losses()[0] / 1000

    def metrics(self):
        tx = self.transformer_loading()
        mv = {b: x for b, x in self.bus_voltages_pu().items() if b != "sub"}
        return {"max_loading": max(tx.values()), "min_voltage_pu": min(mv.values()),
                "loading_by_tx": tx, "max_line_loading": max(self.line_loading().values())}

    def measure(self):
        """Everything the search needs about the solved state, including where each extreme occurs."""
        tx = self.transformer_loading()
        nodes = {b: v for b, v in self.node_voltages_pu().items() if b != "sub"}
        vmin_bus = min(nodes, key=lambda b: min(nodes[b]))
        vmax_bus = max(nodes, key=lambda b: max(nodes[b]))
        lines = self.line_loading()
        line_max = max(lines, key=lines.get)
        tx_max = max(tx, key=tx.get)
        return {"max_loading": tx[tx_max], "max_loading_tx": tx_max, "loading_by_tx": tx,
                "min_voltage_pu": min(nodes[vmin_bus]), "min_voltage_bus": vmin_bus,
                "max_voltage_pu": max(nodes[vmax_bus]), "max_voltage_bus": vmax_bus,
                "max_line_loading": lines[line_max], "max_line": line_max, "losses_kw": self.losses_kw()}

    def full_metrics(self):
        return self.measure()

    def detail(self):
        """Per-element results for the operator: transformer P/Q/S, feeder KPIs, bus voltages, line loading."""
        txp = self.transformer_power()
        volts = self.bus_voltages_pu()
        lines = self.line_loading()
        graph = self._live_graph()
        feeders = {}
        for name, fd in self.grid["feeders"].items():
            dss.Circuit.SetActiveElement(f"Line.{fd['head_line']}")
            p = dss.CktElement.Powers()[:6]
            amps = max(dss.CktElement.CurrentsMagAng()[0:6:2])
            dss.Lines.Name(fd["head_line"])
            kw, kvar = sum(p[0::2]), sum(p[1::2])
            head = _bus(dss.Lines.Bus2())
            g = graph.copy()
            g.remove_node("sub")
            buses = nx.node_connected_component(g, head) if head in g else set()
            served = [t for t, x in self.grid["transformers"].items() if x["mv_bus"].lower() in buses]
            feeders[name] = {"id": fd["id"], "kw": round(kw, 1), "kvar": round(kvar, 1),
                             "kva": round((kw ** 2 + kvar ** 2) ** 0.5, 1), "current_a": round(amps, 1),
                             "loading_pct": round(100 * amps / dss.Lines.NormAmps(), 1),
                             "min_voltage_pu": round(min(volts[b] for b in buses), 4) if buses else None,
                             "transformers": served}
        return {"transformers": {t: {k: round(v, 1) for k, v in x.items()} for t, x in txp.items()},
                "feeders": feeders,
                "bus_voltages_pu": {b: round(v, 4) for b, v in volts.items()},
                "line_loading_pct": {k: round(v, 1) for k, v in lines.items()}}

    # ------------------------------------------------------------------ limits
    def is_feasible(self, m):
        """True if a solved state satisfies every hard limit (voltage band, line rating, transformer limit)."""
        return (self.v_min <= m["min_voltage_pu"] and m["max_voltage_pu"] <= self.v_max
                and m["max_line_loading"] <= self.line_limit and m["max_loading"] <= self.hard_limit)

    def violations(self, m, present=False):
        """Every hard-limit violation in a measured state. Texts read "would ..." for a candidate state and
        "is ..." for the present state (present=True)."""
        verb = "is" if present else "would be"
        out = []

        def add(constraint, element, value, limit, text):
            out.append({"constraint": constraint, "element": element, "value": value, "limit": limit, "text": text})

        for t, pct in sorted(m["loading_by_tx"].items(), key=lambda kv: -kv[1]):
            if pct > self.hard_limit:
                add("transformer", t, round(pct, 1), self.hard_limit,
                    f"{t} {verb} at {pct:.1f}% (> {self.hard_limit:g}% limit)")
        if m["min_voltage_pu"] < self.v_min:
            add("voltage_low", m["min_voltage_bus"], round(m["min_voltage_pu"], 4), self.v_min,
                f"bus {m['min_voltage_bus'].upper()} {verb} at {m['min_voltage_pu']:.4f} pu (< {self.v_min:g} pu)")
        if m["max_voltage_pu"] > self.v_max:
            add("voltage_high", m["max_voltage_bus"], round(m["max_voltage_pu"], 4), self.v_max,
                f"bus {m['max_voltage_bus'].upper()} {verb} at {m['max_voltage_pu']:.4f} pu (> {self.v_max:g} pu)")
        if m["max_line_loading"] > self.line_limit:
            add("line", m["max_line"], round(m["max_line_loading"], 1), self.line_limit,
                f"line {m['max_line']} {verb} at {m['max_line_loading']:.1f}% of its rating (> {self.line_limit:g}%)")
        return out

    # ------------------------------------------------------------------ switching and topology
    def resolve(self, key):
        """Accept an action key ("T1>T2", "MV1") or, for the old API, a transformer name ("T1")."""
        if key in self.actions:
            return self.actions[key]
        if key in self.default_move:
            return self.actions[self.default_move[key]]
        raise KeyError(f"unknown switching action {key!r}")

    @staticmethod
    def _operate(switch, op):
        dss.Command(f"{'Open' if op == 'open' else 'Close'} Line.{switch} term=1")

    def apply(self, keys):
        for k in keys:
            a = self.resolve(k)
            for sw, op in a.ops:
                self._operate(sw, op)
            self._active.append(a.key)

    def revert(self, keys):
        for k in keys:
            a = self.resolve(k)
            for sw, op in a.revert_ops():
                self._operate(sw, op)
            self._active.remove(a.key)

    def switch_state(self, keys):
        """{switch: "open" | "closed"} after applying actions to the normal configuration."""
        state = dict(self.normal_state)
        for k in keys:
            for sw, op in self.resolve(k).ops:
                state[sw] = "open" if op == "open" else "closed"
        return state

    def _read_topology(self):
        """Edges of the network (lines and transformers) and the bus each load sits on, from OpenDSS."""
        edges = []
        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            edges.append((_bus(dss.Lines.Bus1()), _bus(dss.Lines.Bus2()), name.upper()))
        for name in dss.Transformers.AllNames():
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            b = dss.CktElement.BusNames()
            edges.append((_bus(b[0]), _bus(b[1]), None))
        loads = {}
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            dss.Circuit.SetActiveElement(f"Load.{name}")
            loads[name.upper()] = _bus(dss.CktElement.BusNames()[0])
        return {"edges": edges, "loads": loads, "buses": {_bus(b) for b in dss.Circuit.AllBusNames()}}

    def _graph_for(self, state):
        g = nx.MultiGraph()
        g.add_nodes_from(self._topology["buses"])
        for u, v, sw in self._topology["edges"]:
            if sw is None or state.get(sw, "closed") == "closed":
                g.add_edge(u, v)
        return g

    def _live_graph(self):
        return nx.Graph(self._graph_for(self.switch_state(self._active)))

    def check_radial(self, keys):
        """(True, "") if the configuration is radial and supplies every load, else (False, reason)."""
        return self.check_radial_state(self.switch_state(keys))

    def check_radial_state(self, state):
        """Same check for an explicit {switch: "open"/"closed"} state."""
        if self._topology is None:
            self.load_network()
        g = self._graph_for(state)
        simple = nx.Graph(g)
        if g.number_of_edges() != simple.number_of_edges() or not nx.is_forest(simple):
            loop = nx.cycle_basis(simple)
            where = ", ".join(sorted(b.upper() for b in loop[0])) if loop else "parallel elements"
            return False, f"would close a loop ({where})"
        supplied = nx.node_connected_component(simple, "sub")
        cut = sorted(ld for ld, b in self._topology["loads"].items() if b not in supplied)
        if cut:
            return False, f"would leave {', '.join(cut)} unsupplied"
        return True, ""

    def switch_plan(self, from_keys, to_keys):
        """Ordered switch operations that take the network from one configuration to another:
        first undo the actions no longer wanted (last applied first), then apply the new ones."""
        from_keys = [self.resolve(k).key for k in from_keys]
        to_keys = [self.resolve(k).key for k in to_keys]
        steps = []
        for k in reversed([k for k in from_keys if k not in to_keys]):
            steps += [(k, sw, op) for sw, op in self.actions[k].revert_ops()]
        for k in [k for k in to_keys if k not in from_keys]:
            steps += [(k, sw, op) for sw, op in self.actions[k].ops]
        return [{"step": i + 1, "switch": sw, "switch_id": self.switch_ids.get(sw), "operation": op, "action": k}
                for i, (k, sw, op) in enumerate(steps)]

    # ------------------------------------------------------------------ evaluation of one configuration
    def evaluate_configuration(self, loads_kw, keys=()):
        """Solve the given loads with the given actions applied (no optimisation).
        Returns {"ok", "radial", "converged", "reason", "metrics", "violations"}; metrics is None when the
        configuration is not radial or the power flow does not converge."""
        keys = [self.resolve(k).key for k in keys]
        self.load_network()
        self.set_loads(loads_kw)
        radial, why = self.check_radial(keys)
        if not radial:
            return {"ok": False, "radial": False, "converged": None, "reason": why, "metrics": None, "violations": []}
        self.apply(keys)
        if not self.solve():
            return {"ok": False, "radial": True, "converged": False, "reason": "power flow did not converge",
                    "metrics": None, "violations": []}
        m = self.measure()
        viol = self.violations(m)
        return {"ok": not viol, "radial": True, "converged": True, "reason": viol[0]["text"] if viol else "",
                "metrics": {**self._pack(m), **self.detail()}, "violations": viol}

    def evaluate_state(self, loads_kw, moves):
        """Old API: metrics for the loads with the given actions applied, or None (not radial / no convergence)."""
        r = self.evaluate_configuration(loads_kw, moves)
        return r["metrics"]

    # ------------------------------------------------------------------ the rerouting search
    @staticmethod
    def _pack(m):
        return {"max_loading": round(m["max_loading"], 1),
                "min_voltage_pu": round(m["min_voltage_pu"], 4),
                "max_line_loading": round(m["max_line_loading"], 1),
                "loading_by_tx": {k: round(x, 1) for k, x in m["loading_by_tx"].items()},
                "max_voltage_pu": round(m["max_voltage_pu"], 4),
                "min_voltage_bus": m["min_voltage_bus"].upper(),
                "losses_kw": round(m["losses_kw"], 2)}

    def _try(self, active, cand):
        """Evaluate candidate actions on top of the active ones. Always restores the network.
        Returns (measured state or None, violations, reason if rejected)."""
        radial, why = self.check_radial(active + cand)
        if not radial:
            return None, [], why
        self.apply(cand)
        try:
            if not self.solve():
                return None, [], "power flow did not converge"
            m = self.measure()
            viol = self.violations(m)
            return m, viol, (viol[0]["text"] if viol else "")
        finally:
            self.revert(cand)
            self.solve()

    def _better(self, m, best):
        """Lower peak loading wins; peaks within the tie tolerance are decided by lower losses."""
        if best is None:
            return True
        if m["max_loading"] < best["max_loading"] - self.tie_tolerance:
            return True
        return abs(m["max_loading"] - best["max_loading"]) <= self.tie_tolerance and m["losses_kw"] < best["losses_kw"]

    def evaluate_and_reroute(self, loads_kw, threshold=None, use_mv=None):
        """loads_kw: {"T1": kW, ...}. Returns a dict with:
        status ("no_action" | "rerouted" | "no_safe_option"), action (labels), target (action keys),
        plan (ordered switch operations), before/after (metrics), candidates (every evaluated option with its
        outcome and reason), binding (why no safe option exists), violations_before, overloaded, log, compute_s."""
        threshold = self.threshold if threshold is None else threshold
        use_mv = self.use_mv if use_mv is None else use_mv
        allowed = [a for a in self.actions.values() if a.kind == "lv" or use_mv]
        t0 = time.perf_counter()
        log, records = [], []

        def result(status, **kw):
            base = {"status": status, "action": [], "target": [], "plan": [], "candidates": records, "binding": [],
                    "violations_before": [], "overloaded": [], "log": log,
                    "compute_s": round(time.perf_counter() - t0, 3)}
            return {**base, **kw}

        self.load_network()
        self.set_loads(loads_kw)
        if not self.solve():
            logger.warning("base case did not converge for loads %s", loads_kw)
            log.append("Base case did not converge: load level is unrealistic.")
            return result("no_safe_option", before=None, after=None, best_partial=[],
                          binding=["The power flow does not converge at this load level, so no switching state can be checked."])
        before = self.measure()
        before_detail = self.detail()
        viol_before = self.violations(before, present=True)
        other_viol = [v for v in viol_before if v["constraint"] != "transformer"]
        log.append(f"Baseline: max loading {before['max_loading']:.1f}% "
                   f"(threshold {threshold:.0f}%), min V {before['min_voltage_pu']:.4f} pu")
        overloaded = [t for t, x in before["loading_by_tx"].items() if x > threshold]
        packed_before = {**self._pack(before), **before_detail}

        if not overloaded and not other_viol:
            return result("no_action", before=packed_before, after=packed_before,
                          log=log + ["No transformer above threshold."])
        if overloaded:
            log.append("Overloaded: " + ", ".join(f"{t} ({before['loading_by_tx'][t]:.1f}%)" for t in overloaded))
        if other_viol:
            log.append("Limit violations: " + "; ".join(v["text"] for v in other_viol))

        def ok(state, state_other_viol):
            return state["max_loading"] <= threshold and not state_other_viol

        def improves(best, current, current_other_viol):
            return best is not None and (best["max_loading"] < current["max_loading"] - self.min_improvement
                                         or bool(current_other_viol))

        already = {(v["constraint"], v["element"]) for v in viol_before}
        active, current, cur_other = [], before, other_viol
        tested = feasible = 0
        n_groups = len({a.group for a in allowed})
        for rnd in range(1, n_groups + 1):
            if ok(current, cur_other):
                break
            groups = {self.actions[k].group for k in active}
            free = [a for a in allowed if a.key not in active and a.group not in groups]
            best, best_m, round_recs = None, None, []

            def consider(cands, kind):
                nonlocal best, best_m, tested, feasible
                for cand in cands:
                    tested += 1
                    m, viol, why = self._try(active, [a.key for a in cand])
                    for v in viol:                       # say so when the problem was there before switching
                        if (v["constraint"], v["element"]) in already:
                            v["text"] = v["text"].replace(" would be at ", " would still be at ")
                    viol.sort(key=lambda v: (v["constraint"], v["element"]) in already)   # new problems first
                    why = viol[0]["text"] if viol else why
                    rec = {"id": f"C{len(records) + 1}", "round": rnd, "kind": kind,
                           "actions": [a.key for a in cand], "label": " + ".join(a.label for a in cand),
                           "feasible": m is not None and not viol, "reason": why,
                           "max_loading": round(m["max_loading"], 1) if m else None,
                           "max_loading_tx": m["max_loading_tx"] if m else None,
                           "loading_by_tx": {k: round(x, 1) for k, x in m["loading_by_tx"].items()} if m else None,
                           "min_voltage_pu": round(m["min_voltage_pu"], 4) if m else None,
                           "max_line_loading": round(m["max_line_loading"], 1) if m else None,
                           "losses_kw": round(m["losses_kw"], 2) if m else None,
                           "violations": [v["text"] for v in viol], "_m": m}
                    records.append(rec)
                    round_recs.append(rec)
                    if not rec["feasible"]:
                        continue
                    feasible += 1
                    if self._better(m, best_m):
                        best, best_m = cand, m

            consider([[a] for a in free], "single")
            if not improves(best_m, current, cur_other):
                # no single action helps: try pairs (actions from different groups)
                consider([[a, b] for i, a in enumerate(free) for b in free[i + 1:] if a.group != b.group], "pair")
            if not improves(best_m, current, cur_other):
                log.append(f"Round {rnd}: no improving feasible action found.")
                self._label_round(round_recs, None, current, threshold, bool(cur_other))
                break
            self._label_round(round_recs, [a.key for a in best], current, threshold, bool(cur_other))
            self.apply([a.key for a in best])
            self.solve()
            active = list(self._active)
            current, cur_other = best_m, []
            log.append(f"Round {rnd}: {' + '.join(a.label for a in best)} -> max loading {current['max_loading']:.1f}%")

        log.append(f"Tested {tested} candidates, {feasible} feasible.")
        for r in records:
            r.pop("_m", None)
        labels = [self.actions[k].label for k in self._active]
        if ok(current, cur_other):
            target = list(self._active)
            after = {**self._pack(current), **self.detail()}
            res = result("rerouted", action=labels, target=target, plan=self.switch_plan([], target),
                         before=packed_before, after=after, overloaded=overloaded, violations_before=viol_before)
            logger.debug("rerouted %s: %.1f%% -> %.1f%% in %.3f s", labels, before["max_loading"],
                         current["max_loading"], res["compute_s"])
            return res
        # not fixable: restore the original configuration and report the best partial effort
        partial = list(self._active)
        self.revert(list(reversed(partial)))
        self.solve()
        binding = self._binding(before, current, records, threshold, overloaded, other_viol)
        log.append("CRITICAL: no safe reconfiguration brings all transformers under the threshold."
                   + (" Best partial: " + "; ".join(labels) if labels else ""))
        logger.debug("no safe option: peak %.1f%%, overloaded %s, best partial %s",
                     before["max_loading"], overloaded, labels)
        return result("no_safe_option", best_partial=labels, before=packed_before, after=self._pack(current),
                      overloaded=overloaded, violations_before=viol_before, binding=binding)

    @staticmethod
    def _label_round(recs, chosen, current, threshold, fixing_violation=False):
        """Give every candidate of a round its status and a one-line reason."""
        for r in recs:
            if not r["feasible"]:
                r["status"] = "rejected"
                continue
            peak = f"peak {r['max_loading']:.1f}% on {r['max_loading_tx']}, losses {r['losses_kw']:.2f} kW"
            if chosen is not None and r["actions"] == chosen:
                r["status"], r["reason"] = "selected", f"selected: {peak}"
                continue
            r["status"] = "feasible"
            better = "; another option was better" if chosen else ""
            if fixing_violation:
                r["reason"] = f"safe, removes the limit violation: {peak}{better}"
            elif r["max_loading"] >= current["max_loading"] - 0.05:
                r["reason"] = f"safe, but the peak stays at {r['max_loading']:.1f}% on {r['max_loading_tx']}"
            elif r["max_loading"] > threshold:
                r["reason"] = f"safe, lowers the {peak} (still above {threshold:g}%){better}"
            else:
                r["reason"] = f"safe: {peak}{better}"

    def _binding(self, before, current, records, threshold, overloaded, other_viol):
        """Plain-language reasons why no safe configuration exists: for each transformer still above the
        threshold, what happens when its own block b is moved."""
        out = []
        still = [t for t, x in current["loading_by_tx"].items() if x > threshold] or overloaded
        first_round = [r for r in records if r["round"] == 1 and r["kind"] == "single"]
        pct = round(100 * self.movable_fraction)
        for t in sorted(still, key=lambda t: -before["loading_by_tx"][t]):
            was = f"{t} ({before['loading_by_tx'][t]:.1f}%)"
            own = [r for r in first_round if self.actions[r["actions"][0]].source == t]
            if not own:
                out.append(f"{was} has no switchable outlet for its load.")
                continue
            for r in own:
                to = self.actions[r["actions"][0]].target
                if r["loading_by_tx"] is None:                       # not radial or did not converge
                    out.append(f"{was}: moving its block b to {to} is not possible: {r['reason']}.")
                    continue
                t_after, to_after = r["loading_by_tx"][t], r["loading_by_tx"][to]
                if t_after > self.hard_limit:
                    out.append(f"{was}: even with its block b moved to {to} it would be at {t_after:.1f}%, above the "
                               f"{self.hard_limit:g}% limit (only {pct}% of its load is movable).")
                elif to_after > self.hard_limit:
                    out.append(f"{was}: moving its block b to {to} would overload {to} to {to_after:.1f}% "
                               f"(> {self.hard_limit:g}% limit).")
                elif any(v.startswith(("bus", "line")) for v in r["violations"]):
                    out.append(f"{was}: moving its block b to {to} is not safe: "
                               f"{next(v for v in r['violations'] if v.startswith(('bus', 'line')))}.")
                elif t_after > threshold:
                    out.append(f"{was}: moving its block b to {to} only brings it to {t_after:.1f}%, still above "
                               f"{threshold:g}% (only {pct}% of its load is movable).")
                else:
                    blockers = sorted({v.split(" ")[0] for v in r["violations"]} - {t, to})
                    out.append(f"{was} could be relieved by moving its block b to {to} ({t}: {t_after:.1f}%, "
                               f"{to}: {to_after:.1f}%), but no configuration is safe while "
                               f"{', '.join(blockers) or 'another transformer'} stays above the limit.")
        if not any(r["feasible"] for r in records):
            for v in other_viol:
                out.append(f"{v['text']}, and no switching option removes it.")
        return out


# ======================================================================
# Module-level API (the original functions), backed by one default engine
# ======================================================================
_default = GridEngine()

NETWORK = _default.network
TX = _default.tx
MOVABLE_FRACTION = _default.movable_fraction
THRESHOLD = _default.threshold
HARD_LIMIT = _default.hard_limit
V_MIN, V_MAX = _default.v_min, _default.v_max
LINE_LIMIT = _default.line_limit
MIN_IMPROVEMENT = _default.min_improvement
ACTIONS = _default.actions
# transformer -> (home switch, alt switch, neighbour) of its first LV option (original structure)
MOVES = {t: (a.ops[0][0], a.ops[1][0], a.target) for t, a in
         ((t, _default.actions[k]) for t, k in _default.default_move.items())}

load_network = _default.load_network
set_loads = _default.set_loads
solve = _default.solve
node_voltages_pu = _default.node_voltages_pu
bus_voltages_pu = _default.bus_voltages_pu
transformer_loading = _default.transformer_loading
line_loading = _default.line_loading
metrics = _default.metrics
full_metrics = _default.full_metrics
is_feasible = _default.is_feasible
evaluate_and_reroute = _default.evaluate_and_reroute
evaluate_state = _default.evaluate_state
evaluate_configuration = _default.evaluate_configuration
switch_plan = _default.switch_plan
check_radial = _default.check_radial


def move_label(tx):
    return _default.resolve(tx).label


def open_switch(name):
    dss.Command(f"Open Line.{name} term=1")


def close_switch(name):
    dss.Command(f"Close Line.{name} term=1")


def apply_moves(keys):
    _default.apply(keys)


def revert_moves(keys):
    _default.revert(keys)
