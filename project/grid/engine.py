"""Grid engine: OpenDSS helpers (Day 1). Rerouting algorithm is added on Day 2."""
from pathlib import Path
import opendssdirect as dss

NETWORK = Path(__file__).parent / "network.dss"
TX = ["T1", "T2", "T3", "T4", "T5", "T6"]
MOVABLE_FRACTION = 0.20          # block "b" = 20% of each transformer's load


def load_network(path=NETWORK):
    """Read the .dss file and send each line to OpenDSS (avoids Redirect path problems)."""
    dss.Command("Clear")
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("!") and line.lower() != "clear":
            dss.Command(line)


def set_loads(kw_by_tx: dict):
    """kw_by_tx: {"T1": 300.0, ...}. Splits each into block a (80%) and b (20%)."""
    for tx, kw in kw_by_tx.items():
        n = tx[1:]                                   # "T1" -> "1"
        for suffix, share in (("a", 1 - MOVABLE_FRACTION), ("b", MOVABLE_FRACTION)):
            dss.Loads.Name(f"LD{n}{suffix}")
            dss.Loads.kW(kw * share)


def solve():
    dss.Solution.Solve()
    return dss.Solution.Converged()


def bus_voltages_pu():
    return dict(zip(dss.Circuit.AllBusNames(), dss.Circuit.AllBusMagPu()))


def transformer_loading():
    """Loading % = apparent power at primary terminal / kVA rating."""
    out = {}
    for name in dss.Transformers.AllNames():
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        p = dss.CktElement.Powers()[:6]              # terminal 1, 3 phases (kW, kvar pairs)
        s = (sum(p[0::2]) ** 2 + sum(p[1::2]) ** 2) ** 0.5
        dss.Transformers.Name(name)
        out[name.upper()] = 100 * s / dss.Transformers.kVA()
    return out


def line_loading():
    out = {}
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        if dss.Lines.IsSwitch():
            continue
        dss.Circuit.SetActiveElement(f"Line.{name}")
        i = max(dss.CktElement.CurrentsMagAng()[0:6:2])
        out[name.upper()] = 100 * i / dss.Lines.NormAmps()
    return out


def metrics():
    tx = transformer_loading()
    v = bus_voltages_pu()
    mv = {b: x for b, x in v.items() if b.lower() != "sub"}
    return {
        "max_loading": max(tx.values()),
        "min_voltage_pu": min(mv.values()),
        "loading_by_tx": tx,
        "max_line_loading": max(line_loading().values()),
    }


def open_switch(name):
    dss.Command(f"Open Line.{name} term=1")


def close_switch(name):
    dss.Command(f"Close Line.{name} term=1")

    

# =====================================================================
# Day 2: overload evaluation and rerouting
# =====================================================================
import time

THRESHOLD = 85.0            # target max transformer loading (%)
HARD_LIMIT = 100.0          # never accept a state above this
V_MIN, V_MAX = 0.95, 1.05   # bus voltage limits (pu)
LINE_LIMIT = 100.0          # % of normamps

# Each movable block: (home switch, alt switch, neighbour it moves to)
MOVES = {
    "T1": ("SW1H", "SW1A", "T2"),
    "T2": ("SW2H", "SW2A", "T3"),
    "T3": ("SW3H", "SW3A", "T6"),
    "T4": ("SW4H", "SW4A", "T5"),
    "T5": ("SW5H", "SW5A", "T6"),
    "T6": ("SW6H", "SW6A", "T3"),
}


def move_label(tx):
    return f"move {tx} block b -> {MOVES[tx][2]}"


def apply_move(tx):
    home, alt, _ = MOVES[tx]
    open_switch(home)
    close_switch(alt)


def revert_move(tx):
    home, alt, _ = MOVES[tx]
    open_switch(alt)
    close_switch(home)


def apply_moves(txs):
    for tx in txs:
        apply_move(tx)


def revert_moves(txs):
    for tx in txs:
        revert_move(tx)


def full_metrics():
    m = metrics()
    v = bus_voltages_pu()
    m["max_voltage_pu"] = max(x for b, x in v.items() if b.lower() != "sub")
    return m


def is_feasible(m):
    """Constraint check for a solved state."""
    return (V_MIN <= m["min_voltage_pu"] and m["max_voltage_pu"] <= V_MAX
            and m["max_line_loading"] <= LINE_LIMIT
            and m["max_loading"] <= HARD_LIMIT)


def _try(active, candidate):
    """Apply candidate moves on top of `active`, solve, return metrics or None. Always restores."""
    apply_moves(candidate)
    try:
        if not solve():
            return None
        m = full_metrics()
        return m if is_feasible(m) else None
    finally:
        revert_moves(candidate)
        solve()


def _round(v):
    return {k: round(x, 1) for k, x in v.items()}


def _pack(m):
    return {"max_loading": round(m["max_loading"], 1),
            "min_voltage_pu": round(m["min_voltage_pu"], 4),
            "max_line_loading": round(m["max_line_loading"], 1),
            "loading_by_tx": _round(m["loading_by_tx"])}


def evaluate_and_reroute(loads_kw: dict, threshold: float = THRESHOLD) -> dict:
    """loads_kw: {"T1": kW, ... "T6": kW}. Returns status/action/before/after/log."""
    t0 = time.perf_counter()
    log = []
    load_network()
    set_loads(loads_kw)
    if not solve():
        return {"status": "no_safe_option", "action": [], "before": None, "after": None,
                "log": ["Base case did not converge: load level is unrealistic."],
                "compute_s": round(time.perf_counter() - t0, 3)}
    before = full_metrics()
    log.append(f"Baseline: max loading {before['max_loading']:.1f}% "
               f"(threshold {threshold:.0f}%), min V {before['min_voltage_pu']:.4f} pu")
    overloaded = [t for t, x in before["loading_by_tx"].items() if x > threshold]

    if not overloaded:
        return {"status": "no_action", "action": [], "before": _pack(before),
                "after": _pack(before), "log": log + ["No transformer above threshold."],
                "overloaded": [], "compute_s": round(time.perf_counter() - t0, 3)}
    log.append("Overloaded: " + ", ".join(f"{t} ({before['loading_by_tx'][t]:.1f}%)" for t in overloaded))

    active, current, tested, feasible = [], before, 0, 0
    # Greedy rounds: pick the single best move each round; fall back to pairs once.
    for rnd in range(1, len(MOVES) + 1):
        if current["max_loading"] <= threshold:
            break
        free = [t for t in MOVES if t not in active]
        best, best_m = None, None
        for tx in free:
            tested += 1
            m = _try(active, [tx])
            if m is None:
                continue
            feasible += 1
            if best_m is None or m["max_loading"] < best_m["max_loading"]:
                best, best_m = [tx], m
        if best_m is None or best_m["max_loading"] >= current["max_loading"] - 0.05:
            # no single move helps: try pairs
            for i, a in enumerate(free):
                for b in free[i + 1:]:
                    tested += 1
                    m = _try(active, [a, b])
                    if m is None:
                        continue
                    feasible += 1
                    if best_m is None or m["max_loading"] < best_m["max_loading"]:
                        best, best_m = [a, b], m
        if best_m is None or best_m["max_loading"] >= current["max_loading"] - 0.05:
            log.append(f"Round {rnd}: no improving feasible action found.")
            break
        apply_moves(best)
        solve()
        active += best
        current = best_m
        log.append(f"Round {rnd}: {' + '.join(move_label(t) for t in best)} "
                   f"-> max loading {current['max_loading']:.1f}%")

    log.append(f"Tested {tested} candidates, {feasible} feasible.")
    dt = round(time.perf_counter() - t0, 3)
    actions = [move_label(t) for t in active]
    if current["max_loading"] <= threshold:
        return {"status": "rerouted", "action": actions, "before": _pack(before),
                "after": _pack(current), "log": log, "overloaded": overloaded, "compute_s": dt}
    # not fixable: restore original topology, report best partial effort
    revert_moves(active)
    solve()
    log.append("CRITICAL: no safe reconfiguration brings all transformers under the threshold."
               + (" Best partial: " + "; ".join(actions) if actions else ""))
    return {"status": "no_safe_option", "action": [], "best_partial": actions,
            "before": _pack(before), "after": _pack(current), "log": log,
            "overloaded": overloaded, "compute_s": dt}