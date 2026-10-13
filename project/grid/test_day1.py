import engine as e

e.load_network()
base = {t: 300.0 for t in e.TX}

for mult in (1.0, 1.3, 1.5, 1.6):
    e.set_loads({t: kw * mult for t, kw in base.items()})
    ok = e.solve()
    m = e.metrics()
    print(f"x{mult}: converged={ok}  max_loading={m['max_loading']:.1f}%  "
          f"min_V={m['min_voltage_pu']:.4f}  max_line={m['max_line_loading']:.1f}%")

# Overload only T1, then move its block b to T2
e.set_loads({**base, "T1": 420.0})
e.solve()
print("T1 overloaded:", {k: round(v, 1) for k, v in e.metrics()["loading_by_tx"].items()})
e.open_switch("SW1H")
e.close_switch("SW1A")
e.solve()
print("after SW1A:   ", {k: round(v, 1) for k, v in e.metrics()["loading_by_tx"].items()})