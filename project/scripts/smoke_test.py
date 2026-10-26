import opendssdirect as dss

dss.Command("Clear")
dss.Command("New Circuit.Test basekv=11 pu=1.0 phases=3 bus1=SUB MVAsc3=500 MVAsc1=500")
dss.Command("New Line.L1 bus1=SUB bus2=B1 length=1 units=km")
dss.Command("New Load.LD1 bus1=B1 phases=3 kV=11 kW=500 pf=0.95")
dss.Command("Set voltagebases=[11]")
dss.Command("Calcvoltagebases")
dss.Solution.Solve()
print("converged:", dss.Solution.Converged())
print(dict(zip(dss.Circuit.AllBusNames(), dss.Circuit.AllBusMagPu())))