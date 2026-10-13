import sys
print("Python", sys.version.split()[0])
for name in ["opendssdirect", "xgboost", "sklearn", "pandas", "pyarrow",
             "streamlit", "plotly", "networkx", "matplotlib"]:
    try:
        __import__(name)
        print("OK  ", name)
    except Exception as e:
        print("FAIL", name, "->", e)