import sys
print("Python", sys.version.split()[0])
for name in ["opendssdirect", "xgboost", "sklearn", "pandas", "pyarrow", "yaml", "sqlalchemy",
             "streamlit", "plotly", "networkx", "matplotlib", "psutil", "pytest"]:
    try:
        __import__(name)
        print("OK  ", name)
    except Exception as e:
        print("FAIL", name, "->", e)
# Minimum versions the code relies on (see requirements.txt)
try:
    import sklearn
    import xgboost
    for mod, need in ((sklearn, (1, 4)), (xgboost, (2, 0))):
        have = tuple(int(x) for x in mod.__version__.split(".")[:2])
        print("OK  " if have >= need else "FAIL", f"{mod.__name__} {mod.__version__} (needs >= {need[0]}.{need[1]})")
except ImportError:
    pass
