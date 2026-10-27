"""The operator dashboard: one multi-page app (gap items G3 to G9).

    streamlit run app/app.py

Live network pages (they show what the simulation service, sim/service.py, is doing; they read the database
and their buttons only send commands to the service):
    A Overview                city counts, capacity, load, alerts, clock and speed, season, last power flow
    B Network                 single-line diagram; select a transformer or feeder to inspect it
    C Transformers            sortable table; history of the selected transformer
    D Forecasting             accuracy over time by transformer and model version; forecast against actual
    E Alerts and decisions    alerts, the pending proposal, every recommendation with its candidates
    F Controls                service, clock, scenario, mode, approval policy, threshold, emergency stop
    G Audit log               events, recommendations, switch operations, alerts and commands, with filters
Sandbox:
    What-if replay            any hour of the test period with its own controller (separate from the live network)
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402

CITY = config.city_arg()                 # streamlit run app/app.py -- --city demo   (a synthetic city)
st.set_page_config(page_title="Feeder digital twin", layout="wide")

VIEWS = Path(__file__).resolve().parent / "views"
pages = {
    "Live network": [
        st.Page(VIEWS / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(VIEWS / "network.py", title="Network", icon=":material/hub:"),
        st.Page(VIEWS / "transformers.py", title="Transformers", icon=":material/table_chart:"),
        st.Page(VIEWS / "forecasting.py", title="Forecasting", icon=":material/insights:"),
        st.Page(VIEWS / "decisions.py", title="Alerts and decisions", icon=":material/rule:"),
        st.Page(VIEWS / "controls.py", title="Controls", icon=":material/tune:"),
        st.Page(VIEWS / "audit.py", title="Audit log", icon=":material/history:"),
    ],
    "Sandbox": [
        st.Page(VIEWS / "replay.py", title="What-if replay", icon=":material/science:"),
    ],
}
if config.simulation().get("city"):     # the replay sandbox works on the eMARC forecasts only
    pages.pop("Sandbox")
st.navigation(pages).run()
