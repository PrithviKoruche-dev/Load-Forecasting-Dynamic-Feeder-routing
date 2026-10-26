"""Plot helpers for the operator dashboard (Plotly).

The drawing layout below is specific to the two-feeder, six-transformer network; colour bands come
from config/operations.yaml.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config  # noqa: E402

AMBER_FROM = config.operations()["dashboard"]["amber_from_pct"]

GREEN, AMBER, RED, GREY, ORANGE = "#2e9e5b", "#f0a30a", "#d64545", "#8a8f98", "#ff7a1a"

BUS = {"SUB": (0, 0), "A1": (1, 1), "A2": (2, 1), "A3": (3, 1), "B1": (1, -1), "B2": (2, -1), "B3": (3, -1)}
TX_POS = {"T1": (1, 2.4), "T2": (2, 2.4), "T3": (3, 2.4), "T4": (1, -2.4), "T5": (2, -2.4), "T6": (3, -2.4)}
TX_BUS = {t: tx["mv_bus"] for t, tx in config.grid()["transformers"].items()}
MV_LINES = [("SUB", "A1"), ("A1", "A2"), ("A2", "A3"), ("SUB", "B1"), ("B1", "B2"), ("B2", "B3")]
# LV tie paths: (key, polyline, label). Cross-feeder T3<->T6 goes round the right-hand side.
LV_TIES = {
    "T1": ([(1, 2.4), (2, 2.4)], "T1>T2"), "T2": ([(2, 2.4), (3, 2.4)], "T2>T3"),
    "T4": ([(1, -2.4), (2, -2.4)], "T4>T5"), "T5": ([(2, -2.4), (3, -2.4)], "T5>T6"),
    "T3": ([(3, 2.4), (3.6, 2.4), (3.6, -2.4), (3, -2.4)], "T3>T6"),
    "T6": ([(3, -2.4), (3.6, -2.4), (3.6, 2.4), (3, 2.4)], "T6>T3"),
}


def colour(pct, threshold):
    if pct > threshold:
        return RED
    return AMBER if pct >= AMBER_FROM else GREEN


def _line(fig, pts, color, width=2, dash="solid"):
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                             line=dict(color=color, width=width, dash=dash), hoverinfo="skip"))


def topology_figure(loading, moves, threshold):
    fig = go.Figure()
    for a, b in MV_LINES:                                    # 11 kV feeders
        _line(fig, [BUS[a], BUS[b]], "#5b6472", 3)
    _line(fig, [BUS["A3"], BUS["B3"]], GREY, 2, "dash")      # TIE1 (normally open)
    fig.add_annotation(x=3.0, y=0, text="TIE1 (open)", showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=10, color=GREY))
    for tx, bus in TX_BUS.items():                           # bus -> transformer
        _line(fig, [BUS[bus], TX_POS[tx]], "#5b6472", 1.5)
    done = set()
    for tx, (pts, label) in LV_TIES.items():                 # LV tie switches
        active = tx in moves
        _line(fig, pts, ORANGE if active else GREY, 4 if active else 1.5, "solid" if active else "dot")
        mx = (pts[0][0] + pts[1][0]) / 2 if len(pts) == 2 else 3.6
        my = pts[0][1] if len(pts) == 2 else (0.4 if tx == "T3" else -0.4)
        if active:
            fig.add_annotation(x=mx, y=my + (0.3 if len(pts) == 2 else 0), text=f"<b>{label}</b>",
                               showarrow=False, font=dict(size=11, color=ORANGE))
        done.add(tx)
    fig.add_trace(go.Scatter(                                # buses
        x=[p[0] for p in BUS.values()], y=[p[1] for p in BUS.values()], mode="markers+text",
        text=list(BUS), textposition="top center", marker=dict(size=9, symbol="square", color="#5b6472"),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(                                # transformers
        x=[TX_POS[t][0] for t in TX_POS], y=[TX_POS[t][1] for t in TX_POS], mode="markers+text",
        text=[f"<b>{t}</b><br>{loading[t]:.0f}%" for t in TX_POS], textfont=dict(color="white", size=12),
        marker=dict(size=58, color=[colour(loading[t], threshold) for t in TX_POS],
                    line=dict(width=2, color="white")),
        hovertext=[f"{t}: {loading[t]:.1f}% of rating" for t in TX_POS], hoverinfo="text"))
    fig.update_xaxes(visible=False, range=[-0.5, 4.2])
    fig.update_yaxes(visible=False, range=[-3.4, 3.4])
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig


def load_chart(df, t, kw_limit, adjusted=None, window_h=36):
    """df: timestamp, predicted_kw, actual_kw for ONE transformer. adjusted: optional scenario series."""
    t = pd.Timestamp(t)
    w = df[(df["timestamp"] >= t - pd.Timedelta(hours=window_h)) & (df["timestamp"] <= t + pd.Timedelta(hours=window_h))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["actual_kw"], name="actual", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["predicted_kw"], name="forecast", line=dict(color=ORANGE, width=2)))
    if adjusted is not None:
        a = adjusted.loc[w["timestamp"]] if hasattr(adjusted, "loc") else adjusted
        fig.add_trace(go.Scatter(x=w["timestamp"], y=a.values, name="forecast + scenario",
                                 line=dict(color=RED, width=2, dash="dash")))
    fig.add_hline(y=kw_limit, line_dash="dash", line_color=RED, annotation_text="overload limit",
                  annotation_position="top left")
    fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper", line=dict(color=GREY, width=1.5, dash="dot"))
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="load (kW)",
                      legend=dict(orientation="h", y=1.08))
    return fig


def before_after_chart(before, after, threshold):
    txs = list(before)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=txs, y=[before[t] for t in txs], name="before", marker_color=GREY))
    fig.add_trace(go.Bar(x=txs, y=[after[t] for t in txs], name="after", marker_color=GREEN))
    fig.add_hline(y=threshold, line_dash="dash", line_color=RED, annotation_text=f"{threshold:.0f}% limit")
    fig.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="transformer loading (%)", legend=dict(orientation="h", y=1.15))
    return fig