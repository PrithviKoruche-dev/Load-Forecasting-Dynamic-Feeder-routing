"""Controller: operating modes, approve/reject, stale-approval protection, rollback, emergency stop, history."""
import pandas as pd
import pytest

import engine as e
import run_results as rr
from core import scenarios
from decisions import Controller, DecisionError

S3_HOUR = "2020-02-29 21:00"


@pytest.fixture(scope="module")
def loads():
    pred = rr.load_predictions()
    sc = {s.name.split()[0]: s for s in scenarios.benchmark()}

    def f(name, ts):
        ts = pd.Timestamp(ts)
        return {tx: float(sc[name].adjust(tx, ts, pred.loc[ts, tx])) for tx in e.TX}
    return f


@pytest.fixture
def ctl():
    return Controller(e._default, now=lambda: "2026-01-01T00:00:00")


def test_default_mode_is_recommend(ctl):
    assert ctl.mode == "recommend" and ctl.active == [] and not ctl.estopped


def test_recommend_then_approve(ctl, loads):
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    assert rec.rec_id == "REC-0001" and rec.kind == "reroute" and rec.status == "pending"
    assert ctl.active == []                                   # nothing happens until approved
    assert [s["switch_id"] for s in rec.plan] == ["SW-004", "SW-010"]
    assert all(c["ok"] for c in rec.checks)
    assert rec.losses_after_kw < rec.losses_before_kw and rec.confidence["label"] == "medium"
    ctl.approve(rec.rec_id, S3_HOUR, "S3")
    assert ctl.active == ["T4>T5"] and rec.status == "applied" and ctl.pending is None
    assert [h["type"] for h in ctl.history] == ["recommendation", "approved", "applied"]


def test_stale_approval_is_refused(ctl, loads):
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    with pytest.raises(DecisionError, match="conditions have changed"):
        ctl.approve(rec.rec_id, "2020-02-29 22:00", "S3")
    with pytest.raises(DecisionError, match="conditions have changed"):
        ctl.approve(rec.rec_id, S3_HOUR, "another scenario")
    assert ctl.active == []


def test_manual_change_cancels_the_pending_proposal(ctl, loads):
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    ctl.set_configuration(["T1>T2"])
    assert rec.status == "superseded"
    with pytest.raises(DecisionError, match="not the pending"):
        ctl.approve(rec.rec_id, S3_HOUR, "S3")


def test_reject(ctl, loads):
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    ctl.reject(rec.rec_id, "crew not available")
    assert rec.status == "rejected" and ctl.active == []
    assert "crew not available" in ctl.history[-1]["message"]


def test_return_to_normal_and_rollback(ctl, loads):
    ctl.approve(ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3").rec_id, S3_HOUR, "S3")
    rec = ctl.recommend(loads("S1", S3_HOUR), S3_HOUR, "S1")
    assert rec.kind == "restore" and rec.target == []
    assert [(s["switch"], s["operation"]) for s in rec.plan] == [("SW4A", "open"), ("SW4H", "close")]
    ctl.approve(rec.rec_id, S3_HOUR, "S1")
    assert ctl.active == []
    ctl.rollback()
    assert ctl.active == ["T4>T5"]
    ctl.rollback()
    assert ctl.active == []
    with pytest.raises(DecisionError, match="Nothing to roll back"):
        ctl.rollback()


def test_no_safe_option_is_reported_not_applied(ctl, loads):
    rec = ctl.recommend(loads("S5", "2020-02-17 07:00"), "2020-02-17 07:00", "S5")
    assert rec.kind == "no_safe_option" and rec.status == "info" and rec.plan == []
    assert rec.binding and ctl.pending is None and ctl.active == []
    assert ctl.history[-1]["type"] == "no_safe_option"


def test_monitor_mode_proposes_nothing(ctl, loads):
    ctl.set_mode("monitor")
    assert ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3") is None
    with pytest.raises(DecisionError, match="only used in recommend mode"):
        ctl.approve("REC-0001")


def test_auto_mode_applies_by_itself(ctl, loads):
    ctl.set_mode("auto")
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    assert rec.status == "applied" and ctl.active == ["T4>T5"]
    rec = ctl.recommend(loads("S3", "2020-02-29 12:00"), "2020-02-29 12:00", "S3")   # loads fall: back to normal
    assert rec.kind == "restore" and ctl.active == []


def test_emergency_stop(ctl, loads):
    ctl.set_mode("auto")
    rec = ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    assert ctl.active == ["T4>T5"]
    ctl.emergency_stop("2020-02-29 21:00")
    assert ctl.estopped and ctl.mode == "monitor"
    assert ctl.active == ["T4>T5"]                            # switches are NOT changed blindly
    assert ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3") is None
    with pytest.raises(DecisionError, match="Emergency stop"):
        ctl.set_mode("auto")
    ctl.rollback()                                            # the operator can still act manually
    assert ctl.active == []
    ctl.clear_emergency_stop()
    assert not ctl.estopped and ctl.mode == "monitor"
    ctl.set_mode("recommend")
    assert rec.rec_id == "REC-0001"


def test_unsafe_manual_configuration_is_refused(ctl):
    with pytest.raises(DecisionError, match="unsupplied"):
        ctl.set_configuration(["MV1", "MV2"])
    assert ctl.active == []


def test_snapshot_is_plain_data(ctl, loads):
    ctl.recommend(loads("S3", S3_HOUR), S3_HOUR, "S3")
    snap = ctl.snapshot()
    assert snap["mode"] == "recommend" and snap["pending"]["rec_id"] == "REC-0001"
    assert isinstance(snap["history"], list) and snap["seq"] == 1
