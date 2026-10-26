"""Database schema (SQLAlchemy Core, so the same definitions work on SQLite and PostgreSQL).

Tables and what they hold:
    meta              schema version
    assets            substation, feeders, transformers, switches (from config/grid.yaml)
    meters            meter registry: MTR-xxxxx -> household -> transformer (from prepare_data.py)
    forecasts         one row per (model version, target time, transformer, horizon); pending until scored
    forecast_quantiles  interval bounds (e.g. P10, P90) for those forecasts
    measurements      measured load per (time, asset, source), with a quality flag
    recommendations   every recommendation the controller produced, with its full before/after detail
    candidates        every switching option evaluated for a recommendation, with its outcome and reason
    actions           every switch operation executed on the (simulated) network
    events            the audit log: every controller event with severity
    event_assets      which assets an event concerns (for filtering the log by transformer)
    alerts            risk and network alerts as they were observed
    controller_state  the controller's mode, configuration, emergency stop and undo stack
    model_registry    forecasting model versions (filled in Phase 4)
"""
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, MetaData, String, Table,
                        Text, UniqueConstraint)

SCHEMA_VERSION = 1
metadata = MetaData()

meta = Table("meta", metadata,
             Column("key", String(50), primary_key=True),
             Column("value", String(200), nullable=False))

assets = Table("assets", metadata,
               Column("asset_id", String(20), primary_key=True),
               Column("kind", String(20), nullable=False),
               Column("name", String(30), nullable=False),
               Column("parent_id", String(20)),
               Column("rating_kva", Float),
               Column("switch_kind", String(30)),
               Column("normal", String(10)))

meters = Table("meters", metadata,
               Column("meter_id", String(20), primary_key=True),
               Column("transformer_asset_id", String(20), ForeignKey("assets.asset_id"), nullable=False),
               Column("deployment_id", String(40)),
               Column("household_id", String(40)),
               Column("region", String(60)))

forecasts = Table("forecasts", metadata,
                  Column("id", Integer, primary_key=True, autoincrement=True),
                  Column("model_version", String(40), nullable=False),
                  Column("target_time", DateTime, nullable=False),
                  Column("asset_id", String(20), ForeignKey("assets.asset_id"), nullable=False),
                  Column("horizon_h", Float, nullable=False),
                  Column("predicted_kw", Float, nullable=False),
                  Column("status", String(20), nullable=False, default="pending"),   # pending | scored
                  Column("error_kw", Float),
                  Column("created_at", String(30)),
                  UniqueConstraint("model_version", "target_time", "asset_id", "horizon_h", name="uq_forecast"))
Index("ix_forecasts_time", forecasts.c.model_version, forecasts.c.target_time)

forecast_quantiles = Table("forecast_quantiles", metadata,
                           Column("id", Integer, primary_key=True, autoincrement=True),
                           Column("model_version", String(40), nullable=False),
                           Column("target_time", DateTime, nullable=False),
                           Column("asset_id", String(20), ForeignKey("assets.asset_id"), nullable=False),
                           Column("horizon_h", Float, nullable=False),
                           Column("quantile", Float, nullable=False),
                           Column("kw", Float, nullable=False),
                           Column("method", String(40)),
                           UniqueConstraint("model_version", "target_time", "asset_id", "horizon_h", "quantile",
                                            name="uq_forecast_quantile"))

measurements = Table("measurements", metadata,
                     Column("id", Integer, primary_key=True, autoincrement=True),
                     Column("time", DateTime, nullable=False),
                     Column("asset_id", String(20), nullable=False),
                     Column("source", String(40), nullable=False),
                     Column("kw", Float),
                     Column("quality", String(20), nullable=False, default="good"),
                     UniqueConstraint("time", "asset_id", "source", name="uq_measurement"))
Index("ix_measurements_time", measurements.c.source, measurements.c.time)

recommendations = Table("recommendations", metadata,
                        Column("rec_id", String(20), primary_key=True),
                        Column("seq", Integer, nullable=False),                    # numeric part, for ordering
                        Column("controller", String(40), nullable=False),
                        Column("created_at", String(30)),
                        Column("for_time", String(30)),
                        Column("context", Text),
                        Column("kind", String(20), nullable=False),
                        Column("status", String(20), nullable=False),
                        Column("summary", Text),
                        Column("threshold", Float),
                        Column("from_config", JSON),
                        Column("target", JSON),
                        Column("plan", JSON),
                        Column("before", JSON),
                        Column("after", JSON),
                        Column("checks", JSON),
                        Column("losses_before_kw", Float),
                        Column("losses_after_kw", Float),
                        Column("confidence", JSON),
                        Column("binding", JSON),
                        Column("engine_status", String(20)),
                        Column("compute_s", Float))

candidates = Table("candidates", metadata,
                   Column("id", Integer, primary_key=True, autoincrement=True),
                   Column("rec_id", String(20), ForeignKey("recommendations.rec_id"), nullable=False),
                   Column("cand_id", String(10), nullable=False),
                   Column("round", Integer),
                   Column("kind", String(10)),
                   Column("actions", JSON),
                   Column("label", Text),
                   Column("status", String(20)),
                   Column("reason", Text),
                   Column("feasible", Boolean),
                   Column("max_loading", Float),
                   Column("max_loading_tx", String(10)),
                   Column("min_voltage_pu", Float),
                   Column("max_line_loading", Float),
                   Column("losses_kw", Float))
Index("ix_candidates_rec", candidates.c.rec_id)

events = Table("events", metadata,
               Column("seq", Integer, primary_key=True, autoincrement=True),
               Column("controller", String(40), nullable=False),
               Column("wall_time", String(30)),
               Column("sim_time", String(30)),
               Column("type", String(30), nullable=False),
               Column("severity", String(10), nullable=False),
               Column("message", Text),
               Column("rec_id", String(20)),
               Column("mode", String(20)),
               Column("configuration", JSON),
               Column("data", JSON))
Index("ix_events_sim_time", events.c.sim_time)

event_assets = Table("event_assets", metadata,
                     Column("seq", Integer, ForeignKey("events.seq"), primary_key=True),
                     Column("asset_id", String(20), primary_key=True))

actions = Table("actions", metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("event_seq", Integer, ForeignKey("events.seq")),
                Column("controller", String(40), nullable=False),
                Column("wall_time", String(30)),
                Column("sim_time", String(30)),
                Column("rec_id", String(20)),
                Column("how", String(20)),            # operator | auto | manual | rollback | reset
                Column("step", Integer),
                Column("switch", String(20)),
                Column("switch_id", String(20)),
                Column("operation", String(10)),
                Column("action_key", String(20)))

alerts = Table("alerts", metadata,
               Column("id", Integer, primary_key=True, autoincrement=True),
               Column("sim_time", String(30), nullable=False),
               Column("asset", String(20), nullable=False),          # asset ID, bus or line name
               Column("kind", String(30), nullable=False),           # transformer | voltage_low | voltage_high | line
               Column("level", String(30), nullable=False),          # risk class, or violation / approaching
               Column("value", Float),
               Column("text", Text),
               Column("context", Text),
               Column("first_seen", String(30)),
               UniqueConstraint("sim_time", "asset", "kind", "context", name="uq_alert"))
Index("ix_alerts_sim_time", alerts.c.sim_time)

controller_state = Table("controller_state", metadata,
                         Column("controller", String(40), primary_key=True),
                         Column("mode", String(20), nullable=False),
                         Column("active", JSON, nullable=False),
                         Column("estopped", Boolean, nullable=False),
                         Column("undo", JSON, nullable=False),
                         Column("rec_seq", Integer, nullable=False),
                         Column("updated_at", String(30)))

model_registry = Table("model_registry", metadata,
                       Column("version", String(40), primary_key=True),
                       Column("created_at", String(30)),
                       Column("kind", String(40)),
                       Column("path", Text),
                       Column("features", JSON),
                       Column("params", JSON),
                       Column("metrics", JSON),
                       Column("notes", Text))
