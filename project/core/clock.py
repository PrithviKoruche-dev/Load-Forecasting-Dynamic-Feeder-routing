"""Simulation clock (gap items C2 and C3): simulated time kept separate from wall-clock time.

The clock advances simulated time in fixed steps and decides WHEN (in wall-clock time) the next step is
due; it does not run steps itself. The background service (service/run.py) asks `due()` and `wait_s()`,
runs a step, then calls `advance()`.

    speed = number       simulated seconds per wall-clock second (3600 = one simulated hour per second)
    speed = "accelerated" the accelerated mode: accelerated_factor (365 = one simulated year in 24 hours,
                         i.e. a 15-minute step every 2.466 s or a 60-minute step every 9.863 s)
    speed = "max"        no pacing: the next step is due as soon as the previous one has finished

Controls: start/resume, pause, step (one step while paused; the service runs it and calls advance),
set_speed, jump (to any step on the grid), restart (back to the start).

When a step takes longer than its slot the clock cannot keep the target speed. With catch_up false (the
default) it carries on from the present at the same pace: no burst of back-to-back steps, the achieved speed
drops below the target and the lag is reported. With catch_up true it keeps the original schedule and runs
steps back to back until it is on time again.

The wall clock is injectable (wall=...), so the pacing rules are tested without waiting in real time.
"""
import time
from collections import deque

import pandas as pd

SPEEDS = ("max", "accelerated")


class ClockError(ValueError):
    """An invalid clock operation (time off the step grid or outside the range, unknown speed)."""


class SimulationClock:
    def __init__(self, start, end, step_minutes=60, speed=3600, accelerated_factor=365, catch_up=False,
                 wall=time.monotonic, window=50):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        if self.end < self.start:
            raise ClockError(f"end {self.end} is before start {self.start}")
        self.step = pd.Timedelta(minutes=step_minutes)
        self.accelerated_factor = float(accelerated_factor)
        self.catch_up = catch_up
        self._wall = wall
        self.next_time = self.start              # the next simulated time to process
        self.last_time = None                    # the last simulated time completed
        self.running = False
        self.steps_done = 0                      # steps completed since this clock object was created
        self.lag_s = 0.0                         # how late the last step finished relative to its slot
        self._due = None
        self._recent = deque(maxlen=window)      # (wall, simulated seconds) of the most recent steps
        self.speed = None
        self.set_speed(speed)

    # ------------------------------------------------------------------ speed
    @staticmethod
    def check_speed(speed):
        if speed in SPEEDS:
            return speed
        try:
            v = float(speed)
        except (TypeError, ValueError):
            raise ClockError(f"speed must be a positive number, 'accelerated' or 'max', got {speed!r}") from None
        if not v > 0:
            raise ClockError(f"speed must be positive, got {speed!r}")
        return v

    @property
    def factor(self):
        """Target simulated seconds per wall-clock second (None = as fast as possible)."""
        if self.speed == "max":
            return None
        return self.accelerated_factor if self.speed == "accelerated" else self.speed

    @property
    def interval_s(self):
        """Wall-clock seconds between steps at the target speed (0 for max)."""
        f = self.factor
        return 0.0 if f is None else self.step.total_seconds() / f

    def set_speed(self, speed):
        self.speed = self.check_speed(speed)
        self._reset_window()
        if self.running:
            self._due = self._wall()               # the new pace starts from now
        return self.speed

    # ------------------------------------------------------------------ controls
    @property
    def finished(self):
        return self.next_time > self.end

    def resume(self):
        if self.finished:
            raise ClockError(f"the clock has reached the end ({self.end}); restart or jump first")
        if not self.running:
            self.running = True
            self._due = self._wall()               # the first step after resuming is due at once
            self._reset_window()

    def pause(self):
        self.running = False
        self._due = None

    def on_grid(self, t):
        t = pd.Timestamp(t)
        return (t - self.start) % self.step == pd.Timedelta(0)

    def jump(self, t):
        """Make t the next simulated time to process (it must lie on the step grid, within start..end)."""
        t = pd.Timestamp(t)
        if not self.start <= t <= self.end:
            raise ClockError(f"{t} is outside the simulation range {self.start} .. {self.end}")
        if not self.on_grid(t):
            raise ClockError(f"{t} is not on the {self.step} step grid starting at {self.start}")
        self.next_time = t
        self._reset_window()
        if self.running:
            self._due = self._wall()

    def restart(self):
        """Back to the start (the service starts a new run, so earlier records are kept)."""
        self.next_time, self.last_time = self.start, None
        self.lag_s = 0.0
        self._reset_window()
        if self.running:
            self._due = self._wall()

    # ------------------------------------------------------------------ pacing
    def due(self):
        """True if the next step should run now."""
        return self.running and not self.finished and self._wall() >= self._due

    def wait_s(self):
        """Wall-clock seconds until the next step is due (None while paused or finished)."""
        if not self.running or self.finished:
            return None
        return max(0.0, self._due - self._wall())

    def advance(self):
        """Record that the step at next_time has been completed, and schedule the next one."""
        now = self._wall()
        self.last_time = self.next_time
        self.next_time = self.next_time + self.step
        self.steps_done += 1
        self._recent.append((now, (self.last_time - self.start).total_seconds()))
        if self.running and self.factor is None:          # max speed: no schedule, so no lag
            self.lag_s, self._due = 0.0, now
        elif self.running and self._due is not None:
            ideal = self._due + self.interval_s
            self.lag_s = max(0.0, now - ideal)
            self._due = ideal if self.catch_up else max(ideal, now)
        if self.finished:
            self.running = False
        return self.last_time

    def _reset_window(self):
        self._recent.clear()

    @property
    def achieved_factor(self):
        """Measured simulated seconds per wall-clock second over the most recent steps (None until known)."""
        if len(self._recent) < 2:
            return None
        (w0, s0), (w1, s1) = self._recent[0], self._recent[-1]
        return (s1 - s0) / (w1 - w0) if w1 > w0 else None

    # ------------------------------------------------------------------ persistence
    def state(self):
        """Plain-data state for the database."""
        return {"start": str(self.start), "end": str(self.end), "step_minutes": self.step.total_seconds() / 60,
                "next_time": str(self.next_time), "last_time": None if self.last_time is None else str(self.last_time),
                "running": self.running, "speed": self.speed}

    def restore(self, st, resume=True):
        """Continue from a saved state (same start, end and step)."""
        self.next_time = pd.Timestamp(st["next_time"])
        self.last_time = pd.Timestamp(st["last_time"]) if st.get("last_time") else None
        self.set_speed(st.get("speed", self.speed))
        if resume and st.get("running") and not self.finished:
            self.resume()
        else:
            self.pause()


def resolve_range(clock_cfg, available_times):
    """(start, end) from config: test_start / test_end mean the first / last available time."""
    times = pd.DatetimeIndex(available_times)
    start = times.min() if clock_cfg["start"] == "test_start" else pd.Timestamp(clock_cfg["start"])
    end = times.max() if clock_cfg["end"] == "test_end" else pd.Timestamp(clock_cfg["end"])
    return start, end
