"""RateLimiter: proves the instantaneous rate never exceeds the target."""

from __future__ import annotations

from filings_analyst.edgar.throttle import RateLimiter


def _acquire_times(limiter: RateLimiter, clock, n: int) -> list[float]:
    """Record the virtual timestamp at which each of ``n`` acquisitions completes."""
    times = []
    for _ in range(n):
        limiter.acquire()
        times.append(clock.now())
    return times


def test_acquisitions_are_spaced_by_min_interval(fake_clock):
    rate = 8.0
    limiter = RateLimiter(rate, monotonic=fake_clock.now, sleep=fake_clock.sleep)

    times = _acquire_times(limiter, fake_clock, 10)

    min_interval = 1.0 / rate
    # Consecutive acquisitions are never closer than the minimum interval.
    for prev, cur in zip(times, times[1:]):
        assert cur - prev >= min_interval - 1e-9

    # 10 acquisitions span at least (n-1) intervals of virtual time.
    assert times[-1] - times[0] >= (10 - 1) * min_interval - 1e-9


def test_never_more_than_rate_in_any_one_second_window(fake_clock):
    rate = 8.0
    limiter = RateLimiter(rate, monotonic=fake_clock.now, sleep=fake_clock.sleep)

    times = _acquire_times(limiter, fake_clock, 40)

    # Slide a 1-second window; no window may contain more than `rate` acquisitions.
    for i, start in enumerate(times):
        in_window = [t for t in times[i:] if t < start + 1.0]
        assert len(in_window) <= rate


def test_no_sleep_when_caller_is_already_behind_schedule(fake_clock):
    """If real time has already elapsed past the interval, acquire() shouldn't sleep."""
    limiter = RateLimiter(8.0, monotonic=fake_clock.now, sleep=fake_clock.sleep)

    limiter.acquire()  # first call: immediate
    fake_clock.advance(5.0)  # caller was slow; plenty of budget has accrued
    limiter.acquire()  # should not need to sleep

    assert fake_clock.sleeps == []  # never slept
