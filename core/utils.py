import time

class FPSMeter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self._ema_dt = None
        self._last_ts = None

    def tick(self, ts: float | None = None):
        now = ts or time.time()
        if self._last_ts is None:
            self._last_ts = now
            return
        dt = max(1e-6, now - self._last_ts)
        self._last_ts = now
        if self._ema_dt is None:
            self._ema_dt = dt
        else:
            a = self.alpha
            self._ema_dt = a * dt + (1.0 - a) * self._ema_dt

    @property
    def fps(self) -> float | None:
        if self._ema_dt is None:
            return None
        return 1.0 / max(1e-6, self._ema_dt)

