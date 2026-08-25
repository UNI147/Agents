import math
import numpy as np

class DynamicEnvironment:
    def __init__(self, width, height, max_resource, regen_rate, season_period, 
                 season_amplitude, catastrophe_prob, catastrophe_duration, 
                 catastrophe_severity, rng):
        self.width = width
        self.height = height
        self.max_resource = max_resource
        self.regen_rate = regen_rate
        self.season_period = season_period
        self.season_amplitude = season_amplitude
        self.catastrophe_prob = catastrophe_prob
        self.catastrophe_duration = catastrophe_duration
        self.catastrophe_severity = catastrophe_severity
        self.rng = rng
        
        self.resource = self.rng.uniform(0, self.max_resource, size=(self.height, self.width))
        self.capacity = self._make_capacity()
        
        self.season_phase = 0.0
        self.regen_multiplier = 1.0
        
        self.catastrophe_active = False
        self.catastrophe_remaining = 0
        self.step_count = 0

    def _make_capacity(self) -> np.ndarray:
        cap = np.ones((self.height, self.width)) * self.max_resource * 0.3
        n_peaks = 3
        for _ in range(n_peaks):
            cy = int(self.rng.integers(0, self.height))
            cx = int(self.rng.integers(0, self.width))
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.hypot(y - cy, x - cx)
                    cap[y, x] += self.max_resource * max(0, 1.0 - dist / 15.0)
        return np.clip(cap, 0, self.max_resource)

    def step(self):
        self._update_climate()
        self._regenerate_resources()
        self._maybe_catastrophe()
        self.step_count += 1

    def _update_climate(self):
        self.season_phase += 2.0 * math.pi / self.season_period
        base = 1.0 + self.season_amplitude * math.sin(self.season_phase)
        if self.catastrophe_active:
            base *= self.catastrophe_severity
        self.regen_multiplier = max(base, 0.0)

    def _regenerate_resources(self):
        growth = self.regen_rate * self.regen_multiplier
        self.resource = np.minimum(self.resource + growth, self.capacity)

    def _maybe_catastrophe(self):
        if self.catastrophe_active:
            self.catastrophe_remaining -= 1
            if self.catastrophe_remaining <= 0:
                self.catastrophe_active = False
        else:
            if self.rng.random() < self.catastrophe_prob:
                self.catastrophe_active = True
                self.catastrophe_remaining = self.catastrophe_duration
                self.resource *= 0.5

    def get_resource(self, x: int, y: int) -> float:
        return self.resource[y, x]

    def harvest(self, x: int, y: int, amount: float) -> float:
        available = self.resource[y, x]
        taken = min(available, amount)
        self.resource[y, x] -= taken
        return taken

    @property
    def total_resource(self) -> float:
        return float(self.resource.sum())
