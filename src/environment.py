import math
import numpy as np

class DynamicEnvironment:
    def __init__(self, width, height, max_resource, regen_rate,
                 max_spice, regen_rate_spice,
                 season_period, season_amplitude, catastrophe_prob,
                 catastrophe_duration, catastrophe_severity, rng):
        self.width = width
        self.height = height
        self.max_resource = max_resource
        self.regen_rate = regen_rate
        self.max_spice = max_spice
        self.regen_rate_spice = regen_rate_spice
        self.season_period = season_period
        self.season_amplitude = season_amplitude
        self.catastrophe_prob = catastrophe_prob
        self.catastrophe_duration = catastrophe_duration
        self.catastrophe_severity = catastrophe_severity
        self.rng = rng

        self.sugar = self.rng.uniform(0, self.max_resource, size=(self.height, self.width))
        self.spice = self.rng.uniform(0, self.max_spice, size=(self.height, self.width))
        self.capacity_sugar = self._make_capacity(self.max_resource)
        self.capacity_spice = self._make_capacity(self.max_spice)
        self.season_phase = 0.0
        self.regen_multiplier = 1.0
        self.catastrophe_active = False
        self.catastrophe_remaining = 0
        self.step_count = 0

    def _make_capacity(self, max_val) -> np.ndarray:
        """
        ИЗМЕНЕНИЕ: Увеличено число пиков с 3 до 5 и радиус с 15 до 20.
        Обоснование: более однородное распределение ёмкости снижает
        конкуренцию за ограниченные "оазисы" и создаёт более
        равномерную кормовую базу (аналог ландшафтной экологии).
        """
        cap = np.ones((self.height, self.width)) * max_val * 0.4  # Было 0.3
        n_peaks = 5  # Было 3
        for _ in range(n_peaks):
            cy = int(self.rng.integers(0, self.height))
            cx = int(self.rng.integers(0, self.width))
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.hypot(y - cy, x - cx)
                    cap[y, x] += max_val * max(0, 1.0 - dist / 20.0)  # Было /15
        return np.clip(cap, 0, max_val)

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
        self.regen_multiplier = max(base, 0.1)  # Было 0.0 — минимум 10% регенерации

    def _regenerate_resources(self):
        growth_sugar = self.regen_rate * self.regen_multiplier
        self.sugar = np.minimum(self.sugar + growth_sugar, self.capacity_sugar)
        growth_spice = self.regen_rate_spice * self.regen_multiplier
        self.spice = np.minimum(self.spice + growth_spice, self.capacity_spice)

    def _maybe_catastrophe(self):
        if self.catastrophe_active:
            self.catastrophe_remaining -= 1
            if self.catastrophe_remaining <= 0:
                self.catastrophe_active = False
        else:
            if self.rng.random() < self.catastrophe_prob:
                self.catastrophe_active = True
                self.catastrophe_remaining = self.catastrophe_duration
                self.sugar *= 0.5
                self.spice *= 0.5

    @property
    def total_sugar(self) -> float:
        return float(self.sugar.sum())

    @property
    def total_spice(self) -> float:
        return float(self.spice.sum())

    @property
    def total_resource(self) -> float:
        return self.total_sugar + self.total_spice
