import math
import numpy as np

class DynamicEnvironment:
    def __init__(self, width, height, max_resource, regen_rate,
                 max_spice, regen_rate_spice,
                 season_period, season_amplitude, catastrophe_prob,
                 catastrophe_duration, catastrophe_severity, rng,
                 pollution_enabled=True, pollution_diffusion_rate=0.2,
                 pollution_decay_rate=0.05, pollution_capacity_impact=1.5):
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

        # Параметры загрязнения (Пункт 2.2)
        self.pollution_enabled = pollution_enabled
        self.pollution_diffusion_rate = pollution_diffusion_rate
        self.pollution_decay_rate = pollution_decay_rate
        self.pollution_capacity_impact = pollution_capacity_impact
        
        self.sugar = self.rng.uniform(0, self.max_resource, size=(self.height, self.width))
        self.spice = self.rng.uniform(0, self.max_spice, size=(self.height, self.width))
        self.capacity_sugar = self._make_capacity(self.max_resource)
        self.capacity_spice = self._make_capacity(self.max_spice)
        
        self.pollution = np.zeros((self.height, self.width))
        
        self.season_phase = 0.0
        self.regen_multiplier = 1.0
        self.catastrophe_active = False
        self.catastrophe_remaining = 0
        self.step_count = 0

    def _make_capacity(self, max_val) -> np.ndarray:
        cap = np.ones((self.height, self.width)) * max_val * 0.4
        n_peaks = 5
        for _ in range(n_peaks):
            cy = int(self.rng.integers(0, self.height))
            cx = int(self.rng.integers(0, self.width))
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.hypot(y - cy, x - cx)
                    cap[y, x] += max_val * max(0, 1.0 - dist / 20.0)
        return np.clip(cap, 0, max_val)

    def add_pollution(self, pollution_grid):
        if self.pollution_enabled:
            self.pollution += pollution_grid

    def step(self):
        self._update_climate()
        if self.pollution_enabled:
            self._diffuse_and_decay_pollution()
        self._regenerate_resources()
        self._maybe_catastrophe()
        self.step_count += 1

    def _update_climate(self):
        self.season_phase += 2.0 * math.pi / self.season_period
        base = 1.0 + self.season_amplitude * math.sin(self.season_phase)
        if self.catastrophe_active:
            base *= self.catastrophe_severity
        self.regen_multiplier = max(base, 0.1)

    def _diffuse_and_decay_pollution(self):
        # Диффузия по фон Нейману (с учетом тороидальной сетки)
        padded = np.pad(self.pollution, 1, mode='wrap')
        neighbors_sum = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                         padded[1:-1, :-2] + padded[1:-1, 2:])
        neighbors_avg = neighbors_sum / 4.0
        
        diffused = self.pollution * (1 - self.pollution_diffusion_rate) + neighbors_avg * self.pollution_diffusion_rate
        self.pollution = np.maximum(0, diffused - self.pollution_decay_rate)

    def _regenerate_resources(self):
        # Загрязнение снижает эффективную ёмкость среды
        if self.pollution_enabled:
            eff_cap_sugar = np.maximum(0, self.capacity_sugar - self.pollution * self.pollution_capacity_impact)
            eff_cap_spice = np.maximum(0, self.capacity_spice - self.pollution * self.pollution_capacity_impact)
        else:
            eff_cap_sugar = self.capacity_sugar
            eff_cap_spice = self.capacity_spice
            
        growth_sugar = self.regen_rate * self.regen_multiplier
        self.sugar = np.minimum(self.sugar + growth_sugar, eff_cap_sugar)
        growth_spice = self.regen_rate_spice * self.regen_multiplier
        self.spice = np.minimum(self.spice + growth_spice, eff_cap_spice)

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

    @property
    def total_pollution(self) -> float:
        return float(self.pollution.sum())
