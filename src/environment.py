import math
import numpy as np

class ResourcePeak:
    """Представляет динамический пик ресурсов (гору/оазис), который дрейфует в пространстве."""
    def __init__(self, x, y, max_val, radius, resource_type, rng):
        self.x = x
        self.y = y
        self.max_val = max_val
        self.radius = radius
        self.resource_type = resource_type
        self.age = 0
        # Пики живут от 50 до 200 шагов, затем исчезают (смерть пика)
        self.max_age = int(rng.integers(50, 200))
        
    def drift(self, width, height, speed, rng):
        """Случайное блуждание (дрейф) пика по тороидальной сетке."""
        if speed > 0:
            dx = int(rng.integers(-speed, speed + 1))
            dy = int(rng.integers(-speed, speed + 1))
            self.x = (self.x + dx) % width
            self.y = (self.y + dy) % height
        self.age += 1
        
    def is_alive(self):
        return self.age < self.max_age

class DynamicEnvironment:
    def __init__(self, width, height, max_resource, regen_rate,
                 max_spice, regen_rate_spice,
                 season_period, season_amplitude, catastrophe_prob,
                 catastrophe_duration, catastrophe_severity, rng,
                 pollution_enabled=True, pollution_diffusion_rate=0.2,
                 pollution_decay_rate=0.05, pollution_capacity_impact=1.5,
                 resource_peaks_drift_speed=1,
                 resource_peaks_mutation_prob=0.05,
                 island_model_enabled=False,
                 islands_count=4):
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
        
        self.resource_peaks_drift_speed = resource_peaks_drift_speed
        self.resource_peaks_mutation_prob = resource_peaks_mutation_prob
        self.island_model_enabled = island_model_enabled
        self.islands_count = islands_count
        
        # Динамические пики (Пункт 2.5)
        self.sugar_peaks = []
        self.spice_peaks = []
        self._init_peaks()
        
        # Островная карта (Пункт 2.3)
        self.island_map = np.full((self.height, self.width), -1, dtype=int)
        self.island_cells = {i: [] for i in range(self.islands_count)}
        if self.island_model_enabled:
            self._generate_islands()
            
        # Емкость теперь динамическая
        self.capacity_sugar = np.zeros((self.height, self.width))
        self.capacity_spice = np.zeros((self.height, self.width))
        self._update_capacity() # Первичный расчет

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
        self._update_peaks()      # <-- Дрейф и мутация пиков (П. 2.5)
        self._update_capacity()   # <-- Пересчет карты ресурсов
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

    def _init_peaks(self):
        """Создание начальных пиков ресурсов."""
        n_peaks = 5
        for _ in range(n_peaks):
            self.sugar_peaks.append(ResourcePeak(
                int(self.rng.integers(0, self.width)),
                int(self.rng.integers(0, self.height)),
                self.max_resource * 0.8, 20.0, 'sugar', self.rng
            ))
            self.spice_peaks.append(ResourcePeak(
                int(self.rng.integers(0, self.width)),
                int(self.rng.integers(0, self.height)),
                self.max_spice * 0.8, 20.0, 'spice', self.rng
            ))

    def _generate_islands(self):
        """Генерация островов через диаграмму Вороного на тороиде."""
        centers_x = self.rng.integers(0, self.width, size=self.islands_count)
        centers_y = self.rng.integers(0, self.height, size=self.islands_count)
        
        for y in range(self.height):
            for x in range(self.width):
                min_dist = float('inf')
                best_island = 0
                for i in range(self.islands_count):
                    # Учет тороидальной топологии
                    dx = min(abs(x - centers_x[i]), self.width - abs(x - centers_x[i]))
                    dy = min(abs(y - centers_y[i]), self.height - abs(y - centers_y[i]))
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                        best_island = i
                self.island_map[y, x] = best_island
                self.island_cells[best_island].append((x, y))

    def _update_capacity(self):
        """Векторизованный пересчет емкости среды на основе дрейфующих пиков (Пункт 2.5)."""
        cap_s = np.full((self.height, self.width), self.max_resource * 0.1)
        cap_sp = np.full((self.height, self.width), self.max_spice * 0.1)
        
        y_indices, x_indices = np.indices((self.height, self.width))
        
        def add_peaks(capacity_grid, peaks):
            for peak in peaks:
                dx = np.abs(x_indices - peak.x)
                dx = np.minimum(dx, self.width - dx)
                dy = np.abs(y_indices - peak.y)
                dy = np.minimum(dy, self.height - dy)
                dist = np.hypot(dx, dy)
                capacity_grid += peak.max_val * np.maximum(0, 1.0 - dist / peak.radius)
                
        add_peaks(cap_s, self.sugar_peaks)
        add_peaks(cap_sp, self.spice_peaks)
        
        self.capacity_sugar = np.clip(cap_s, 0, self.max_resource)
        self.capacity_spice = np.clip(cap_sp, 0, self.max_spice)

    def _update_peaks(self):
        """Обновление состояния пиков: дрейф, смерть и рождение."""
        speed = self.resource_peaks_drift_speed
        
        # Дрейф существующих
        for peak in self.sugar_peaks:
            peak.drift(self.width, self.height, speed, self.rng)
        for peak in self.spice_peaks:
            peak.drift(self.width, self.height, speed, self.rng)
            
        # Исчезновение (смерть или мутация)
        self.sugar_peaks = [p for p in self.sugar_peaks if p.is_alive() and self.rng.random() > self.resource_peaks_mutation_prob]
        self.spice_peaks = [p for p in self.spice_peaks if p.is_alive() and self.rng.random() > self.resource_peaks_mutation_prob]
        
        # Появление новых пиков
        if self.rng.random() < self.resource_peaks_mutation_prob:
            self.sugar_peaks.append(ResourcePeak(
                int(self.rng.integers(0, self.width)),
                int(self.rng.integers(0, self.height)),
                self.max_resource * 0.8, 20.0, 'sugar', self.rng
            ))
        if self.rng.random() < self.resource_peaks_mutation_prob:
            self.spice_peaks.append(ResourcePeak(
                int(self.rng.integers(0, self.width)),
                int(self.rng.integers(0, self.height)),
                self.max_spice * 0.8, 20.0, 'spice', self.rng
            ))

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
