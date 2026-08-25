"""Среда."""
import math
import random
import numpy as np


class DynamicEnvironment:
    def __init__(self, config):
        self.cfg = config
        self.width = config.width
        self.height = config.height

        # Поле ресурсов: случайное начальное заполнение
        self.resource = np.random.uniform(
            0, config.max_resource, size=(self.height, self.width)
        )
        # Ёмкость: неоднородная, чтобы был рельеф
        self.capacity = self._make_capacity()

        # Климатическая динамика (независимая)
        self.season_phase = 0.0
        self.regen_multiplier = 1.0

        # Катастрофы
        self.catastrophe_active = False
        self.catastrophe_remaining = 0

        # Для логирования
        self.step_count = 0

    def _make_capacity(self) -> np.ndarray:
        """Неоднородная ёмкость: несколько «пиков» ресурсов."""
        cap = np.ones((self.height, self.width)) * self.cfg.max_resource * 0.3
        n_peaks = 3
        for _ in range(n_peaks):
            cy, cx = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.hypot(y - cy, x - cx)
                    cap[y, x] += self.cfg.max_resource * max(0, 1.0 - dist / 15.0)
        return np.clip(cap, 0, self.cfg.max_resource)

    # ------------------------------------------------------------------
    # Динамика среды
    # ------------------------------------------------------------------
    def step(self):
        """Один шаг среды. Вызывается ДО активации агентов."""
        self._update_climate()
        self._regenerate_resources()
        self._maybe_catastrophe()
        self.step_count += 1

    def _update_climate(self):
        """Сезонный цикл: синусоидальная модуляция регенерации."""
        self.season_phase += 2.0 * math.pi / self.cfg.season_period
        base = 1.0 + self.cfg.season_amplitude * math.sin(self.season_phase)
        if self.catastrophe_active:
            base *= self.cfg.catastrophe_severity
        self.regen_multiplier = max(base, 0.0)

    def _regenerate_resources(self):
        """Регенерация ресурсов (не зависит от агентов)."""
        growth = self.cfg.regen_rate * self.regen_multiplier
        self.resource = np.minimum(self.resource + growth, self.capacity)

    def _maybe_catastrophe(self):
        """Случайные катастрофы: засуха, шторм и т.п."""
        if self.catastrophe_active:
            self.catastrophe_remaining -= 1
            if self.catastrophe_remaining <= 0:
                self.catastrophe_active = False
        else:
            if random.random() < self.cfg.catastrophe_prob:
                self.catastrophe_active = True
                self.catastrophe_remaining = self.cfg.catastrophe_duration
                # Немедленное частичное уничтожение ресурсов
                self.resource *= 0.5

    # ------------------------------------------------------------------
    # Интерфейс для агентов
    # ------------------------------------------------------------------
    def get_resource(self, x: int, y: int) -> float:
        return self.resource[y, x]

    def harvest(self, x: int, y: int, amount: float) -> float:
        """Агент потребляет ресурс. Возвращает фактически собранное."""
        available = self.resource[y, x]
        taken = min(available, amount)
        self.resource[y, x] -= taken
        return taken

    @property
    def total_resource(self) -> float:
        return float(self.resource.sum())
