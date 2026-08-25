from mesa import Agent
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class Genome:
    vision: int
    metabolism: float
    strategy: str
    max_age: int

    def mutate(self, rate, min_vision, max_vision, min_metabolism, max_metabolism, rng):
        g = replace(self)
        if rng.random() < rate:
            g.vision = int(np.clip(g.vision + rng.choice([-1, 1]), min_vision, max_vision))
        if rng.random() < rate:
            g.metabolism = float(
                np.clip(g.metabolism + rng.uniform(-0.5, 0.5), min_metabolism, max_metabolism)
            )
        if rng.random() < rate:
            g.strategy = "C" if g.strategy == "D" else "D"
        if rng.random() < rate:
            g.max_age = int(max(50, g.max_age + rng.integers(-20, 21)))
        return g


class EcoAgent(Agent):
    """Агент без собственного метода step(). Логика вынесена в Model для синхронности."""

    def __init__(self, model, genome):
        super().__init__(model)
        self.genome = genome
        # Ресурс теперь задаётся явно при создании (родителем или моделью)
        self.resource = 0.0
        self.age = 0

    @property
    def alive(self):
        return self.resource > 0 and self.age < self.genome.max_age

    def perceive_and_move(self):
        """Выбирает лучшую клетку в радиусе зрения."""
        vision = self.genome.vision
        x, y = self.pos
        env_resource = self.model.env.resource
        
        best_x, best_y = x, y
        best_val = env_resource[y, x]

        # Используем кэш модели для ускорения
        neighbors = self.model.get_neighborhood_cached(self.pos, vision)
        
        for nx, ny in neighbors:
            val = env_resource[ny, nx]
            if val > best_val:
                best_val = val
                best_x, best_y = nx, ny

        if (best_x, best_y) != self.pos:
            self.model.grid.move_agent(self, (best_x, best_y))

    def metabolize(self):
        self.resource -= self.genome.metabolism

    def can_reproduce(self):
        # Нужно иметь достаточно ресурса, чтобы поделиться им с потомком
        # Минимум: reproduction_threshold + способность отдать половину
        threshold = self.model.cfg.reproduction_threshold
        return self.alive and self.resource > threshold

    def reproduce(self):
        """Консервативное размножение: ресурс делится пополам."""
        rng = self.model.rng
        child_genome = self.genome.mutate(
            self.model.cfg.mutation_rate,
            self.model.cfg.min_vision,
            self.model.cfg.max_vision,
            self.model.cfg.min_metabolism,
            self.model.cfg.max_metabolism,
            rng,
        )
        
        # Делим ресурс родителя пополам
        child_resource = self.resource / 2.0
        self.resource = child_resource

        child = EcoAgent(self.model, genome=child_genome)
        child.resource = child_resource
        
        return child
