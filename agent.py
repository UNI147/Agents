"""Геном и агент. Геном несёт: зрение, метаболизм, стратегию, макс. возраст."""
import random
from dataclasses import dataclass, replace


@dataclass
class Genome:
    vision: int          # радиус восприятия
    metabolism: float    # потребление ресурса за шаг
    strategy: str        # 'C' (кооператор) или 'D' (дефектор)
    max_age: int

    def mutate(self, rate: float, cfg) -> "Genome":
        """Копия генома с возможными мутациями каждого гена."""
        g = replace(self)
        if random.random() < rate:
            g.vision = max(1, g.vision + random.choice([-1, 1]))
            g.vision = min(max(g.vision, cfg.min_vision), cfg.max_vision)
        if random.random() < rate:
            g.metabolism = max(0.5, g.metabolism + random.uniform(-0.5, 0.5))
            g.metabolism = min(max(g.metabolism, cfg.min_metabolism),
                               cfg.max_metabolism)
        if random.random() < rate:
            g.strategy = 'C' if g.strategy == 'D' else 'D'
        if random.random() < rate:
            g.max_age = max(50, g.max_age + random.randint(-20, 20))
        return g


class Agent:
    _next_id = 0

    def __init__(self, env, cfg, pos=None, genome=None):
        self.env = env
        self.cfg = cfg
        self.id = Agent._next_id
        Agent._next_id += 1

        if pos is None:
            self.x = random.randint(0, env.width - 1)
            self.y = random.randint(0, env.height - 1)
        else:
            self.x, self.y = pos

        if genome is None:
            genome = Genome(
                vision=random.randint(cfg.min_vision, cfg.max_vision),
                metabolism=random.uniform(cfg.min_metabolism, cfg.max_metabolism),
                strategy=random.choice(['C', 'D']),
                max_age=cfg.max_age,
            )
        self.genome = genome
        self.resource = cfg.initial_resource
        self.age = 0
        self.alive = True

    # ------------------------------------------------------------------
    # Основной шаг агента
    # ------------------------------------------------------------------
    def step(self, all_agents):
        if not self.alive:
            return
        self.age += 1
        self._perceive_and_move()
        self._harvest()
        self._interact(all_agents)
        self._metabolize()
        self._survival_check()

    def _perceive_and_move(self):
        """Движение к клетке с наибольшим ресурсом в радиусе зрения."""
        vision = self.genome.vision
        best_pos = (self.x, self.y)
        best_val = self.env.get_resource(self.x, self.y)
        for dy in range(-vision, vision + 1):
            for dx in range(-vision, vision + 1):
                nx, ny = self.x + dx, self.y + dy
                nx %= self.env.width      # тор
                ny %= self.env.height
                val = self.env.get_resource(nx, ny)
                if val > best_val:
                    best_val = val
                    best_pos = (nx, ny)
        self.x, self.y = best_pos

    def _harvest(self):
        """Сбор ресурса: объём ограничен метаболизмом (эффективность)."""
        amount = self.genome.metabolism * 2.0  # множитель эффективности
        self.resource += self.env.harvest(self.x, self.y, amount)

    def _interact(self, all_agents):
        """Эволюционная игра: при встрече на одной клетке."""
        neighbors = [a for a in all_agents
                     if a.alive and a is not self
                     and a.x == self.x and a.y == self.y]
        for other in neighbors:
            payoff = self.cfg.game.payoff(self.genome.strategy,
                                          other.genome.strategy)
            self.resource += payoff

    def _metabolize(self):
        self.resource -= self.genome.metabolism

    def _survival_check(self):
        if self.resource <= 0 or self.age >= self.genome.max_age:
            self.alive = False

    # ------------------------------------------------------------------
    # Эволюция
    # ------------------------------------------------------------------
    def can_reproduce(self) -> bool:
        return self.alive and self.resource > self.cfg.reproduction_threshold

    def reproduce(self) -> "Agent":
        """Деление: половина ресурса передаётся потомку, геном мутирует."""
        child_genome = self.genome.mutate(self.cfg.mutation_rate, self.cfg)
        self.resource /= 2.0
        return Agent(self.env, self.cfg, pos=(self.x, self.y),
                     genome=child_genome)

    @property
    def pos(self):
        return (self.x, self.y)
