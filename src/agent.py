from mesa import Agent
from dataclasses import dataclass, replace
import numpy as np
from collections import deque

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
            strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
            g.strategy = rng.choice(strategies)
        if rng.random() < rate:
            g.max_age = int(max(50, g.max_age + rng.integers(-20, 21)))
        return g

class EcoAgent(Agent):
    """Агент без собственного метода step(). Логика вынесена в Model для синхронности."""
    
    def __init__(self, model, genome):
        super().__init__(model)
        self.genome = genome
        self.resource = 0.0
        self.age = 0
        
        # Базовая память
        self.last_action = "C"
        self.last_payoff = 0.0
        self.last_cell_coop_rate = 1.0
        
        # Память о партнерах и истории (Пункт 1.2)
        memory_size = getattr(self.model.cfg, "memory_size", 10)
        self.partners = {}  # {partner_id: {"last_action": "C"/"D", "last_seen": step}}
        self.interaction_history = deque(maxlen=memory_size)

    @property
    def alive(self):
        return self.resource > 0 and self.age < self.genome.max_age

    def get_action(self, current_partners=None):
        """Определяет действие агента на основе его стратегии и памяти."""
        strat = self.genome.strategy
        if strat in ("C", "AlwaysC"):
            return "C"
        elif strat in ("D", "AlwaysD"):
            return "D"
            
        elif strat == "TFT":
            # Tit-for-Tat: копирует поведение последних партнёров
            if current_partners and len(current_partners) > 1:
                remembered_actions = []
                for other in current_partners:
                    if other.unique_id != self.unique_id and other.unique_id in self.partners:
                        remembered_actions.append(self.partners[other.unique_id]["last_action"])
                if remembered_actions:
                    coop_rate = remembered_actions.count("C") / len(remembered_actions)
                    return "C" if coop_rate >= 0.5 else "D"
            # Fallback на общую память клетки
            return "C" if self.last_cell_coop_rate >= 0.5 else "D"
            
        elif strat == "GTFT":
            # Generous Tit-for-Tat: прощает дефекцию с вероятностью ~33%
            coop_rate = self.last_cell_coop_rate
            if current_partners and len(current_partners) > 1:
                remembered_actions = []
                for other in current_partners:
                    if other.unique_id != self.unique_id and other.unique_id in self.partners:
                        remembered_actions.append(self.partners[other.unique_id]["last_action"])
                if remembered_actions:
                    coop_rate = remembered_actions.count("C") / len(remembered_actions)
                    
            if coop_rate >= 0.5:
                return "C"
            else:
                return "C" if self.model.rng.random() < 0.33 else "D"
                
        elif strat == "WSLS":
            # Win-Stay, Lose-Shift (Pavlov)
            P = self.model.cfg.game.P
            # Используем историю, если она есть
            if self.interaction_history:
                last_mem = self.interaction_history[-1]
                if last_mem["payoff"] > P + 1e-6:
                    return last_mem["action"]
                else:
                    return "D" if last_mem["action"] == "C" else "C"
            # Fallback
            if self.last_payoff > P + 1e-6:
                return self.last_action
            else:
                return "D" if self.last_action == "C" else "C"
                
        return "C"

    def perceive_and_move(self):
        """Выбирает лучшую клетку в радиусе зрения."""
        vision = self.genome.vision
        x, y = self.pos
        env_resource = self.model.env.resource
        
        best_x, best_y = x, y
        best_val = env_resource[y, x]
        
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
        
        child_resource = self.resource / 2.0
        self.resource = child_resource
        
        child = EcoAgent(self.model, genome=child_genome)
        child.resource = child_resource
        return child
