from mesa import Agent
from dataclasses import dataclass, replace
import numpy as np
from collections import deque
import math

IMITATION_TYPES = ["none", "best_neighbor", "pairwise_diff", "proportional_m", "fermi_m"]

@dataclass
class Genome:
    vision: int
    metabolism: float
    strategy: str
    max_age: int
    # === НОВЫЕ ПОЛЯ ДЛЯ ПУНКТА 1.3 ===
    imitation_type: str = "none"          # Тип имитации (генетически обусловлен)
    imitation_intensity: float = 1.0      # Параметр m (чувствительность к разнице payoff)
    imitation_rate: float = 0.1           # Вероятность попытки имитации за шаг

    def mutate(self, rate, min_vision, max_vision, min_metabolism, max_metabolism,
               min_intensity, max_intensity, rng):
        g = replace(self)
        # Мутация зрения
        if rng.random() < rate:
            g.vision = int(np.clip(g.vision + rng.choice([-1, 1]), min_vision, max_vision))
        # Мутация метаболизма
        if rng.random() < rate:
            g.metabolism = float(
                np.clip(g.metabolism + rng.uniform(-0.5, 0.5), min_metabolism, max_metabolism)
            )
        # Мутация стратегии (редкая, для "консерваторов" это единственный путь)
        if rng.random() < rate:
            strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
            g.strategy = rng.choice(strategies)
        # Мутация максимального возраста
        if rng.random() < rate:
            g.max_age = int(max(50, g.max_age + rng.integers(-20, 21)))
        # === МУТАЦИИ ИМИТАЦИОННЫХ ПРИЗНАКОВ ===
        if rng.random() < rate:
            g.imitation_type = rng.choice(IMITATION_TYPES)
        if rng.random() < rate:
            g.imitation_intensity = float(np.clip(
                g.imitation_intensity + rng.uniform(-0.5, 0.5),
                min_intensity, max_intensity
            ))
        if rng.random() < rate:
            g.imitation_rate = float(np.clip(
                g.imitation_rate + rng.uniform(-0.05, 0.05),
                0.0, 1.0
            ))
        return g


class EcoAgent(Agent):
    """Агент с генетически обусловленным типом социального обучения."""

    def __init__(self, model, genome, group_id=0):
        super().__init__(model)
        self.genome = genome
        self.resource = 0.0
        self.age = 0
        self.group_id = group_id  # === Принадлежность к группе ===
        
        # Базовая память
        self.last_action = "C"
        self.last_payoff = 0.0
        self.last_cell_coop_rate = 1.0
        
        # === Позиция в социальном графе ===
        self.network_slot = None

        # Память о партнерах (Пункт 1.2)
        memory_size = getattr(self.model.cfg, "memory_size", 10)
        self.partners = {}
        self.interaction_history = deque(maxlen=memory_size)

        # === СЧЁТЧИКИ ДЛЯ АНАЛИТИКИ ИМИТАЦИИ ===
        self.imitation_attempts = 0
        self.imitation_successes = 0
        self.strategy_changes = 0  # Сколько раз сменил стратегию за жизнь

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
            if current_partners and len(current_partners) > 1:
                remembered_actions = []
                for other in current_partners:
                    if other.unique_id != self.unique_id and other.unique_id in self.partners:
                        remembered_actions.append(self.partners[other.unique_id]["last_action"])
                if remembered_actions:
                    coop_rate = remembered_actions.count("C") / len(remembered_actions)
                    return "C" if coop_rate >= 0.5 else "D"
            return "C" if self.last_cell_coop_rate >= 0.5 else "D"
        elif strat == "GTFT":
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
            P = self.model.cfg.game.P
            if self.interaction_history:
                last_mem = self.interaction_history[-1]
                if last_mem["payoff"] > P + 1e-6:
                    return last_mem["action"]
                else:
                    return "D" if last_mem["action"] == "C" else "C"
            if self.last_payoff > P + 1e-6:
                return self.last_action
            else:
                return "D" if self.last_action == "C" else "C"
        return "C"

    def perceive_and_move(self):
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
        rng = self.model.rng
        child_genome = self.genome.mutate(
            self.model.cfg.mutation_rate,
            self.model.cfg.min_vision, self.model.cfg.max_vision,
            self.model.cfg.min_metabolism, self.model.cfg.max_metabolism,
            self.model.cfg.min_imitation_intensity, self.model.cfg.max_imitation_intensity,
            rng,
        )
        child_resource = self.resource / 2.0
        self.resource = child_resource
        
        child = EcoAgent(self.model, genome=child_genome)
        child.resource = child_resource
        # network_slot будет назначен в model._evolution_step()
        return child

    # ============================================================
    # === БЛОК СОЦИАЛЬНОГО ОБУЧЕНИЯ (ПУНКТ 1.3) ===
    # ============================================================

    def try_imitate(self, neighbors: list):
        """
        Попытка социального обучения: агент может скопировать стратегию
        более успешного соседа согласно своему генетическому типу имитации.

        Все 5 протоколов из Izquierdo et al. (2019):
        - none: не имитирует (только генетическая передача)
        - best_neighbor: копирует стратегию лучшего соседа
        - pairwise_diff: попарная разница (replicator-like)
        - proportional_m: Моран-процесс (выбор пропорционален payoff^m)
        - fermi_m: Ферми/Logit-правило
        """
        itype = self.genome.imitation_type
        if itype == "none":
            return  # Консерватор не имитирует

        # Решаем, будет ли агент вообще пытаться имитировать в этом шаге
        if self.model.rng.random() >= self.genome.imitation_rate:
            return

        self.imitation_attempts += 1

        # Отфильтровываем соседей-агентов (исключая себя)
        other_agents = [a for a in neighbors if isinstance(a, EcoAgent) and a.unique_id != self.unique_id]
        if not other_agents:
            return

        new_strategy = None

        if itype == "best_neighbor":
            new_strategy = self._imitate_best_neighbor(other_agents)

        elif itype == "pairwise_diff":
            new_strategy = self._imitate_pairwise_difference(other_agents)

        elif itype == "proportional_m":
            new_strategy = self._imitate_proportional_moran(other_agents)

        elif itype == "fermi_m":
            new_strategy = self._imitate_fermi_logit(other_agents)

        # Применяем смену стратегии
        if new_strategy is not None and new_strategy != self.genome.strategy:
            self.genome.strategy = new_strategy
            self.imitation_successes += 1
            self.strategy_changes += 1

    def _imitate_best_neighbor(self, others):
        """
        Unconditional imitation (Nowak & May, 1992):
        Копирует стратегию соседа с НАИБОЛЬШИМ payoff,
        если его payoff больше собственного.
        """
        best = max(others, key=lambda a: a.last_payoff)
        if best.last_payoff > self.last_payoff:
            return best.genome.strategy
        return None

    def _imitate_pairwise_difference(self, others):
        """
        Imitative pairwise-difference protocol (Replicator-like):
        Выбирает случайного соседа. Копирует его стратегию
        с вероятностью (π_neighbor - π_self) / max_Δπ.
        """
        observed = self.model.rng.choice(others)

        max_diff = getattr(
            self.model.cfg,
            "max_payoff_difference",
            self.model.cfg.game.T - self.model.cfg.game.P
        )

        if max_diff <= 0:
            return None

        payoff_diff = observed.last_payoff - self.last_payoff

        if payoff_diff <= 0:
            return None

        prob = min(1.0, max(0.0, payoff_diff / max_diff))

        if self.model.rng.random() < prob:
            return observed.genome.strategy

        return None

    def _imitate_proportional_moran(self, others):
        """
        Imitative positive-proportional-m (Moran rule):
        Выбирает соседа с вероятностью, пропорциональной payoff^m,
        и копирует его стратегию.
        Требует неотрицательных payoff (сдвигаем если нужно).
        """
        m = self.genome.imitation_intensity
        # Сдвигаем payoff чтобы они были неотрицательными
        payoffs = np.array([max(0.0, a.last_payoff) for a in others], dtype=float)
        weights = np.power(payoffs, m)
        total = weights.sum()
        if total <= 0:
            return None
        probs = weights / total
        idx = self.model.rng.choice(len(others), p=probs)
        return others[idx].genome.strategy

    def _imitate_fermi_logit(self, others):
        """
        Imitative logit-m (Fermi rule / Logit dynamics):
        Выбирает случайного соседа. Копирует его стратегию
        с вероятностью 1 / (1 + exp(-m * (π_neighbor - π_self))).
        """
        m = self.genome.imitation_intensity
        observed = self.model.rng.choice(others)
        payoff_diff = observed.last_payoff - self.last_payoff
        # Защита от overflow
        exponent = -m * payoff_diff
        exponent = np.clip(exponent, -500, 500)
        prob = 1.0 / (1.0 + math.exp(exponent))
        if self.model.rng.random() < prob:
            return observed.genome.strategy
        return None
