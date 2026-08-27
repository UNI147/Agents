from mesa import Agent
from dataclasses import dataclass, replace
import numpy as np
from collections import deque
import math

IMITATION_TYPES = ["none", "best_neighbor", "pairwise_diff", "proportional_m", "fermi_m"]

@dataclass
class Genome:
    vision: int
    metabolism_sugar: float
    metabolism_spice: float
    strategy: str
    max_age: int
    imitation_type: str = "none"
    imitation_intensity: float = 1.0
    imitation_rate: float = 0.1

    def mutate(self, rate, min_vision, max_vision, min_met_s, max_met_s,
               min_met_sp, max_met_sp, min_intensity, max_intensity, rng):
        g = replace(self)
        if rng.random() < rate:
            g.vision = int(np.clip(g.vision + rng.choice([-1, 1]), min_vision, max_vision))
        if rng.random() < rate:
            g.metabolism_sugar = float(np.clip(
                g.metabolism_sugar + rng.uniform(-0.5, 0.5), min_met_s, max_met_s))
        if rng.random() < rate:
            g.metabolism_spice = float(np.clip(
                g.metabolism_spice + rng.uniform(-0.5, 0.5), min_met_sp, max_met_sp))
        if rng.random() < rate:
            strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
            g.strategy = rng.choice(strategies)
        if rng.random() < rate:
            g.max_age = int(max(50, g.max_age + rng.integers(-20, 21)))
        if rng.random() < rate:
            g.imitation_type = rng.choice(IMITATION_TYPES)
        if rng.random() < rate:
            g.imitation_intensity = float(np.clip(
                g.imitation_intensity + rng.uniform(-0.5, 0.5),
                min_intensity, max_intensity))
        if rng.random() < rate:
            g.imitation_rate = float(np.clip(
                g.imitation_rate + rng.uniform(-0.05, 0.05), 0.0, 1.0))
        return g


class EcoAgent(Agent):
    def __init__(self, model, genome, group_id=0):
        super().__init__(model)
        self.genome = genome
        self.sugar = 0.0
        self.spice = 0.0
        self.age = 0
        self.group_id = group_id
        self.last_action = "C"
        self.last_payoff = 0.0
        self.last_cell_coop_rate = 1.0
        self.network_slot = None
        memory_size = getattr(self.model.cfg, "memory_size", 10)
        self.partners = {}
        self.interaction_history = deque(maxlen=memory_size)
        self.imitation_attempts = 0
        self.imitation_successes = 0
        self.strategy_changes = 0

    @property
    def alive(self):
        """
        ИЗМЕНЕНИЕ: Допускаем нулевой ресурс как "голодание" с порогом -1.0.
        Это моделирует биологический механизм: организм погибает не мгновенно
        при нулевом запасе, а после истощения резервов (жира, гликогена).
        """
        return (self.sugar > -1.0 and self.spice > -1.0
                and self.age < self.genome.max_age)

    def welfare(self, w_sugar, w_spice):
        m1 = self.genome.metabolism_sugar
        m2 = self.genome.metabolism_spice
        mT = m1 + m2
        if mT <= 0:
            return 0.0
        w1 = max(1e-6, w_sugar)
        w2 = max(1e-6, w_spice)
        return (w1 ** (m1 / mT)) * (w2 ** (m2 / mT))

    def calculate_mrs(self):
        m1 = self.genome.metabolism_sugar
        m2 = self.genome.metabolism_spice
        w1 = max(1e-6, self.sugar)
        w2 = max(1e-6, self.spice)
        if m2 <= 1e-6:
            return float('inf')
        if m1 <= 1e-6:
            return 0.0
        return (m1 * w2) / (m2 * w1)

    def get_action(self, current_partners=None):
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
        env_sugar = self.model.env.sugar
        env_spice = self.model.env.spice
        best_x, best_y = x, y
        best_welfare = self.welfare(
            self.sugar + env_sugar[y, x], self.spice + env_spice[y, x])
        neighbors = self.model.get_neighborhood_cached(self.pos, vision)
        for nx, ny in neighbors:
            welf = self.welfare(
                self.sugar + env_sugar[ny, nx], self.spice + env_spice[ny, nx])
            if welf > best_welfare:
                best_welfare = welf
                best_x, best_y = nx, ny
        if (best_x, best_y) != self.pos:
            self.model.grid.move_agent(self, (best_x, best_y))

    def metabolize(self):
        """
        ИЗМЕНЕНИЕ: Метаболическая адаптация при голодании.
        Биологическое обоснование: при дефиците ресурсов организмы
        снижают базальный метаболизм (торпор, диапауза, снижение
        активности щитовидной железы у млекопитающих).
        При ресурсе < 3.0 метаболизм снижается до 60% от нормы.
        """
        met_s = self.genome.metabolism_sugar
        met_sp = self.genome.metabolism_spice

        # Адаптивное снижение метаболизма при голодании
        if self.sugar < 3.0:
            met_s *= 0.6
        if self.spice < 3.0:
            met_sp *= 0.6

        self.sugar -= met_s
        self.spice -= met_sp

    def can_reproduce(self):
        threshold = self.model.cfg.reproduction_threshold
        return self.alive and (self.sugar + self.spice) > threshold

    def reproduce(self):
        rng = self.model.rng
        child_genome = self.genome.mutate(
            self.model.cfg.mutation_rate,
            self.model.cfg.min_vision, self.model.cfg.max_vision,
            self.model.cfg.min_metabolism, self.model.cfg.max_metabolism,
            self.model.cfg.min_metabolism_spice, self.model.cfg.max_metabolism_spice,
            self.model.cfg.min_imitation_intensity, self.model.cfg.max_imitation_intensity,
            rng,
        )
        child_sugar = self.sugar / 2.0
        child_spice = self.spice / 2.0
        self.sugar = child_sugar
        self.spice = child_spice
        child = EcoAgent(self.model, genome=child_genome)
        child.sugar = child_sugar
        child.spice = child_spice
        return child

    def trade(self, other):
        if not getattr(self.model.cfg, "trade_enabled", False):
            return
        mrs_self = self.calculate_mrs()
        mrs_other = other.calculate_mrs()
        if math.isinf(mrs_self) or math.isinf(mrs_other):
            return
        if math.isclose(mrs_self, mrs_other, rel_tol=1e-4):
            return
        if mrs_self > mrs_other:
            buyer, seller = self, other
        else:
            buyer, seller = other, self
        max_iter = 10
        for _ in range(max_iter):
            mrs_b = buyer.calculate_mrs()
            mrs_s = seller.calculate_mrs()
            if mrs_b <= mrs_s or math.isinf(mrs_b) or math.isinf(mrs_s):
                break
            p = math.sqrt(mrs_b * mrs_s)
            if p >= 1.0:
                dsugar = 1.0
                dspice = p
            else:
                dsugar = 1.0 / p
                dspice = 1.0
            if buyer.spice < dspice or seller.sugar < dsugar:
                break
            w_b_old = buyer.welfare(buyer.sugar, buyer.spice)
            w_s_old = seller.welfare(seller.sugar, seller.spice)
            w_b_new = buyer.welfare(buyer.sugar + dsugar, buyer.spice - dspice)
            w_s_new = seller.welfare(seller.sugar - dsugar, seller.spice + dspice)
            if w_b_new > w_b_old and w_s_new > w_s_old:
                buyer.sugar += dsugar
                buyer.spice -= dspice
                seller.sugar -= dsugar
                seller.spice += dspice
            else:
                break

    def try_imitate(self, neighbors: list):
        itype = self.genome.imitation_type
        if itype == "none":
            return
        if self.model.rng.random() >= self.genome.imitation_rate:
            return
        self.imitation_attempts += 1
        other_agents = [a for a in neighbors
                        if isinstance(a, EcoAgent) and a.unique_id != self.unique_id]
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
        if new_strategy is not None and new_strategy != self.genome.strategy:
            self.genome.strategy = new_strategy
            self.imitation_successes += 1
            self.strategy_changes += 1

    def _imitate_best_neighbor(self, others):
        best = max(others, key=lambda a: a.last_payoff)
        if best.last_payoff > self.last_payoff:
            return best.genome.strategy
        return None

    def _imitate_pairwise_difference(self, others):
        observed = self.model.rng.choice(others)
        max_diff = getattr(self.model.cfg, "max_payoff_difference",
                           self.model.cfg.game.T - self.model.cfg.game.P)
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
        m = self.genome.imitation_intensity
        payoffs = np.array([max(0.0, a.last_payoff) for a in others], dtype=float)
        weights = np.power(payoffs, m)
        total = weights.sum()
        if total <= 0:
            return None
        probs = weights / total
        idx = self.model.rng.choice(len(others), p=probs)
        return others[idx].genome.strategy

    def _imitate_fermi_logit(self, others):
        m = self.genome.imitation_intensity
        observed = self.model.rng.choice(others)
        payoff_diff = observed.last_payoff - self.last_payoff
        exponent = -m * payoff_diff
        exponent = np.clip(exponent, -500, 500)
        prob = 1.0 / (1.0 + math.exp(exponent))
        if self.model.rng.random() < prob:
            return observed.genome.strategy
        return None
