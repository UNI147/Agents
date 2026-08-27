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
    max_age: int
    imitation_type: str = "none"
    imitation_intensity: float = 1.0
    imitation_rate: float = 0.1
    
    # === ГЕНЫ ОБУЧЕНИЯ (Roth-Erev) ===
    learning_rate: float = 1.0       # Масштаб подкрепления
    propensity_decay: float = 0.95   # Параметр забывания (recency)
    exploration_rate: float = 0.05   # Параметр эксперимента (эксплорация)

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
        memory_size = self.model.cfg.memory_size
        self.partners = {}
        self.interaction_history = deque(maxlen=memory_size)
        self.imitation_attempts = 0
        self.imitation_successes = 0
        self.strategy_changes = 0

    @property
    def accumulated_payoff(self):
        """Накопленный социальный выигрыш за последние K шагов (memory_size)."""
        if not self.interaction_history:
            return 0.0
        return sum(h["payoff"] for h in self.interaction_history)

    @property
    def alive(self):
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
        """Выбор действия на основе накопленных склонностей + шум."""
        if self.model.rng.random() < self.genome.exploration_rate:
            return self.model.rng.choice(["C", "D"])
            
        total = self.propensities["C"] + self.propensities["D"]
        if total <= 0: return self.model.rng.choice(["C", "D"])
            
        p_c = self.propensities["C"] / total
        return "C" if self.model.rng.random() < p_c else "D"

    def update_learning(self, action_played, payoff):
        """Обновление по правилу Рота-Эрева"""
        e = self.genome.exploration_rate
        decay = self.genome.propensity_decay
        lr = self.genome.learning_rate
        
        if payoff > 0:
            # Сыгранное действие получает (1-e)*payoff, альтернативное e*payoff
            main_update = payoff * (1.0 - e) * lr
            alt_update = payoff * e * lr
            
            if action_played == "C":
                self.propensities["C"] += main_update
                self.propensities["D"] += alt_update
            else:
                self.propensities["D"] += main_update
                self.propensities["C"] += alt_update
                
        # Эффект забывания (decay)
        self.propensities["C"] *= decay
        self.propensities["D"] *= decay
        
        self.propensities["C"] = max(0.01, self.propensities["C"])
        self.propensities["D"] = max(0.01, self.propensities["D"])

    def perceive_and_move(self):
        vision = self.genome.vision
        x, y = self.pos
        env_sugar = self.model.env.sugar
        env_spice = self.model.env.spice
        best_x, best_y = x, y
        best_welfare = self.welfare(
            self.sugar + env_sugar[y, x], self.spice + env_spice[y, x])
        neighbors = self.model.get_neighborhood_cells(self.pos, vision)
        for nx, ny in neighbors:
            welf = self.welfare(
                self.sugar + env_sugar[ny, nx], self.spice + env_spice[ny, nx])
            if welf > best_welfare:
                best_welfare = welf
                best_x, best_y = nx, ny
        if (best_x, best_y) != self.pos:
            self.model.grid.move_agent(self, (best_x, best_y))

    def metabolize(self):
        met_s = self.genome.metabolism_sugar
        met_sp = self.genome.metabolism_spice

        if self.sugar < 3.0: met_s *= 0.6
        if self.spice < 3.0: met_sp *= 0.6

        # === БИОЛОГИЧЕСКИЕ ЗАТРАТЫ НА ПОДДЕРЖАНИЕ ОРГАНОВ И МОЗГА ===
        # 1. Зрение требует энергии (больше радиус = выше стоимость)
        vision_cost = 0.1 * self.genome.vision
        
        # 2. Нервная ткань (память о социальных партнерах) требует энергии
        memory_cost = 0.1 * len(self.partners)

        # Итоговые затраты распределяются между sugar и spice
        total_met_s = met_s + vision_cost + memory_cost
        total_met_sp = met_sp + (vision_cost + memory_cost) * 0.5

        self.sugar -= total_met_s
        self.spice -= total_met_sp
        
        return total_met_s, total_met_sp

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
        if not self.model.cfg.trade_enabled:
            return
            
        m1_b, m2_b = self.genome.metabolism_sugar, self.genome.metabolism_spice
        m1_s, m2_s = other.genome.metabolism_sugar, other.genome.metabolism_spice
        
        # Избегаем деления на ноль и логарифмических ошибок
        w1_b, w2_b = max(1e-6, self.sugar), max(1e-6, self.spice)
        w1_s, w2_s = max(1e-6, other.sugar), max(1e-6, other.spice)
        
        mT_b = m1_b + m2_b
        mT_s = m1_s + m2_s
        
        if mT_b <= 1e-6 or mT_s <= 1e-6:
            return

        # Быстрая проверка MRS для определения ролей (Buyer/Seller)
        # MRS = (m1 * w2) / (m2 * w1)
        mrs_self = (m1_b * w2_b) / (m2_b * w1_b) if m2_b > 1e-6 else float('inf')
        mrs_other = (m1_s * w2_s) / (m2_s * w1_s) if m2_s > 1e-6 else float('inf')
        
        if math.isinf(mrs_self) or math.isinf(mrs_other):
            return
        if math.isclose(mrs_self, mrs_other, rel_tol=1e-4):
            return
            
        # Buyer покупает sugar (w1), продает spice (w2)
        if mrs_self > mrs_other:
            buyer, seller = self, other
        else:
            buyer, seller = other, self
            # Меняем местами переменные для формулы
            m1_b, m2_b, mT_b, w1_b, w2_b = m1_s, m2_s, mT_s, w1_s, w2_s
            m1_s, m2_s, mT_s, w1_s, w2_s = m1_b, m2_b, mT_b, w1_b, w2_b

        # --- ТОЧНОЕ АНАЛИТИЧЕСКОЕ РЕШЕНИЕ (Кривая контрактов) ---
        N = (m1_b * w2_b * m2_s * w1_s) - (m1_s * w2_s * m2_b * w1_b)
        D = (m1_b * w2_b * mT_s) + (m1_s * w2_s * mT_b)
        
        if D <= 1e-9:
            return
            
        delta_sugar = N / D
        if delta_sugar <= 1e-6:
            return
            
        delta_spice = (delta_sugar * m1_b * w2_b) / (m2_b * w1_b + mT_b * delta_sugar)
        
        # Проверка физических ограничений (нельзя продать больше, чем есть)
        max_sugar = seller.sugar * 0.9999
        max_spice = buyer.spice * 0.9999
        
        # Если упремся в границы запасов, масштабируем сделку пропорционально
        if delta_sugar > max_sugar or delta_spice > max_spice:
            scale = min(max_sugar / delta_sugar, max_spice / delta_spice)
            delta_sugar *= scale
            delta_spice *= scale
            
        # Применяем транзакцию
        buyer.sugar += delta_sugar
        buyer.spice -= delta_spice
        seller.sugar -= delta_sugar
        seller.spice += delta_spice

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
        best = max(others, key=lambda a: a.accumulated_payoff)
        if best.accumulated_payoff > self.accumulated_payoff:
            return best.propensities # Возвращаем не строку стратегии, а накопленный опыт
        return None

    def _imitate_pairwise_difference(self, others):
        observed = self.model.rng.choice(others)
        max_diff = self.model.cfg.max_payoff_difference
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
