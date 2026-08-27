import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from .environment import DynamicEnvironment
from .agent import EcoAgent, Genome, IMITATION_TYPES
from .managers import (
    NetworkManager,
    InteractionManager,
    TradeManager,
    ImitationManager,
    EvolutionManager
)

def compute_stats(model):
    n = len(model.agents)
    if n == 0: return _empty_stats()

    sum_sugar = sum_spice = sum_vis = 0.0
    sum_met_s = sum_met_sp = 0.0
    sum_propensity_c = 0.0  # Накопитель реальной склонности к кооперации
    
    strat_counts = {s: 0 for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]}
    imit_type_counts = {t: 0 for t in IMITATION_TYPES}
    imit_type_payoff = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_resource = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_attempts = {t: 0 for t in IMITATION_TYPES}
    imit_type_successes = {t: 0 for t in IMITATION_TYPES}
    sum_intensity = 0.0
    sum_imitation_rate = 0.0
    sum_learning_rate = 0.0
    sum_exploration_rate = 0.0
    sum_prop_c = 0.0
    sum_prop_d = 0.0

    for a in model.agents:
        sum_sugar += a.sugar
        sum_spice += a.spice
        sum_vis += a.genome.vision
        sum_met_s += a.genome.metabolism_sugar
        sum_met_sp += a.genome.metabolism_spice
        sum_intensity += a.genome.imitation_intensity
        sum_imitation_rate += a.genome.imitation_rate

        # Защита от TypeError (если стратегия вдруг стала не строкой)
        strat = a.genome.strategy if isinstance(a.genome.strategy, str) else "Unknown"
        if strat in strat_counts: strat_counts[strat] += 1
            
        # === СТАТИСТИКА ОБУЧЕННОГО ПОВЕДЕНИЯ (Roth-Erev) ===
        total_prop = a.propensities["C"] + a.propensities["D"]
        p_c = a.propensities["C"] / total_prop if total_prop > 0 else 0.5
        sum_propensity_c += p_c

        itype = a.genome.imitation_type
        imit_type_counts[itype] += 1
        # Считаем средний накопленный выигрыш (K шагов), а не last_payoff
        imit_type_payoff[itype] += a.accumulated_payoff 
        imit_type_resource[itype] += (a.sugar + a.spice)
        imit_type_attempts[itype] += getattr(a, "imitation_attempts", 0)
        imit_type_successes[itype] += getattr(a, "imitation_successes", 0)

        # Собираем гены психики
        sum_learning_rate += a.genome.learning_rate
        sum_exploration_rate += a.genome.exploration_rate
        
        # Собираем реальные склонности (propensities) для графика обучения
        total_prop = a.propensities["C"] + a.propensities["D"]
        if total_prop > 0:
            sum_prop_c += a.propensities["C"] / total_prop
            sum_prop_d += a.propensities["D"] / total_prop
        else:
            sum_prop_c += 0.5
            sum_prop_d += 0.5

    # Реальная частота кооператоров теперь основана на выученном поведении
    n_action_c = sum(1 for a in model.agents if getattr(a, "last_action", "C") == "C")
    
    stats = {
        "Population": n, 
        "Freq_Cooperators": sum_propensity_c / n, # Реальная частота кооперации
        "Freq_Action_C": n_action_c / n,
        "Avg_Sugar": sum_sugar / n, "Avg_Spice": sum_spice / n, "Avg_Vision": sum_vis / n,
        "Avg_Metabolism_Sugar": sum_met_s / n, "Avg_Metabolism_Spice": sum_met_sp / n,
        "Avg_Imitation_Intensity": sum_intensity / n, "Avg_Imitation_Rate": sum_imitation_rate / n,
        "Total_Pollution": model.env.total_pollution if model.cfg.pollution_enabled else 0.0,
    }
    for s, cnt in strat_counts.items(): stats[f"Freq_{s}"] = cnt / n
    for t in IMITATION_TYPES:
        stats[f"ImitFreq_{t}"] = imit_type_counts[t] / n
        stats[f"ImitAvgPayoff_{t}"] = imit_type_payoff[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        stats[f"ImitAvgResource_{t}"] = imit_type_resource[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        attempts = imit_type_attempts[t]
        stats[f"ImitSuccessRate_{t}"] = imit_type_successes[t] / attempts if attempts > 0 else 0.0

    if model.cfg.group_selection_enabled:
        group_res = {}
        for a in model.agents: group_res.setdefault(a.group_id, []).append(a.sugar + a.spice)
        stats["Alive_Groups"] = len(group_res)
        if len(group_res) > 1:
            means = [sum(v)/len(v) for v in group_res.values()]
            stats["Group_Fitness_Variance"] = float(np.var(means))
        else: stats["Group_Fitness_Variance"] = 0.0
    else:
        stats["Alive_Groups"] = 1
        stats["Group_Fitness_Variance"] = 0.0

    stats["Avg_Learning_Rate"] = sum_learning_rate / n
    stats["Avg_Exploration_Rate"] = sum_exploration_rate / n
    stats["Avg_Propensity_C"] = sum_prop_c / n
    stats["Avg_Propensity_D"] = sum_prop_d / n

    return stats

def _empty_stats():
    stats = {
        "Population": 0,
        "Freq_Cooperators": 0.0,
        "Freq_Action_C": 0.0,
        "Avg_Sugar": 0.0,
        "Avg_Spice": 0.0,
        "Avg_Vision": 0.0,
        "Avg_Metabolism_Sugar": 0.0,
        "Avg_Metabolism_Spice": 0.0,
        "Avg_Imitation_Intensity": 0.0,
        "Avg_Imitation_Rate": 0.0,
        "Alive_Groups": 1,
        "Group_Fitness_Variance": 0.0,
        "Total_Pollution": 0.0,
        "Avg_Learning_Rate": 0.0,
        "Avg_Exploration_Rate": 0.0,
        "Avg_Propensity_C": 0.5,
        "Avg_Propensity_D": 0.5,
    }
    for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]:
        stats[f"Freq_{s}"] = 0.0
    for t in IMITATION_TYPES:
        stats[f"ImitFreq_{t}"] = 0.0
        stats[f"ImitAvgPayoff_{t}"] = 0.0
        stats[f"ImitAvgResource_{t}"] = 0.0
        stats[f"ImitSuccessRate_{t}"] = 0.0
    return stats


class AgentsModel(Model):
    def __init__(self, seed=None, **kwargs):
        from .config import Config
        if "cfg" in kwargs and isinstance(kwargs["cfg"], Config): self.cfg = kwargs["cfg"]
        else: self.cfg = Config(**kwargs)

        self._seed = seed if seed is not None else self.cfg.seed
        self.rng = np.random.default_rng(self._seed)

        try: super().__init__(rng=self._seed)
        except TypeError: super().__init__(seed=self._seed)

        self.grid = MultiGrid(self.cfg.width, self.cfg.height, torus=True)
        self._neighborhood_offsets = {}
        max_r = self.cfg.max_vision
        for r in range(1, max_r + 1):
            offsets = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    offsets.append((dx, dy))
            self._neighborhood_offsets[r] = offsets
            
        self.env = DynamicEnvironment(
            width=self.cfg.width, height=self.cfg.height,
            max_resource=self.cfg.max_resource, regen_rate=self.cfg.regen_rate,
            max_spice=self.cfg.max_spice, regen_rate_spice=self.cfg.regen_rate_spice,
            season_period=self.cfg.season_period, season_amplitude=self.cfg.season_amplitude,
            catastrophe_prob=self.cfg.catastrophe_prob, catastrophe_duration=self.cfg.catastrophe_duration,
            catastrophe_severity=self.cfg.catastrophe_severity, rng=self.rng,
            pollution_enabled=self.cfg.pollution_enabled, pollution_diffusion_rate=self.cfg.pollution_diffusion_rate,
            pollution_decay_rate=self.cfg.pollution_decay_rate, pollution_capacity_impact=self.cfg.pollution_capacity_impact,
            # Новые параметры
            resource_peaks_drift_speed=self.cfg.resource_peaks_drift_speed,
            resource_peaks_mutation_prob=self.cfg.resource_peaks_mutation_prob,
            island_model_enabled=self.cfg.island_model_enabled,
            islands_count=self.cfg.islands_count
        )

        # Инициализация Менеджеров
        self.network_manager = NetworkManager(self)
        self.interaction_manager = InteractionManager(self)
        self.trade_manager = TradeManager(self)
        self.imitation_manager = ImitationManager(self)
        self.evolution_manager = EvolutionManager(self)

        self.max_slots = 0
        strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
        num_groups = self.cfg.num_groups
        gs_enabled = self.cfg.group_selection_enabled
        
        for i in range(self.cfg.initial_agents):
            genome = Genome(
                vision=int(self.rng.integers(self.cfg.min_vision, self.cfg.max_vision + 1)),
                metabolism_sugar=float(self.rng.uniform(self.cfg.min_metabolism, self.cfg.max_metabolism)),
                metabolism_spice=float(self.rng.uniform(self.cfg.min_metabolism_spice, self.cfg.max_metabolism_spice)),
                strategy=str(self.rng.choice(strategies)), max_age=self.cfg.max_age,
                imitation_type=str(self.rng.choice(IMITATION_TYPES)),
                imitation_intensity=float(self.rng.uniform(self.cfg.min_imitation_intensity, self.cfg.max_imitation_intensity)),
                imitation_rate=float(self.rng.uniform(0.05, self.cfg.initial_imitation_rate + 0.1)),
            )
            x = int(self.rng.integers(0, self.cfg.width))
            y = int(self.rng.integers(0, self.cfg.height))
            
            group_id = i % num_groups if gs_enabled else 0
            agent = EcoAgent(self, genome=genome, group_id=group_id)
            agent.sugar = self.cfg.initial_resource
            agent.spice = self.cfg.initial_spice
            agent.network_slot = self.max_slots
            self.max_slots += 1
            self.grid.place_agent(agent, (x, y))
            if self.cfg.island_model_enabled:
                agent.home_island = self.env.island_map[y, x]

        self.network_manager.build_network(self.max_slots)

        reporters = {
            "Population": lambda m: m._stats["Population"],
            "Total_Env_Sugar": lambda m: m.env.total_sugar,
            "Total_Env_Spice": lambda m: m.env.total_spice,
            "Total_Pollution": lambda m: m._stats["Total_Pollution"],
            "Season_Phase": lambda m: m.env.season_phase,
            "Catastrophe_Active": lambda m: 1 if m.env.catastrophe_active else 0,
            "Freq_Cooperators": lambda m: m._stats["Freq_Cooperators"],
            "Freq_Action_C": lambda m: m._stats["Freq_Action_C"],
            "Avg_Sugar": lambda m: m._stats["Avg_Sugar"],
            "Avg_Spice": lambda m: m._stats["Avg_Spice"],
            "Avg_Vision": lambda m: m._stats["Avg_Vision"],
            "Avg_Metabolism_Sugar": lambda m: m._stats["Avg_Metabolism_Sugar"],
            "Avg_Metabolism_Spice": lambda m: m._stats["Avg_Metabolism_Spice"],
            "Avg_Imitation_Intensity": lambda m: m._stats["Avg_Imitation_Intensity"],
            "Avg_Imitation_Rate": lambda m: m._stats["Avg_Imitation_Rate"],
            "Alive_Groups": lambda m: m._stats["Alive_Groups"],
            "Group_Fitness_Variance": lambda m: m._stats["Group_Fitness_Variance"],
            "Avg_Learning_Rate": lambda m: m._stats["Avg_Learning_Rate"],
            "Avg_Exploration_Rate": lambda m: m._stats["Avg_Exploration_Rate"],
            "Avg_Propensity_C": lambda m: m._stats["Avg_Propensity_C"],
            "Avg_Propensity_D": lambda m: m._stats["Avg_Propensity_D"],
        }
        for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]:
            reporters[f"Freq_{s}"] = lambda m, _s=s: m._stats[f"Freq_{_s}"]
        for t in IMITATION_TYPES:
            reporters[f"ImitFreq_{t}"] = lambda m, _t=t: m._stats[f"ImitFreq_{_t}"]
            reporters[f"ImitAvgPayoff_{t}"] = lambda m, _t=t: m._stats[f"ImitAvgPayoff_{_t}"]
            reporters[f"ImitAvgResource_{t}"] = lambda m, _t=t: m._stats[f"ImitAvgResource_{_t}"]
            reporters[f"ImitSuccessRate_{t}"] = lambda m, _t=t: m._stats[f"ImitSuccessRate_{_t}"]

        self.datacollector = DataCollector(model_reporters=reporters)
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run = 0

    def get_neighborhood_cells(self, pos, radius):
        x, y = pos
        w, h = self.cfg.width, self.cfg.height
        offsets = self._neighborhood_offsets.get(radius, [])
        return list(dict.fromkeys(((x + dx) % w, (y + dy) % h) for dx, dy in offsets))

    def step(self):
        self.env.step()
        active_agents = [a for a in self.agents if a.alive]

        for agent in active_agents:
            agent.age += 1
            agent.perceive_and_move()
            agent.migrate() # <-- Вызов логики миграции между островами (П. 2.3)

        pollution_enabled = self.cfg.pollution_enabled
        cons_rate = self.cfg.pollution_consumption_rate
        
        pollution_grid = np.zeros((self.cfg.height, self.cfg.width))
        self.interaction_manager.harvest_cells(active_agents, pollution_grid)

        if self.cfg.network_type != "none" and self.network_manager.social_network is not None:
            self.network_manager.interact_network(active_agents)
        else:
            self.interaction_manager.interact_cells_aggregated(active_agents)

        self.imitation_manager.imitation_step(active_agents)

        if self.cfg.group_selection_enabled:
            alpha = self.cfg.group_selection_intensity
            if alpha > 0:
                group_payoffs, group_counts = {}, {}
                for a in active_agents:
                    gid = a.group_id
                    group_payoffs[gid] = group_payoffs.get(gid, 0.0) + a.last_payoff
                    group_counts[gid] = group_counts.get(gid, 0) + 1
                group_avg = {gid: group_payoffs[gid] / group_counts[gid] for gid in group_payoffs}
                for a in active_agents:
                    avg_g = group_avg.get(a.group_id, a.last_payoff)
                    a.sugar += alpha * (avg_g - a.last_payoff)

        self.trade_manager.trade_step(active_agents)

        for agent in active_agents: 
            met_s, met_sp = agent.metabolize()
            if pollution_enabled:
                x, y = agent.pos
                pollution_grid[y, x] += (met_s + met_sp) * cons_rate
                
        if pollution_enabled:
            self.env.add_pollution(pollution_grid)

        if self.cfg.group_selection_enabled:
            comp_step = self.cfg.group_competition_step
            if comp_step > 0 and self.steps_run > 0 and self.steps_run % comp_step == 0:
                self.interaction_manager.group_competition_step()

        self.evolution_manager.evolution_step()
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run += 1

    def run_model(self, steps, log_every=25, max_seconds=None):
        import time
        start = time.time()
        for _ in range(steps):
            if len(self.agents) == 0:
                print(f"Популяция вымерла на шаге {self.steps_run}", flush=True)
                break
            step_start = time.time()
            self.step()
            step_time = time.time() - step_start
            if log_every and self.steps_run % log_every == 0:
                edges = self.network_manager.num_edges
                print(f"[step {self.steps_run}] agents={len(self.agents)}, edges={edges}, step_time={step_time:.3f}s", flush=True)
            if max_seconds is not None and max_seconds > 0:
                elapsed = time.time() - start
                if elapsed > max_seconds:
                    print(f"Симуляция остановлена по лимиту времени: {elapsed:.1f}s > {max_seconds:.1f}s", flush=True)
                    break
