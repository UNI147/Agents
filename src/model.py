from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np

from .agent import EcoAgent, Genome
from .environment import DynamicEnvironment

def compute_stats(model):
    """Единый проход по агентам для сбора всей статистики."""
    n = len(model.agents)
    if n == 0:
        return {
            "Population": 0, "Freq_Cooperators": 0.0, "Freq_Action_C": 0.0,
            "Avg_Agent_Resource": 0.0, "Avg_Vision": 0.0, "Avg_Metabolism": 0.0,
            "Freq_AlwaysC": 0.0, "Freq_AlwaysD": 0.0, "Freq_TFT": 0.0, 
            "Freq_WSLS": 0.0, "Freq_GTFT": 0.0
        }
    
    sum_res = sum_vis = sum_met = n_c = n_action_c = 0
    n_always_c = n_always_d = n_tft = n_wsls = n_gtft = 0
    
    for a in model.agents:
        sum_res += a.resource
        sum_vis += a.genome.vision
        sum_met += a.genome.metabolism
        
        strat = a.genome.strategy
        if strat in ("C", "AlwaysC"):
            n_always_c += 1
            n_c += 1
        elif strat in ("D", "AlwaysD"):
            n_always_d += 1
        elif strat == "TFT":
            n_tft += 1
            n_c += 1 
        elif strat == "WSLS":
            n_wsls += 1
        elif strat == "GTFT":
            n_gtft += 1
            n_c += 1
            
        if getattr(a, "last_action", "C") == "C":
            n_action_c += 1

    return {
        "Population": n,
        "Freq_Cooperators": n_c / n,
        "Freq_Action_C": n_action_c / n,
        "Avg_Agent_Resource": sum_res / n,
        "Avg_Vision": sum_vis / n,
        "Avg_Metabolism": sum_met / n,
        "Freq_AlwaysC": n_always_c / n,
        "Freq_AlwaysD": n_always_d / n,
        "Freq_TFT": n_tft / n,
        "Freq_WSLS": n_wsls / n,
        "Freq_GTFT": n_gtft / n,
    }

class AgentsModel(Model):
    def __init__(self, seed=None, **kwargs):
        from .config import Config
        if "cfg" in kwargs and isinstance(kwargs["cfg"], Config):
            self.cfg = kwargs["cfg"]
        else:
            self.cfg = Config(**kwargs)
            
        self._seed = seed if seed is not None else getattr(self.cfg, "seed", 42)
        self.rng = np.random.default_rng(self._seed)
        
        try:
            super().__init__(rng=self._seed)
        except TypeError:
            super().__init__(seed=self._seed)
            
        self.grid = MultiGrid(self.cfg.width, self.cfg.height, torus=True)
        self.env = DynamicEnvironment(
            width=self.cfg.width,
            height=self.cfg.height,
            max_resource=self.cfg.max_resource,
            regen_rate=self.cfg.regen_rate,
            season_period=self.cfg.season_period,
            season_amplitude=self.cfg.season_amplitude,
            catastrophe_prob=self.cfg.catastrophe_prob,
            catastrophe_duration=self.cfg.catastrophe_duration,
            catastrophe_severity=self.cfg.catastrophe_severity,
            rng=self.rng,
        )
        
        self._neighborhood_cache = {}
        strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
        
        for _ in range(self.cfg.initial_agents):
            genome = Genome(
                vision=int(self.rng.integers(self.cfg.min_vision, self.cfg.max_vision + 1)),
                metabolism=float(self.rng.uniform(self.cfg.min_metabolism, self.cfg.max_metabolism)),
                strategy=str(self.rng.choice(strategies)),
                max_age=self.cfg.max_age,
            )
            x = int(self.rng.integers(0, self.cfg.width))
            y = int(self.rng.integers(0, self.cfg.height))
            agent = EcoAgent(self, genome=genome)
            agent.resource = self.cfg.initial_resource
            self.grid.place_agent(agent, (x, y))
            
        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: m._stats["Population"],
                "Total_Env_Resource": lambda m: m.env.total_resource,
                "Season_Phase": lambda m: m.env.season_phase,
                "Catastrophe_Active": lambda m: 1 if m.env.catastrophe_active else 0,
                "Freq_Cooperators": lambda m: m._stats["Freq_Cooperators"],
                "Freq_Action_C": lambda m: m._stats["Freq_Action_C"],
                "Avg_Agent_Resource": lambda m: m._stats["Avg_Agent_Resource"],
                "Avg_Vision": lambda m: m._stats["Avg_Vision"],
                "Avg_Metabolism": lambda m: m._stats["Avg_Metabolism"],
                "Freq_AlwaysC": lambda m: m._stats["Freq_AlwaysC"],
                "Freq_AlwaysD": lambda m: m._stats["Freq_AlwaysD"],
                "Freq_TFT": lambda m: m._stats["Freq_TFT"],
                "Freq_WSLS": lambda m: m._stats["Freq_WSLS"],
                "Freq_GTFT": lambda m: m._stats["Freq_GTFT"],
            }
        )
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run = 0

    def get_neighborhood_cached(self, pos, radius):
        key = (pos[0], pos[1], radius)
        cached = self._neighborhood_cache.get(key)
        if cached is not None:
            return cached
            
        raw = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=radius)
        seen = set()
        unique = []
        for cell in raw:
            if cell not in seen:
                seen.add(cell)
                unique.append(cell)
        self._neighborhood_cache[key] = unique
        return unique

    def step(self):
        """Синхронный шаг модели."""
        self.env.step()
        active_agents = [a for a in self.agents if a.alive]
        
        for agent in active_agents:
            agent.age += 1
            agent.perceive_and_move()
            
        self._harvest_cells(active_agents)
        self._interact_cells_aggregated(active_agents)
        
        for agent in active_agents:
            agent.metabolize()
            
        self._evolution_step()
        
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run += 1

    def _harvest_cells(self, agents):
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)
            
        env_resource = self.env.resource
        for (x, y), cell_agents in by_cell.items():
            demands = [a.genome.metabolism * 2.0 for a in cell_agents]
            total_demand = sum(demands)
            if total_demand <= 0:
                continue
                
            available = env_resource[y, x]
            if available <= 0:
                continue
                
            if total_demand <= available:
                for agent, demand in zip(cell_agents, demands):
                    agent.resource += demand
                env_resource[y, x] -= total_demand
            else:
                scale = available / total_demand
                for agent, demand in zip(cell_agents, demands):
                    agent.resource += demand * scale
                env_resource[y, x] = 0.0

    def _interact_cells_aggregated(self, agents):
        """Агрегированные взаимодействия с использованием памяти (Пункт 1.2)."""
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)
            
        game = self.cfg.game
        memory_size = getattr(self.cfg, "memory_size", 10)
        
        for cell_agents in by_cell.values():
            n = len(cell_agents)
            if n <= 1:
                # Даже если агент один, обновляем его состояние и историю
                for a in cell_agents:
                    a.last_action = a.get_action([])
                    a.last_payoff = 0.0
                    a.last_cell_coop_rate = 1.0
                    a.interaction_history.append({
                        "step": self.steps_run,
                        "action": a.last_action,
                        "payoff": 0.0,
                        "cell_coop_rate": 1.0
                    })
                continue
                
            # 1. Агенты принимают решения на основе памяти и стратегии
            actions = {}
            for a in cell_agents:
                actions[a.unique_id] = a.get_action(cell_agents)
                
            n_c = sum(1 for a in cell_agents if actions[a.unique_id] == "C")
            n_d = n - n_c
            denom = max(1, n - 1)
            
            c_payoff = ((n_c - 1) * game.R + n_d * game.S) / denom
            d_payoff = (n_c * game.T + (n_d - 1) * game.P) / denom
            
            # 2. Начисление выигрышей и обновление памяти
            for a in cell_agents:
                action = actions[a.unique_id]
                if action == "C":
                    payoff = c_payoff
                else:
                    payoff = d_payoff
                    
                a.resource += payoff
                
                a.last_action = action
                a.last_payoff = payoff
                
                # Расчет доли кооператоров среди СОСЕДЕЙ
                other_c = n_c - (1 if action == "C" else 0)
                other_n = n - 1
                a.last_cell_coop_rate = other_c / other_n if other_n > 0 else 1.0
                
                # Обновление памяти о партнерах (Пункт 1.2)
                for other in cell_agents:
                    if other.unique_id != a.unique_id:
                        a.partners[other.unique_id] = {
                            "last_action": actions[other.unique_id],
                            "last_seen": self.steps_run
                        }
                        
                # Очистка старой памяти (агент забывает тех, с кем давно не пересекался)
                a.partners = {
                    pid: info for pid, info in a.partners.items()
                    if self.steps_run - info["last_seen"] <= memory_size
                }
                
                # Сохранение в историю взаимодействий
                a.interaction_history.append({
                    "step": self.steps_run,
                    "action": action,
                    "payoff": payoff,
                    "cell_coop_rate": a.last_cell_coop_rate
                })

    def _evolution_step(self):
        newborns = []
        for agent in list(self.agents):
            if agent.can_reproduce():
                child = agent.reproduce()
                newborns.append((child, agent.pos))
                
        for child, spawn_pos in newborns:
            self.grid.place_agent(child, spawn_pos)
            
        for agent in list(self.agents):
            if not agent.alive:
                self.grid.remove_agent(agent)
                agent.remove()

    def run_model(self, steps):
        for _ in range(steps):
            if len(self.agents) == 0:
                print(f"Популяция вымерла на шаге {self.steps_run}", flush=True)
                break
            self.step()
