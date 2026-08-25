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
            "Population": 0, "Freq_Cooperators": 0.0,
            "Avg_Agent_Resource": 0.0, "Avg_Vision": 0.0, "Avg_Metabolism": 0.0
        }
    
    sum_res = sum_vis = sum_met = n_c = 0
    for a in model.agents:
        sum_res += a.resource
        sum_vis += a.genome.vision
        sum_met += a.genome.metabolism
        if a.genome.strategy == "C":
            n_c += 1
            
    return {
        "Population": n,
        "Freq_Cooperators": n_c / n,
        "Avg_Agent_Resource": sum_res / n,
        "Avg_Vision": sum_vis / n,
        "Avg_Metabolism": sum_met / n,
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

        # Кэш соседних клеток
        self._neighborhood_cache = {}

        # Инициализация агентов
        for _ in range(self.cfg.initial_agents):
            genome = Genome(
                vision=int(self.rng.integers(self.cfg.min_vision, self.cfg.max_vision + 1)),
                metabolism=float(self.rng.uniform(self.cfg.min_metabolism, self.cfg.max_metabolism)),
                strategy=str(self.rng.choice(["C", "D"])),
                max_age=self.cfg.max_age,
            )
            x = int(self.rng.integers(0, self.cfg.width))
            y = int(self.rng.integers(0, self.cfg.height))
            
            agent = EcoAgent(self, genome=genome)
            agent.resource = self.cfg.initial_resource  # Стартовый ресурс только при инициализации
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: m._stats["Population"],
                "Total_Env_Resource": lambda m: m.env.total_resource,
                "Season_Phase": lambda m: m.env.season_phase,
                "Catastrophe_Active": lambda m: 1 if m.env.catastrophe_active else 0,
                "Freq_Cooperators": lambda m: m._stats["Freq_Cooperators"],
                "Avg_Agent_Resource": lambda m: m._stats["Avg_Agent_Resource"],
                "Avg_Vision": lambda m: m._stats["Avg_Vision"],
                "Avg_Metabolism": lambda m: m._stats["Avg_Metabolism"],
            }
        )
        
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run = 0

    def get_neighborhood_cached(self, pos, radius):
        """Кэшированное получение соседей с удалением дубликатов."""
        key = (pos[0], pos[1], radius)
        cached = self._neighborhood_cache.get(key)
        if cached is not None:
            return cached

        raw = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=radius)
        
        # Удаляем дубликаты, сохраняя порядок
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
        # 1. Обновление среды
        self.env.step()

        # Получаем список активных агентов (живых на начало шага)
        active_agents = [a for a in self.agents if a.alive]

        # 2. Фаза движения (все видят среду ДО сбора ресурса)
        for agent in active_agents:
            agent.age += 1
            agent.perceive_and_move()

        # 3. Фаза сбора ресурса (одновременная конкуренция)
        self._harvest_cells(active_agents)

        # 4. Фаза взаимодействий (агрегированная, средний выигрыш)
        self._interact_cells_aggregated(active_agents)

        # 5. Фаза метаболизма
        for agent in active_agents:
            agent.metabolize()

        # 6. Эволюция (размножение и смерть)
        self._evolution_step()

        # 7. Сбор данных
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run += 1

    def _harvest_cells(self, agents):
        """Одновременный сбор ресурса с пропорциональным дележом."""
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
                # Ресурса хватает всем
                for agent, demand in zip(cell_agents, demands):
                    agent.resource += demand
                env_resource[y, x] -= total_demand
            else:
                # Ресурса не хватает — делим пропорционально
                scale = available / total_demand
                for agent, demand in zip(cell_agents, demands):
                    agent.resource += demand * scale
                env_resource[y, x] = 0.0

    def _interact_cells_aggregated(self, agents):
        """Агрегированные взаимодействия со средним выигрышем."""
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)

        game = self.cfg.game

        for cell_agents in by_cell.values():
            n = len(cell_agents)
            if n <= 1:
                continue

            n_c = sum(1 for a in cell_agents if a.genome.strategy == "C")
            n_d = n - n_c
            denom = max(1, n - 1)

            # Средний выигрыш за контакт (нормировка)
            c_payoff = ((n_c - 1) * game.R + n_d * game.S) / denom
            d_payoff = (n_c * game.T + (n_d - 1) * game.P) / denom

            for a in cell_agents:
                if a.genome.strategy == "C":
                    a.resource += c_payoff
                else:
                    a.resource += d_payoff

    def _evolution_step(self):
        # Размножение
        newborns = []
        for agent in list(self.agents):
            if agent.can_reproduce():
                child = agent.reproduce()
                newborns.append((child, agent.pos))

        for child, spawn_pos in newborns:
            self.grid.place_agent(child, spawn_pos)

        # Удаление мёртвых
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
