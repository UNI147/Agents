from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np

from .agent import EcoAgent, Genome, IMITATION_TYPES
from .environment import DynamicEnvironment
import networkx as nx


def compute_stats(model):
    """Единый проход по агентам для сбора ВСЕЙ статистики, включая имитацию."""
    n = len(model.agents)
    if n == 0:
        return _empty_stats()

    # Базовые суммы
    sum_res = sum_vis = sum_met = n_c = n_action_c = 0
    # Счётчики стратегий
    strat_counts = {s: 0 for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]}
    # === СЧЁТЧИКИ ПО ТИПАМ ИМИТАЦИИ ===
    imit_type_counts = {t: 0 for t in IMITATION_TYPES}
    imit_type_payoff = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_resource = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_attempts = {t: 0 for t in IMITATION_TYPES}
    imit_type_successes = {t: 0 for t in IMITATION_TYPES}
    # Интенсивность имитации
    sum_intensity = 0.0
    sum_imitation_rate = 0.0

    for a in model.agents:
        sum_res += a.resource
        sum_vis += a.genome.vision
        sum_met += a.genome.metabolism
        sum_intensity += a.genome.imitation_intensity
        sum_imitation_rate += a.genome.imitation_rate

        strat = a.genome.strategy
        if strat in strat_counts:
            strat_counts[strat] += 1
        if strat in ("AlwaysC", "TFT", "GTFT"):
            n_c += 1

        if getattr(a, "last_action", "C") == "C":
            n_action_c += 1

        # Агрегируем по типу имитации
        itype = a.genome.imitation_type
        imit_type_counts[itype] += 1
        imit_type_payoff[itype] += a.last_payoff
        imit_type_resource[itype] += a.resource
        imit_type_attempts[itype] += getattr(a, "imitation_attempts", 0)
        imit_type_successes[itype] += getattr(a, "imitation_successes", 0)

    stats = {
        "Population": n,
        "Freq_Cooperators": n_c / n,
        "Freq_Action_C": n_action_c / n,
        "Avg_Agent_Resource": sum_res / n,
        "Avg_Vision": sum_vis / n,
        "Avg_Metabolism": sum_met / n,
        "Avg_Imitation_Intensity": sum_intensity / n,
        "Avg_Imitation_Rate": sum_imitation_rate / n,
    }
    # Частоты стратегий
    for s, cnt in strat_counts.items():
        stats[f"Freq_{s}"] = cnt / n
    # === ЧАСТОТЫ ТИПОВ ИМИТАЦИИ ===
    for t in IMITATION_TYPES:
        stats[f"ImitFreq_{t}"] = imit_type_counts[t] / n
        stats[f"ImitAvgPayoff_{t}"] = (
            imit_type_payoff[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        )
        stats[f"ImitAvgResource_{t}"] = (
            imit_type_resource[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        )
        attempts = imit_type_attempts[t]
        stats[f"ImitSuccessRate_{t}"] = (
            imit_type_successes[t] / attempts if attempts > 0 else 0.0
        )


    # === СТАТИСТИКА ГРУПП (Пункт 1.6) ===
    if getattr(model.cfg, "group_selection_enabled", False):
        group_res = {}
        for a in model.agents:
            group_res.setdefault(a.group_id, []).append(a.resource)
        stats["Alive_Groups"] = len(group_res)
        if len(group_res) > 1:
            means = [sum(v)/len(v) for v in group_res.values()]
            stats["Group_Fitness_Variance"] = float(np.var(means))
        else:
            stats["Group_Fitness_Variance"] = 0.0
    else:
        stats["Alive_Groups"] = 1
        stats["Group_Fitness_Variance"] = 0.0

    return stats


def _empty_stats():
    stats = {
        "Population": 0, "Freq_Cooperators": 0.0, "Freq_Action_C": 0.0,
        "Avg_Agent_Resource": 0.0, "Avg_Vision": 0.0, "Avg_Metabolism": 0.0,
        "Avg_Imitation_Intensity": 0.0, "Avg_Imitation_Rate": 0.0,
        "Alive_Groups": 1, "Group_Fitness_Variance": 0.0,
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
            width=self.cfg.width, height=self.cfg.height,
            max_resource=self.cfg.max_resource, regen_rate=self.cfg.regen_rate,
            season_period=self.cfg.season_period,
            season_amplitude=self.cfg.season_amplitude,
            catastrophe_prob=self.cfg.catastrophe_prob,
            catastrophe_duration=self.cfg.catastrophe_duration,
            catastrophe_severity=self.cfg.catastrophe_severity,
            rng=self.rng,
        )

        self._neighborhood_cache = {}
        
        # === СОЦИАЛЬНАЯ СЕТЬ ===
        self.max_slots = 0
        self._slot_to_agent = {}
        
        strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
        num_groups = getattr(self.cfg, "num_groups", 10)
        gs_enabled = getattr(self.cfg, "group_selection_enabled", False)
        
        for i in range(self.cfg.initial_agents):
            genome = Genome(
                vision=int(self.rng.integers(self.cfg.min_vision, self.cfg.max_vision + 1)),
                metabolism=float(self.rng.uniform(self.cfg.min_metabolism, self.cfg.max_metabolism)),
                strategy=str(self.rng.choice(strategies)),
                max_age=self.cfg.max_age,
                # === ИНИЦИАЛИЗАЦИЯ ИМИТАЦИОННЫХ ГЕНОВ ===
                imitation_type=str(self.rng.choice(IMITATION_TYPES)),
                imitation_intensity=float(self.rng.uniform(
                    self.cfg.min_imitation_intensity, self.cfg.max_imitation_intensity
                )),
                imitation_rate=float(self.rng.uniform(0.05, self.cfg.initial_imitation_rate + 0.1)),
            )
            x = int(self.rng.integers(0, self.cfg.width))
            y = int(self.rng.integers(0, self.cfg.height))
            
            group_id = i % num_groups if gs_enabled else 0
            agent = EcoAgent(self, genome=genome, group_id=group_id)
            agent.resource = self.cfg.initial_resource
            agent.network_slot = self.max_slots
            self.max_slots += 1
            
            self.grid.place_agent(agent, (x, y))

        # Построение начального графа
        self.social_network = self._build_social_network()
        self._network_neighbors_cache = {}
        self._network_neighbors_cache_step = -1

        # === РЕГИСТРАЦИЯ ВСЕХ РЕПОРТЁРОВ ===
        reporters = {
            "Population": lambda m: m._stats["Population"],
            "Total_Env_Resource": lambda m: m.env.total_resource,
            "Season_Phase": lambda m: m.env.season_phase,
            "Catastrophe_Active": lambda m: 1 if m.env.catastrophe_active else 0,
            "Freq_Cooperators": lambda m: m._stats["Freq_Cooperators"],
            "Freq_Action_C": lambda m: m._stats["Freq_Action_C"],
            "Avg_Agent_Resource": lambda m: m._stats["Avg_Agent_Resource"],
            "Avg_Vision": lambda m: m._stats["Avg_Vision"],
            "Avg_Metabolism": lambda m: m._stats["Avg_Metabolism"],
            "Avg_Imitation_Intensity": lambda m: m._stats["Avg_Imitation_Intensity"],
            "Avg_Imitation_Rate": lambda m: m._stats["Avg_Imitation_Rate"],
            "Alive_Groups": lambda m: m._stats["Alive_Groups"],
            "Group_Fitness_Variance": lambda m: m._stats["Group_Fitness_Variance"],
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

    def _build_social_network(self):
        n = self.max_slots
        net_type = getattr(self.cfg, "network_type", "none")
        if net_type == "none" or n <= 1:
            return None
            
        if net_type == "barabasi_albert":
            m = getattr(self.cfg, "network_param_m", 3)
            m = max(1, min(m, n - 1))
            G = nx.barabasi_albert_graph(n, m, seed=self._seed)
        elif net_type == "watts_strogatz":
            k = getattr(self.cfg, "network_param_k", 4)
            p = getattr(self.cfg, "network_param_p", 0.1)
            k = max(2, min(k, n - 1))
            if k % 2 != 0: k += 1
            G = nx.watts_strogatz_graph(n, k, p, seed=self._seed)
        elif net_type == "random":
            p = getattr(self.cfg, "network_param_p", 0.1)
            G = nx.erdos_renyi_graph(n, p, seed=self._seed)
        else:
            return None
        return G

    def _interact_network(self, agents):
        game = self.cfg.game
        memory_size = getattr(self.cfg, "memory_size", 10)

        self._slot_to_agent = {a.network_slot: a for a in agents}

        mean_mode = getattr(self.cfg, "network_payoff_mode", "mean") == "mean"

        for a in agents:
            a.last_payoff = 0.0
            a.last_action = None
            a._network_neighbors_count = 0
            a._network_coop_count = 0
            a._payoff_sum = 0.0

        # 1. Один раз кешируем живые сетевые связи и определяем действия.
        network_neighbors = {}

        for a in agents:
            slot = a.network_slot
            neigh = [
                self._slot_to_agent[s]
                for s in self.social_network.neighbors(slot)
                if s in self._slot_to_agent
            ]

            network_neighbors[slot] = neigh
            a.last_action = a.get_action(current_partners=neigh)

        # 2. Розыгрыш игр вдоль рёбер.
        for u_slot, v_slot in self.social_network.edges():
            agent_u = self._slot_to_agent.get(u_slot)
            agent_v = self._slot_to_agent.get(v_slot)

            if agent_u is None or agent_v is None:
                continue

            action_u = agent_u.last_action
            action_v = agent_v.last_action

            payoff_u = game.payoff(action_u, action_v)
            payoff_v = game.payoff(action_v, action_u)

            agent_u._payoff_sum += payoff_u
            agent_v._payoff_sum += payoff_v

            agent_u._network_neighbors_count += 1
            agent_v._network_neighbors_count += 1

            if action_v == "C":
                agent_u._network_coop_count += 1

            if action_u == "C":
                agent_v._network_coop_count += 1

        # 3. Финализация payoff, ресурс, память.
        for a in agents:
            cnt = a._network_neighbors_count

            if cnt > 0:
                if mean_mode:
                    a.last_payoff = a._payoff_sum / cnt
                else:
                    a.last_payoff = a._payoff_sum

                a.last_cell_coop_rate = a._network_coop_count / cnt
            else:
                a.last_payoff = 0.0
                a.last_cell_coop_rate = 1.0

            a.resource += a.last_payoff

            for other in network_neighbors.get(a.network_slot, []):
                a.partners[other.unique_id] = {
                    "last_action": other.last_action,
                    "last_seen": self.steps_run
                }

            a.partners = {
                pid: info
                for pid, info in a.partners.items()
                if self.steps_run - info["last_seen"] <= memory_size
            }

            a.interaction_history.append({
                "step": self.steps_run,
                "action": a.last_action,
                "payoff": a.last_payoff,
                "cell_coop_rate": a.last_cell_coop_rate
            })

        # Кеш используется дальше в _imitation_step.
        self._network_neighbors_cache = network_neighbors
        self._network_neighbors_cache_step = self.steps_run

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
        self.env.step()
        active_agents = [a for a in self.agents if a.alive]

        # 1. Движение
        for agent in active_agents:
            agent.age += 1
            agent.perceive_and_move()

        # 2. Сбор ресурсов
        self._harvest_cells(active_agents)

        # 3. Игровые взаимодействия (Сеть или Клетки)
        if getattr(self.cfg, "network_type", "none") != "none" and self.social_network is not None:
            self._interact_network(active_agents)
        else:
            self._interact_cells_aggregated(active_agents)

        # === МНОГОУРОВНЕВЫЙ ОТБОР: ВНУТРИГРУППОВОЙ БОНОУС ===
        if getattr(self.cfg, "group_selection_enabled", False):
            alpha = getattr(self.cfg, "group_selection_intensity", 0.0)
            if alpha > 0:
                group_payoffs = {}
                group_counts = {}
                for a in active_agents:
                    gid = a.group_id
                    group_payoffs[gid] = group_payoffs.get(gid, 0.0) + a.last_payoff
                    group_counts[gid] = group_counts.get(gid, 0) + 1
                    
                group_avg = {gid: group_payoffs[gid] / group_counts[gid] for gid in group_payoffs}
                
                for a in active_agents:
                    avg_g = group_avg.get(a.group_id, a.last_payoff)
                    # Корректировка ресурса: смесь личного и группового успеха
                    bonus = alpha * (avg_g - a.last_payoff)
                    a.resource += bonus

        # 4. Фаза социального обучения (Пункт 1.3)
        self._imitation_step(active_agents)

        # 5. Метаболизм
        for agent in active_agents:
            agent.metabolize()

        # === МНОГОУРОВНЕВЫЙ ОТБОР: КОНКУРЕНЦИЯ ГРУПП ===
        if getattr(self.cfg, "group_selection_enabled", False):
            comp_step = getattr(self.cfg, "group_competition_step", 50)
            if comp_step > 0 and self.steps_run > 0 and self.steps_run % comp_step == 0:
                self._group_competition_step()

        # 6. Эволюция
        self._evolution_step()

        # 7. Сбор статистики
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run += 1

    def _group_competition_step(self):
        """
        Конкуренция групп: группа с худшим средним ресурсом ассимилируется
        группой с лучшим средним ресурсом (смена group_id + копирование стратегии).
        """
        groups = {}
        for a in self.agents:
            if a.alive:
                groups.setdefault(a.group_id, []).append(a)
                
        if len(groups) < 2:
            return
            
        group_fitness = {}
        for gid, members in groups.items():
            group_fitness[gid] = sum(m.resource for m in members) / len(members)
            
        best_gid = max(group_fitness, key=group_fitness.get)
        worst_gid = min(group_fitness, key=group_fitness.get)
        
        if best_gid == worst_gid:
            return
            
        worst_members = groups[worst_gid]
        best_members = groups[best_gid]
        
        for agent in worst_members:
            agent.group_id = best_gid
            # Культурная/генетическая ассимиляция
            if self.rng.random() < 0.5 and best_members:
                role_model = self.rng.choice(best_members)
                agent.genome.strategy = role_model.genome.strategy

    def _imitation_step(self, agents):
        use_network = (
            getattr(self.cfg, "network_type", "none") != "none"
            and self.social_network is not None
        )

        cache_ok = (
            use_network
            and getattr(self, "_network_neighbors_cache_step", -1) == self.steps_run
        )

        for agent in agents:
            if use_network:
                if cache_ok:
                    neighbors = self._network_neighbors_cache.get(agent.network_slot, [])
                else:
                    neighbor_slots = list(self.social_network.neighbors(agent.network_slot))
                    neighbors = [
                        self._slot_to_agent[s]
                        for s in neighbor_slots
                        if s in self._slot_to_agent
                    ]
            else:
                # Fallback на пространственных соседей.
                neighbor_cells = self.get_neighborhood_cached(agent.pos, radius=1)
                neighbors = []

                for cell in neighbor_cells:
                    cell_agents = self.grid.get_cell_list_contents([cell])
                    neighbors.extend([
                        a for a in cell_agents
                        if isinstance(a, EcoAgent)
                    ])

            agent.try_imitate(neighbors)

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
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)
        game = self.cfg.game
        memory_size = getattr(self.cfg, "memory_size", 10)

        for cell_agents in by_cell.values():
            n = len(cell_agents)
            if n <= 1:
                for a in cell_agents:
                    a.last_action = a.get_action([])
                    a.last_payoff = 0.0
                    a.last_cell_coop_rate = 1.0
                    a.interaction_history.append({
                        "step": self.steps_run, "action": a.last_action,
                        "payoff": 0.0, "cell_coop_rate": 1.0
                    })
                continue

            actions = {}
            for a in cell_agents:
                actions[a.unique_id] = a.get_action(cell_agents)

            n_c = sum(1 for a in cell_agents if actions[a.unique_id] == "C")
            n_d = n - n_c
            denom = max(1, n - 1)
            c_payoff = ((n_c - 1) * game.R + n_d * game.S) / denom
            d_payoff = (n_c * game.T + (n_d - 1) * game.P) / denom

            for a in cell_agents:
                action = actions[a.unique_id]
                payoff = c_payoff if action == "C" else d_payoff
                a.resource += payoff
                a.last_action = action
                a.last_payoff = payoff

                other_c = n_c - (1 if action == "C" else 0)
                other_n = n - 1
                a.last_cell_coop_rate = other_c / other_n if other_n > 0 else 1.0

                for other in cell_agents:
                    if other.unique_id != a.unique_id:
                        a.partners[other.unique_id] = {
                            "last_action": actions[other.unique_id],
                            "last_seen": self.steps_run
                        }
                a.partners = {
                    pid: info for pid, info in a.partners.items()
                    if self.steps_run - info["last_seen"] <= memory_size
                }
                a.interaction_history.append({
                    "step": self.steps_run, "action": action,
                    "payoff": payoff, "cell_coop_rate": a.last_cell_coop_rate
                })

    def _evolution_step(self):
        newborns = []

        capacity = getattr(self.cfg, "population_capacity", 10**9)

        # Считаем живых агентов до рождения новых.
        # Мертвые будут удалены ниже, но сейчас они еще могут быть в agents.
        planned_population = sum(1 for a in self.agents if a.alive)

        for agent in list(self.agents):
            if planned_population >= capacity:
                break

            if agent.can_reproduce():
                child = agent.reproduce()
                newborns.append((child, agent.pos, agent))
                planned_population += 1

        cfg = self.cfg

        for child, spawn_pos, parent in newborns:
            child.network_slot = self.max_slots
            self.max_slots += 1
            
            # === НАСЛЕДОВАНИЕ ГРУППЫ И МИГРАЦИЯ ===
            child.group_id = parent.group_id
            if getattr(self.cfg, "group_selection_enabled", False):
                mig_rate = getattr(self.cfg, "group_migration_rate", 0.0)
                if self.rng.random() < mig_rate:
                    child.group_id = int(self.rng.integers(0, getattr(self.cfg, "num_groups", 10)))

            self.grid.place_agent(child, spawn_pos)

            if self.social_network is not None:
                self.social_network.add_node(child.network_slot)

                parent_slot = parent.network_slot

                if parent_slot in self.social_network:
                    candidates = [
                        s for s in self.social_network.neighbors(parent_slot)
                        if s != child.network_slot
                    ]

                    max_deg = getattr(cfg, "max_network_degree", 0)

                    target_edges = getattr(
                        cfg,
                        "target_offspring_edges",
                        max(1, cfg.network_param_m)
                    )

                    k = int(target_edges)

                    if max_deg > 0:
                        # Не подключаемся к узлам, которые уже перегружены.
                        candidates = [
                            s for s in candidates
                            if self.social_network.degree(s) < max_deg
                        ]
                        k = min(k, max_deg)

                    if k > 0 and len(candidates) > k:
                        candidates = list(
                            self.rng.choice(candidates, size=k, replace=False)
                        )

                    for nb in candidates[:k]:
                        self.social_network.add_edge(child.network_slot, nb)

        # Смерть и удаление из графа.
        for agent in list(self.agents):
            if not agent.alive:
                self.grid.remove_agent(agent)

                if self.social_network is not None and agent.network_slot in self.social_network:
                    self.social_network.remove_node(agent.network_slot)

                agent.remove()

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
                edges = 0
                if self.social_network is not None:
                    edges = self.social_network.number_of_edges()

                print(
                    f"[step {self.steps_run}] "
                    f"agents={len(self.agents)}, "
                    f"edges={edges}, "
                    f"step_time={step_time:.3f}s",
                    flush=True
                )

            if max_seconds is not None and max_seconds > 0:
                elapsed = time.time() - start
                if elapsed > max_seconds:
                    print(
                        f"Симуляция остановлена по лимиту времени: "
                        f"{elapsed:.1f}s > {max_seconds:.1f}s",
                        flush=True
                    )
                    break
