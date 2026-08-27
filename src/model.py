from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np

from .agent import EcoAgent, Genome, IMITATION_TYPES
from .environment import DynamicEnvironment
import networkx as nx

def compute_stats(model):
    n = len(model.agents)
    if n == 0: return _empty_stats()

    sum_sugar = sum_spice = sum_vis = n_c = n_action_c = 0
    sum_met_s = sum_met_sp = 0
    strat_counts = {s: 0 for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]}
    imit_type_counts = {t: 0 for t in IMITATION_TYPES}
    imit_type_payoff = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_resource = {t: 0.0 for t in IMITATION_TYPES}
    imit_type_attempts = {t: 0 for t in IMITATION_TYPES}
    imit_type_successes = {t: 0 for t in IMITATION_TYPES}
    sum_intensity = 0.0
    sum_imitation_rate = 0.0

    for a in model.agents:
        sum_sugar += a.sugar
        sum_spice += a.spice
        sum_vis += a.genome.vision
        sum_met_s += a.genome.metabolism_sugar
        sum_met_sp += a.genome.metabolism_spice
        sum_intensity += a.genome.imitation_intensity
        sum_imitation_rate += a.genome.imitation_rate

        strat = a.genome.strategy
        if strat in strat_counts: strat_counts[strat] += 1
        if strat in ("AlwaysC", "TFT", "GTFT"): n_c += 1
        if getattr(a, "last_action", "C") == "C": n_action_c += 1

        itype = a.genome.imitation_type
        imit_type_counts[itype] += 1
        imit_type_payoff[itype] += a.last_payoff
        imit_type_resource[itype] += (a.sugar + a.spice)
        imit_type_attempts[itype] += getattr(a, "imitation_attempts", 0)
        imit_type_successes[itype] += getattr(a, "imitation_successes", 0)

    stats = {
        "Population": n,
        "Freq_Cooperators": n_c / n,
        "Freq_Action_C": n_action_c / n,
        "Avg_Sugar": sum_sugar / n,
        "Avg_Spice": sum_spice / n,
        "Avg_Vision": sum_vis / n,
        "Avg_Metabolism_Sugar": sum_met_s / n,
        "Avg_Metabolism_Spice": sum_met_sp / n,
        "Avg_Imitation_Intensity": sum_intensity / n,
        "Avg_Imitation_Rate": sum_imitation_rate / n,
        "Total_Pollution": model.env.total_pollution if getattr(model.cfg, "pollution_enabled", False) else 0.0,
    }
    for s, cnt in strat_counts.items(): stats[f"Freq_{s}"] = cnt / n
    for t in IMITATION_TYPES:
        stats[f"ImitFreq_{t}"] = imit_type_counts[t] / n
        stats[f"ImitAvgPayoff_{t}"] = imit_type_payoff[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        stats[f"ImitAvgResource_{t}"] = imit_type_resource[t] / imit_type_counts[t] if imit_type_counts[t] > 0 else 0.0
        attempts = imit_type_attempts[t]
        stats[f"ImitSuccessRate_{t}"] = imit_type_successes[t] / attempts if attempts > 0 else 0.0

    if getattr(model.cfg, "group_selection_enabled", False):
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

    return stats

def _empty_stats():
    stats = {
        "Population": 0, "Freq_Cooperators": 0.0, "Freq_Action_C": 0.0,
        "Avg_Sugar": 0.0, "Avg_Spice": 0.0, "Avg_Vision": 0.0, 
        "Avg_Metabolism_Sugar": 0.0, "Avg_Metabolism_Spice": 0.0,
        "Avg_Imitation_Intensity": 0.0, "Avg_Imitation_Rate": 0.0,
        "Alive_Groups": 1, "Group_Fitness_Variance": 0.0,
        "Total_Pollution": 0.0,
    }
    for s in ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]: stats[f"Freq_{s}"] = 0.0
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

        self._seed = seed if seed is not None else getattr(self.cfg, "seed", 42)
        self.rng = np.random.default_rng(self._seed)

        try: super().__init__(rng=self._seed)
        except TypeError: super().__init__(seed=self._seed)

        self.grid = MultiGrid(self.cfg.width, self.cfg.height, torus=True)
        self.env = DynamicEnvironment(
            width=self.cfg.width, height=self.cfg.height,
            max_resource=self.cfg.max_resource, regen_rate=self.cfg.regen_rate,
            max_spice=self.cfg.max_spice, regen_rate_spice=self.cfg.regen_rate_spice,
            season_period=self.cfg.season_period,
            season_amplitude=self.cfg.season_amplitude,
            catastrophe_prob=self.cfg.catastrophe_prob,
            catastrophe_duration=self.cfg.catastrophe_duration,
            catastrophe_severity=self.cfg.catastrophe_severity,
            rng=self.rng,
            # Pollution params (Пункт 2.2)
            pollution_enabled=getattr(self.cfg, "pollution_enabled", True),
            pollution_diffusion_rate=getattr(self.cfg, "pollution_diffusion_rate", 0.2),
            pollution_decay_rate=getattr(self.cfg, "pollution_decay_rate", 0.05),
            pollution_capacity_impact=getattr(self.cfg, "pollution_capacity_impact", 1.5),
        )

        self._neighborhood_cache = {}
        self.max_slots = 0
        self._slot_to_agent = {}
        
        strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
        num_groups = getattr(self.cfg, "num_groups", 10)
        gs_enabled = getattr(self.cfg, "group_selection_enabled", False)
        
        for i in range(self.cfg.initial_agents):
            genome = Genome(
                vision=int(self.rng.integers(self.cfg.min_vision, self.cfg.max_vision + 1)),
                metabolism_sugar=float(self.rng.uniform(self.cfg.min_metabolism, self.cfg.max_metabolism)),
                metabolism_spice=float(self.rng.uniform(self.cfg.min_metabolism_spice, self.cfg.max_metabolism_spice)),
                strategy=str(self.rng.choice(strategies)),
                max_age=self.cfg.max_age,
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

        self.social_network = self._build_social_network()
        self._network_neighbors_cache = {}
        self._network_neighbors_cache_step = -1

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
        if net_type == "none" or n <= 1: return None
        if net_type == "barabasi_albert":
            m = max(1, min(getattr(self.cfg, "network_param_m", 3), n - 1))
            G = nx.barabasi_albert_graph(n, m, seed=self._seed)
        elif net_type == "watts_strogatz":
            k = max(2, min(getattr(self.cfg, "network_param_k", 4), n - 1))
            if k % 2 != 0: k += 1
            G = nx.watts_strogatz_graph(n, k, getattr(self.cfg, "network_param_p", 0.1), seed=self._seed)
        elif net_type == "random":
            G = nx.erdos_renyi_graph(n, getattr(self.cfg, "network_param_p", 0.1), seed=self._seed)
        else: return None
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

        network_neighbors = {}
        for a in agents:
            slot = a.network_slot
            neigh = [self._slot_to_agent[s] for s in self.social_network.neighbors(slot)
                     if s in self._slot_to_agent]
            network_neighbors[slot] = neigh
            a.last_action = a.get_action(current_partners=neigh)

        for u_slot, v_slot in self.social_network.edges():
            agent_u = self._slot_to_agent.get(u_slot)
            agent_v = self._slot_to_agent.get(v_slot)
            if agent_u is None or agent_v is None:
                continue
            action_u, action_v = agent_u.last_action, agent_v.last_action
            agent_u._payoff_sum += game.payoff(action_u, action_v)
            agent_v._payoff_sum += game.payoff(action_v, action_u)
            agent_u._network_neighbors_count += 1
            agent_v._network_neighbors_count += 1
            if action_v == "C":
                agent_u._network_coop_count += 1
            if action_u == "C":
                agent_v._network_coop_count += 1

        for a in agents:
            cnt = a._network_neighbors_count
            if cnt > 0:
                a.last_payoff = a._payoff_sum / cnt if mean_mode else a._payoff_sum
                a.last_cell_coop_rate = a._network_coop_count / cnt
            else:
                a.last_payoff = 0.0
                a.last_cell_coop_rate = 1.0

            m_s = a.genome.metabolism_sugar
            m_sp = a.genome.metabolism_spice
            m_total = m_s + m_sp
            if m_total > 0:
                frac_sugar = m_s / m_total
                frac_spice = m_sp / m_total
            else:
                frac_sugar = 0.5
                frac_spice = 0.5
            a.sugar += a.last_payoff * frac_sugar
            a.spice += a.last_payoff * frac_spice

            for other in network_neighbors.get(a.network_slot, []):
                a.partners[other.unique_id] = {
                    "last_action": other.last_action, "last_seen": self.steps_run}
            a.partners = {pid: info for pid, info in a.partners.items()
                          if self.steps_run - info["last_seen"] <= memory_size}
            a.interaction_history.append({
                "step": self.steps_run, "action": a.last_action,
                "payoff": a.last_payoff, "cell_coop_rate": a.last_cell_coop_rate})

        self._network_neighbors_cache = network_neighbors
        self._network_neighbors_cache_step = self.steps_run

    def get_neighborhood_cached(self, pos, radius):
        key = (pos[0], pos[1], radius)
        cached = self._neighborhood_cache.get(key)
        if cached is not None: return cached
        raw = self.grid.get_neighborhood(pos, moore=True, include_center=True, radius=radius)
        seen = set(); unique = []
        for cell in raw:
            if cell not in seen: seen.add(cell); unique.append(cell)
        self._neighborhood_cache[key] = unique
        return unique

    def step(self):
        self.env.step()
        active_agents = [a for a in self.agents if a.alive]

        for agent in active_agents:
            agent.age += 1
            agent.perceive_and_move()

        pollution_enabled = getattr(self.cfg, "pollution_enabled", True)
        prod_rate = getattr(self.cfg, "pollution_production_rate", 0.15)
        cons_rate = getattr(self.cfg, "pollution_consumption_rate", 0.25)
        
        # Сетка для накопления отходов за текущий шаг
        pollution_grid = np.zeros((self.cfg.height, self.cfg.width))

        self._harvest_cells(active_agents, pollution_grid, prod_rate)

        if getattr(self.cfg, "network_type", "none") != "none" and self.social_network is not None:
            self._interact_network(active_agents)
        else:
            self._interact_cells_aggregated(active_agents)

        if getattr(self.cfg, "group_selection_enabled", False):
            alpha = getattr(self.cfg, "group_selection_intensity", 0.0)
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

        if getattr(self.cfg, "trade_enabled", False):
            self._trade_step(active_agents)

        for agent in active_agents: 
            met_s, met_sp = agent.metabolize()
            if pollution_enabled:
                x, y = agent.pos
                pollution_grid[y, x] += (met_s + met_sp) * cons_rate
                
        if pollution_enabled:
            self.env.add_pollution(pollution_grid)

        if getattr(self.cfg, "group_selection_enabled", False):
            comp_step = getattr(self.cfg, "group_competition_step", 50)
            if comp_step > 0 and self.steps_run > 0 and self.steps_run % comp_step == 0:
                self._group_competition_step()

        self._evolution_step()
        self._stats = compute_stats(self)
        self.datacollector.collect(self)
        self.steps_run += 1

    def _trade_step(self, agents):
        use_network = (getattr(self.cfg, "network_type", "none") != "none"
                       and self.social_network is not None)
        traded_pairs = set()
        self.rng.shuffle(agents)

        max_trades_per_agent = getattr(self.cfg, "max_trades_per_step", 2)

        for agent in agents:
            if not agent.alive:
                continue
            trades_done = 0
            for _ in range(max_trades_per_agent):
                if trades_done >= max_trades_per_agent:
                    break
                if use_network:
                    neighbor_slots = list(self.social_network.neighbors(agent.network_slot))
                    neighbors = [self._slot_to_agent[s] for s in neighbor_slots
                                 if s in self._slot_to_agent and self._slot_to_agent[s].alive]
                else:
                    neighbor_cells = self.get_neighborhood_cached(agent.pos, radius=1)
                    neighbors = []
                    for cell in neighbor_cells:
                        cell_agents = self.grid.get_cell_list_contents([cell])
                        neighbors.extend([a for a in cell_agents
                                          if isinstance(a, EcoAgent) and a.alive and a != agent])
                if not neighbors:
                    break
                other = self.rng.choice(neighbors)
                pair_id = tuple(sorted((agent.unique_id, other.unique_id)))
                if pair_id in traded_pairs:
                    continue
                agent.trade(other)
                traded_pairs.add(pair_id)
                trades_done += 1

    def _group_competition_step(self):
        groups = {}
        for a in self.agents:
            if a.alive: groups.setdefault(a.group_id, []).append(a)
        if len(groups) < 2: return
        group_fitness = {gid: sum(m.sugar + m.spice for m in members) / len(members) for gid, members in groups.items()}
        best_gid = max(group_fitness, key=group_fitness.get)
        worst_gid = min(group_fitness, key=group_fitness.get)
        if best_gid == worst_gid: return
        for agent in groups[worst_gid]:
            agent.group_id = best_gid
            if self.rng.random() < 0.5 and groups[best_gid]:
                agent.genome.strategy = self.rng.choice(groups[best_gid]).genome.strategy

    def _imitation_step(self, agents):
        use_network = (getattr(self.cfg, "network_type", "none") != "none" and self.social_network is not None)
        cache_ok = (use_network and getattr(self, "_network_neighbors_cache_step", -1) == self.steps_run)
        for agent in agents:
            if use_network:
                if cache_ok: neighbors = self._network_neighbors_cache.get(agent.network_slot, [])
                else:
                    neighbor_slots = list(self.social_network.neighbors(agent.network_slot))
                    neighbors = [self._slot_to_agent[s] for s in neighbor_slots if s in self._slot_to_agent]
            else:
                neighbor_cells = self.get_neighborhood_cached(agent.pos, radius=1)
                neighbors = []
                for cell in neighbor_cells:
                    cell_agents = self.grid.get_cell_list_contents([cell])
                    neighbors.extend([a for a in cell_agents if isinstance(a, EcoAgent)])
            agent.try_imitate(neighbors)

    def _harvest_cells(self, agents, pollution_grid, prod_rate):
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)
        env_sugar = self.env.sugar
        env_spice = self.env.spice

        harvest_multiplier = getattr(self.cfg, "harvest_multiplier", 3.0)

        for (x, y), cell_agents in by_cell.items():
            demands_sugar = [a.genome.metabolism_sugar * harvest_multiplier
                             for a in cell_agents]
            demands_spice = [a.genome.metabolism_spice * harvest_multiplier
                             for a in cell_agents]
            total_demand_sugar = sum(demands_sugar)
            total_demand_spice = sum(demands_spice)
            avail_sugar = env_sugar[y, x]
            avail_spice = env_spice[y, x]

            if total_demand_sugar > 0 and avail_sugar > 0:
                if total_demand_sugar <= avail_sugar:
                    harvested_s = total_demand_sugar
                    for agent, demand in zip(cell_agents, demands_sugar):
                        agent.sugar += demand
                    env_sugar[y, x] -= total_demand_sugar
                else:
                    harvested_s = avail_sugar
                    scale = avail_sugar / total_demand_sugar
                    for agent, demand in zip(cell_agents, demands_sugar):
                        agent.sugar += demand * scale
                    env_sugar[y, x] = 0.0
                pollution_grid[y, x] += harvested_s * prod_rate

            if total_demand_spice > 0 and avail_spice > 0:
                if total_demand_spice <= avail_spice:
                    harvested_sp = total_demand_spice
                    for agent, demand in zip(cell_agents, demands_spice):
                        agent.spice += demand
                    env_spice[y, x] -= total_demand_spice
                else:
                    harvested_sp = avail_spice
                    scale = avail_spice / total_demand_spice
                    for agent, demand in zip(cell_agents, demands_spice):
                        agent.spice += demand * scale
                    env_spice[y, x] = 0.0
                pollution_grid[y, x] += harvested_sp * prod_rate

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
                        "payoff": 0.0, "cell_coop_rate": 1.0})
                continue
            actions = {a.unique_id: a.get_action(cell_agents) for a in cell_agents}
            n_c = sum(1 for a in cell_agents if actions[a.unique_id] == "C")
            n_d = n - n_c
            denom = max(1, n - 1)
            c_payoff = ((n_c - 1) * game.R + n_d * game.S) / denom
            d_payoff = (n_c * game.T + (n_d - 1) * game.P) / denom

            for a in cell_agents:
                action = actions[a.unique_id]
                payoff = c_payoff if action == "C" else d_payoff

                m_s = a.genome.metabolism_sugar
                m_sp = a.genome.metabolism_spice
                m_total = m_s + m_sp
                if m_total > 0:
                    frac_sugar = m_s / m_total
                    frac_spice = m_sp / m_total
                else:
                    frac_sugar = 0.5
                    frac_spice = 0.5
                a.sugar += payoff * frac_sugar
                a.spice += payoff * frac_spice

                a.last_action = action
                a.last_payoff = payoff
                other_c = n_c - (1 if action == "C" else 0)
                a.last_cell_coop_rate = other_c / (n - 1) if n > 1 else 1.0
                for other in cell_agents:
                    if other.unique_id != a.unique_id:
                        a.partners[other.unique_id] = {
                            "last_action": actions[other.unique_id],
                            "last_seen": self.steps_run}
                a.partners = {pid: info for pid, info in a.partners.items()
                              if self.steps_run - info["last_seen"] <= memory_size}
                a.interaction_history.append({
                    "step": self.steps_run, "action": action,
                    "payoff": payoff, "cell_coop_rate": a.last_cell_coop_rate})

    def _evolution_step(self):
        newborns = []
        capacity = getattr(self.cfg, "population_capacity", 10**9)
        planned_population = sum(1 for a in self.agents if a.alive)

        for agent in list(self.agents):
            if planned_population >= capacity: break
            if agent.can_reproduce():
                child = agent.reproduce()
                newborns.append((child, agent.pos, agent))
                planned_population += 1

        cfg = self.cfg
        for child, spawn_pos, parent in newborns:
            child.network_slot = self.max_slots
            self.max_slots += 1
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
                    candidates = [s for s in self.social_network.neighbors(parent_slot) if s != child.network_slot]
                    max_deg = getattr(cfg, "max_network_degree", 0)
                    target_edges = getattr(cfg, "target_offspring_edges", max(1, cfg.network_param_m))
                    k = int(target_edges)
                    if max_deg > 0:
                        candidates = [s for s in candidates if self.social_network.degree(s) < max_deg]
                        k = min(k, max_deg)
                    if k > 0 and len(candidates) > k:
                        candidates = list(self.rng.choice(candidates, size=k, replace=False))
                    for nb in candidates[:k]:
                        self.social_network.add_edge(child.network_slot, nb)

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
                edges = self.social_network.number_of_edges() if self.social_network is not None else 0
                print(f"[step {self.steps_run}] agents={len(self.agents)}, edges={edges}, step_time={step_time:.3f}s", flush=True)
            if max_seconds is not None and max_seconds > 0:
                elapsed = time.time() - start
                if elapsed > max_seconds:
                    print(f"Симуляция остановлена по лимиту времени: {elapsed:.1f}s > {max_seconds:.1f}s", flush=True)
                    break
