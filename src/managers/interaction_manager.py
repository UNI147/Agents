import numpy as np

class InteractionManager:
    def __init__(self, model):
        self.model = model

    def harvest_cells(self, agents, pollution_grid):
        by_cell = {}
        for agent in agents:
            by_cell.setdefault(agent.pos, []).append(agent)
        env_sugar = self.model.env.sugar
        env_spice = self.model.env.spice

        harvest_multiplier = getattr(self.model.cfg, 'harvest_multiplier', 1.0)
        prod_rate = self.model.cfg.pollution_production_rate

        for (x, y), cell_agents in by_cell.items():
            demands_sugar = [a.genome.metabolism_sugar * harvest_multiplier for a in cell_agents]
            demands_spice = [a.genome.metabolism_spice * harvest_multiplier for a in cell_agents]
            total_demand_sugar = sum(demands_sugar)
            total_demand_spice = sum(demands_spice)
            avail_sugar = env_sugar[y, x]
            avail_spice = env_spice[y, x]

            if total_demand_sugar > 0 and avail_sugar > 0:
                if total_demand_sugar <= avail_sugar:
                    harvested_s = total_demand_sugar
                    for agent, demand in zip(cell_agents, demands_sugar): agent.sugar += demand
                    env_sugar[y, x] -= total_demand_sugar
                else:
                    harvested_s = avail_sugar
                    scale = avail_sugar / total_demand_sugar
                    for agent, demand in zip(cell_agents, demands_sugar): agent.sugar += demand * scale
                    env_sugar[y, x] = 0.0
                if self.model.cfg.pollution_enabled: pollution_grid[y, x] += harvested_s * prod_rate

            if total_demand_spice > 0 and avail_spice > 0:
                if total_demand_spice <= avail_spice:
                    harvested_sp = total_demand_spice
                    for agent, demand in zip(cell_agents, demands_spice): agent.spice += demand
                    env_spice[y, x] -= total_demand_spice
                else:
                    harvested_sp = avail_spice
                    scale = avail_spice / total_demand_spice
                    for agent, demand in zip(cell_agents, demands_spice): agent.spice += demand * scale
                    env_spice[y, x] = 0.0
                if self.model.cfg.pollution_enabled: pollution_grid[y, x] += harvested_sp * prod_rate

    def interact_cells_aggregated(self, agents):
        by_cell = {}
        for agent in agents: by_cell.setdefault(agent.pos, []).append(agent)
        game = self.model.cfg.game
        memory_size = self.model.cfg.memory_size

        for cell_agents in by_cell.values():
            n = len(cell_agents)
            if n <= 1:
                for a in cell_agents:
                    a.last_action = a.get_action([])
                    a.last_payoff = 0.0
                    a.last_cell_coop_rate = 1.0
                    a.interaction_history.append({"step": self.model.steps_run, "action": a.last_action, "payoff": 0.0, "cell_coop_rate": 1.0})
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
                m_s, m_sp = a.genome.metabolism_sugar, a.genome.metabolism_spice
                m_total = m_s + m_sp
                frac_sugar = m_s / m_total if m_total > 0 else 0.5
                frac_spice = m_sp / m_total if m_total > 0 else 0.5
                
                a.sugar += payoff * frac_sugar
                a.spice += payoff * frac_spice
                a.last_action = action
                a.last_payoff = payoff
                other_c = n_c - (1 if action == "C" else 0)
                a.last_cell_coop_rate = other_c / (n - 1) if n > 1 else 1.0
                for other in cell_agents:
                    if other.unique_id != a.unique_id:
                        a.partners[other.unique_id] = {"last_action": actions[other.unique_id], "last_seen": self.model.steps_run}
                a.partners = {pid: info for pid, info in a.partners.items() if self.model.steps_run - info["last_seen"] <= memory_size}
                a.interaction_history.append({"step": self.model.steps_run, "action": action, "payoff": payoff, "cell_coop_rate": a.last_cell_coop_rate})

    def group_competition_step(self):
        groups = {}
        for a in self.model.agents:
            if a.alive: groups.setdefault(a.group_id, []).append(a)
        if len(groups) < 2: return
        
        group_fitness = {gid: sum(m.sugar + m.spice for m in members) / len(members) for gid, members in groups.items()}
        best_gid = max(group_fitness, key=group_fitness.get)
        max_fitness = group_fitness[best_gid]
        
        for gid, members in groups.items():
            if gid == best_gid: continue
            fitness_gap = max_fitness - group_fitness[gid]
            migration_prob = min(0.8, fitness_gap / (max_fitness + 1e-6))
            
            for agent in members:
                if self.model.rng.random() < migration_prob:
                    agent.group_id = best_gid
                    if self.model.rng.random() < 0.5 and groups[best_gid]:
                        agent.genome.strategy = self.model.rng.choice(groups[best_gid]).genome.strategy