class EvolutionManager:
    def __init__(self, model):
        self.model = model

    def evolution_step(self):
        newborns = []
        capacity = getattr(self.model.cfg, 'population_capacity', getattr(self.model.cfg, 'max_population', 3000))
        planned_population = sum(1 for a in self.model.agents if a.alive)

        for agent in list(self.model.agents):
            if planned_population >= capacity: break
            if agent.can_reproduce():
                child = agent.reproduce()
                newborns.append((child, agent.pos, agent))
                planned_population += 1

        cfg = self.model.cfg
        for child, spawn_pos, parent in newborns:
            child.network_slot = self.model.max_slots
            self.model.max_slots += 1
            child.group_id = parent.group_id
            if cfg.group_selection_enabled:
                mig_rate = getattr(cfg, 'group_migration_rate', 0.0)
                if self.model.rng.random() < mig_rate:
                    child.group_id = int(self.model.rng.integers(0, cfg.num_groups))

            self.model.grid.place_agent(child, spawn_pos)
            self.model.network_manager.add_agent_to_network(child, parent)

        for agent in list(self.model.agents):
            if not agent.alive:
                self.model.grid.remove_agent(agent)
                self.model.network_manager.remove_agent_from_network(agent)
                agent.remove()