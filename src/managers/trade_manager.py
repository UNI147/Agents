class TradeManager:
    def __init__(self, model):
        self.model = model

    def trade_step(self, agents):
        if not self.model.cfg.trade_enabled: return

        use_network = (self.model.cfg.network_type != "none" and self.model.network_manager.social_network is not None)
        traded_pairs = set()
        self.model.rng.shuffle(agents)

        max_trades_per_agent = getattr(self.model.cfg, 'max_trades_per_step', 1)

        for agent in agents:
            if not agent.alive: continue
            trades_done = 0
            for _ in range(max_trades_per_agent):
                if trades_done >= max_trades_per_agent: break
                if use_network:
                    neighbor_slots = list(self.model.network_manager.social_network.neighbors(agent.network_slot))
                    neighbors = [self.model.network_manager._slot_to_agent[s] for s in neighbor_slots
                                 if s in self.model.network_manager._slot_to_agent and self.model.network_manager._slot_to_agent[s].alive]
                else:
                    neighbor_cells = self.model.get_neighborhood_cells(agent.pos, radius=1)
                    neighbors = []
                    for cell in neighbor_cells:
                        cell_agents = self.model.grid.get_cell_list_contents([cell])
                        neighbors.extend([a for a in cell_agents if hasattr(a, 'alive') and a.alive and a != agent])
                if not neighbors: break
                other = self.model.rng.choice(neighbors)
                pair_id = tuple(sorted((agent.unique_id, other.unique_id)))
                if pair_id in traded_pairs: continue
                agent.trade(other)
                traded_pairs.add(pair_id)
                trades_done += 1