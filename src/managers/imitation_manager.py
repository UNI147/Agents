class ImitationManager:
    def __init__(self, model):
        self.model = model

    def imitation_step(self, agents):
        use_network = (self.model.cfg.network_type != "none" and self.model.network_manager.social_network is not None)
        
        for agent in agents:
            if use_network:
                neighbors = self.model.network_manager.get_neighbors(agent)
            else:
                neighbor_cells = self.model.get_neighborhood_cells(agent.pos, radius=1)
                neighbors = []
                for cell in neighbor_cells:
                    cell_agents = self.model.grid.get_cell_list_contents([cell])
                    neighbors.extend([a for a in cell_agents if hasattr(a, 'genome')])
            agent.try_imitate(neighbors)