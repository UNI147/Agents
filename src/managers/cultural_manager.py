class CulturalManager:
    def __init__(self, model):
        self.model = model

    def cultural_step(self, agents):
        """
        Пункт 4.4: Культурная передача (Tag-flipping).
        Реализация правила Эпштейна (Rule K): агент копирует тег соседа.
        """
        if not getattr(self.model.cfg, 'tag_flipping_enabled', False):
            return
            
        use_network = (self.model.cfg.network_type != "none" and self.model.network_manager.social_network is not None)
        
        for agent in agents:
            # 1. Определение соседей (через социальную сеть или пространственно)
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
            
            if not neighbors:
                continue
                
            # 2. Выбор случайного соседа и случайной позиции тега
            neighbor = self.model.rng.choice(neighbors)
            tag_length = getattr(self.model.cfg, 'tag_length', 11)
            idx = int(self.model.rng.integers(0, tag_length))
            
            # 3. Tag-flipping: если теги не совпадают, агент копирует тег соседа
            if agent.cultural_tags[idx] != neighbor.cultural_tags[idx]:
                agent.cultural_tags[idx] = neighbor.cultural_tags[idx]