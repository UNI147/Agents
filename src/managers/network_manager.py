import networkx as nx

class NetworkManager:
    def __init__(self, model):
        self.model = model
        self.social_network = None
        self._slot_to_agent = {}
        self._network_neighbors_cache = {}
        self._network_neighbors_cache_step = -1

    def build_network(self, num_agents):
        cfg = self.model.cfg
        net_type = cfg.network_type
        if net_type == "none" or num_agents <= 1:
            self.social_network = None
            return

        if net_type == "barabasi_albert":
            m = max(1, min(cfg.network_param_m, num_agents - 1))
            self.social_network = nx.barabasi_albert_graph(num_agents, m, seed=self.model._seed)
        elif net_type == "watts_strogatz":
            k = max(2, min(cfg.network_param_k, num_agents - 1))
            if k % 2 != 0: k += 1
            self.social_network = nx.watts_strogatz_graph(num_agents, k, cfg.network_param_p, seed=self.model._seed)
        elif net_type == "random":
            self.social_network = nx.erdos_renyi_graph(num_agents, cfg.network_param_p, seed=self.model._seed)
        else:
            self.social_network = None

    def interact_network(self, agents):
        if self.social_network is None: return
            
        game = self.model.cfg.game
        memory_size = self.model.cfg.memory_size
        self._slot_to_agent = {a.network_slot: a for a in agents}
        mean_mode = self.model.cfg.network_payoff_mode == "mean"

        for a in agents:
            a.last_payoff = 0.0
            a.last_action = None
            a._network_neighbors_count = 0
            a._network_coop_count = 0
            a._payoff_sum = 0.0

        network_neighbors = {}
        for a in agents:
            slot = a.network_slot
            neigh = [self._slot_to_agent[s] for s in self.social_network.neighbors(slot) if s in self._slot_to_agent]
            network_neighbors[slot] = neigh
            a.last_action = a.get_action(current_partners=neigh)

        for u_slot, v_slot in self.social_network.edges():
            agent_u = self._slot_to_agent.get(u_slot)
            agent_v = self._slot_to_agent.get(v_slot)
            if agent_u is None or agent_v is None: continue
            action_u, action_v = agent_u.last_action, agent_v.last_action
            agent_u._payoff_sum += game.payoff(action_u, action_v)
            agent_v._payoff_sum += game.payoff(action_v, action_u)
            agent_u._network_neighbors_count += 1
            agent_v._network_neighbors_count += 1
            if action_v == "C": agent_u._network_coop_count += 1
            if action_u == "C": agent_v._network_coop_count += 1

        for a in agents:
            cnt = a._network_neighbors_count
            if cnt > 0:
                a.last_payoff = a._payoff_sum / cnt if mean_mode else a._payoff_sum
                a.last_cell_coop_rate = a._network_coop_count / cnt
            else:
                a.last_payoff = 0.0
                a.last_cell_coop_rate = 1.0

            m_s, m_sp = a.genome.metabolism_sugar, a.genome.metabolism_spice
            m_total = m_s + m_sp
            frac_sugar = m_s / m_total if m_total > 0 else 0.5
            frac_spice = m_sp / m_total if m_total > 0 else 0.5
            
            a.sugar += a.last_payoff * frac_sugar
            a.spice += a.last_payoff * frac_spice

            for other in network_neighbors.get(a.network_slot, []):
                a.partners[other.unique_id] = {"last_action": other.last_action, "last_seen": self.model.steps_run}
            a.partners = {pid: info for pid, info in a.partners.items() if self.model.steps_run - info["last_seen"] <= memory_size}
            a.interaction_history.append({"step": self.model.steps_run, "action": a.last_action, "payoff": a.last_payoff, "cell_coop_rate": a.last_cell_coop_rate})

        self._network_neighbors_cache = network_neighbors
        self._network_neighbors_cache_step = self.model.steps_run

    def add_agent_to_network(self, agent, parent):
        if self.social_network is None: return
            
        cfg = self.model.cfg
        self.social_network.add_node(agent.network_slot)
        parent_slot = parent.network_slot
        if parent_slot in self.social_network:
            candidates = [s for s in self.social_network.neighbors(parent_slot) if s != agent.network_slot]
            max_deg = getattr(cfg, 'max_network_degree', 0)
            target_edges = getattr(cfg, 'target_offspring_edges', getattr(cfg, 'offspring_network_edges', 0))
            k = int(target_edges)
            if max_deg > 0:
                candidates = [s for s in candidates if self.social_network.degree(s) < max_deg]
                k = min(k, max_deg)
            if k > 0 and len(candidates) > k:
                candidates = list(self.model.rng.choice(candidates, size=k, replace=False))
            for nb in candidates[:k]:
                self.social_network.add_edge(agent.network_slot, nb)

    def remove_agent_from_network(self, agent):
        if self.social_network is not None and agent.network_slot in self.social_network:
            self.social_network.remove_node(agent.network_slot)
            
    def get_neighbors(self, agent):
        use_network = (self.model.cfg.network_type != "none" and self.social_network is not None)
        if not use_network: return []
            
        cache_ok = (self._network_neighbors_cache_step == self.model.steps_run)
        if cache_ok:
            return self._network_neighbors_cache.get(agent.network_slot, [])
        else:
            neighbor_slots = list(self.social_network.neighbors(agent.network_slot))
            return [self._slot_to_agent[s] for s in neighbor_slots if s in self._slot_to_agent]

    @property
    def num_edges(self):
        return self.social_network.number_of_edges() if self.social_network is not None else 0