"""Сбор статистики симуляции."""
from collections import defaultdict
from statistics import mean


class DataCollector:
    def __init__(self):
        self.data = defaultdict(list)

    def collect(self, simulation):
        agents = simulation.agents
        env = simulation.env
        self.data['step'].append(simulation.step_count)
        self.data['population'].append(len(agents))
        self.data['total_env_resource'].append(env.total_resource)
        self.data['season_phase'].append(env.season_phase)
        self.data['catastrophe_active'].append(1 if env.catastrophe_active else 0)

        if agents:
            self.data['avg_agent_resource'].append(
                mean(a.resource for a in agents))
            self.data['avg_vision'].append(
                mean(a.genome.vision for a in agents))
            self.data['avg_metabolism'].append(
                mean(a.genome.metabolism for a in agents))

            # Частоты стратегий — основа для репликаторной динамики
            n_c = sum(1 for a in agents if a.genome.strategy == 'C')
            n_d = len(agents) - n_c
            self.data['n_cooperators'].append(n_c)
            self.data['n_defectors'].append(n_d)
            total = n_c + n_d
            self.data['freq_cooperators'].append(n_c / total if total else 0)
        else:
            self.data['avg_agent_resource'].append(0)
            self.data['avg_vision'].append(0)
            self.data['avg_metabolism'].append(0)
            self.data['n_cooperators'].append(0)
            self.data['n_defectors'].append(0)
            self.data['freq_cooperators'].append(0)
