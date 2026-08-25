"""Планировщик: определяет порядок активации агентов."""
import random


class RandomScheduler:
    """Каждый шаг агенты активируются в случайном порядке."""
    def step(self, agents, all_agents):
        active = [a for a in agents if a.alive]
        random.shuffle(active)
        for agent in active:
            agent.step(all_agents)


class SequentialScheduler:
    """Агенты активируются последовательно (для детерминизма)."""
    def step(self, agents, all_agents):
        for agent in agents:
            if agent.alive:
                agent.step(all_agents)
