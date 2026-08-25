"""Главный цикл симуляции. Порядок шага:
  1. Среда делает шаг.
  2. Агенты действуют через планировщик.
  3. Эволюция: смерть и размножение.
  4. Сбор данных.
"""
from environment import DynamicEnvironment
from agent import Agent
from scheduler import RandomScheduler
from datacollector import DataCollector


class Simulation:
    def __init__(self, config):
        self.cfg = config
        self.env = DynamicEnvironment(config)
        self.agents = [Agent(self.env, config)
                       for _ in range(config.initial_agents)]
        self.scheduler = RandomScheduler()
        self.datacollector = DataCollector()
        self.step_count = 0

    # ------------------------------------------------------------------
    def step(self):
        # 1. Динамика среды
        self.env.step()

        # 2. Агенты действуют
        self.scheduler.step(self.agents, self.agents)

        # 3. Эволюция: смерть + размножение
        self._evolution_step()

        # 4. Сбор данных
        self.datacollector.collect(self)
        self.step_count += 1

    def _evolution_step(self):
        # Удаляем мёртвых
        self.agents = [a for a in self.agents if a.alive]

        # Размножение
        newborns = []
        for agent in self.agents:
            if agent.can_reproduce():
                newborns.append(agent.reproduce())
        self.agents.extend(newborns)

    # ------------------------------------------------------------------
    def run(self, steps=None):
        max_steps = steps or self.cfg.max_steps
        while self.step_count < max_steps:
            if not self.agents:
                print(f"Популяция вымерла на шаге {self.step_count}")
                break
            self.step()
            if self.step_count % 50 == 0:
                n = len(self.agents)
                nc = sum(1 for a in self.agents if a.genome.strategy == 'C')
                print(f"Шаг {self.step_count:4d} | популяция: {n:4d} | "
                      f"C: {nc} ({nc/n*100:.0f}%) | "
                      f"ресурс среды: {self.env.total_resource:.0f} | "
                      f"сезон: {'☀' if self.env.regen_multiplier > 1 else '❄'}")
        print("Симуляция завершена.")
