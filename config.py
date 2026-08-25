"""Конфигурация прототипа. Все параметры в одном месте."""
from dataclasses import dataclass, field


@dataclass
class GamePayoffs:
    """Матрица дилеммы заключённого (из Nowak, гл. 5; Izquierdo).
    R — награда за взаимную кооперацию, S — «лох»,
    T — искушение дефектора, P — наказание за взаимный дефолт.
    Условие: T > R > P > S.
    """
    R: float = 3.0   # C vs C
    S: float = 0.0   # C vs D
    T: float = 5.0   # D vs C
    P: float = 1.0   # D vs D

    def payoff(self, my_action: str, other_action: str) -> float:
        table = {('C', 'C'): self.R, ('C', 'D'): self.S,
                 ('D', 'C'): self.T, ('D', 'D'): self.P}
        return table[(my_action, other_action)]


@dataclass
class Config:
    # --- Сетка ---
    width: int = 50
    height: int = 50

    # --- Ресурсы ---
    max_resource: float = 4.0       # ёмкость клетки
    regen_rate: float = 1.0         # базовая скорость регенерации за шаг

    # --- Динамика среды ---
    season_period: int = 100        # период сезонного цикла (шагов)
    season_amplitude: float = 0.5   # амплитуда модуляции регенерации
    catastrophe_prob: float = 0.005 # вероятность катастрофы за шаг
    catastrophe_duration: int = 15  # длительность катастрофы (шагов)
    catastrophe_severity: float = 0.3  # множитель регенерации при катастрофе

    # --- Агенты ---
    initial_agents: int = 200
    min_vision: int = 1
    max_vision: int = 6
    min_metabolism: float = 1.0
    max_metabolism: float = 4.0
    initial_resource: float = 15.0
    max_age: int = 200

    # --- Эволюция ---
    reproduction_threshold: float = 40.0  # ресурс для размножения
    mutation_rate: float = 0.1            # вероятность мутации гена

    # --- Игра ---
    game: GamePayoffs = field(default_factory=GamePayoffs)

    # --- Симуляция ---
    max_steps: int = 500
