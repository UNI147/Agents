from dataclasses import dataclass
import yaml


@dataclass
class GamePayoffs:
    R: float
    S: float
    T: float
    P: float

    def payoff(self, my_action: str, other_action: str) -> float:
        table = {('C', 'C'): self.R, ('C', 'D'): self.S,
                 ('D', 'C'): self.T, ('D', 'D'): self.P}
        return table[(my_action, other_action)]


@dataclass
class Config:
    # Сетка
    width: int
    height: int
    # Ресурсы
    max_resource: float
    regen_rate: float
    # Динамика среды
    season_period: int
    season_amplitude: float
    catastrophe_prob: float
    catastrophe_duration: int
    catastrophe_severity: float
    # Агенты
    initial_agents: int
    min_vision: int
    max_vision: int
    min_metabolism: float
    max_metabolism: float
    initial_resource: float
    max_age: int
    # Память (Пункт 1.2)
    memory_size: int
    # Эволюция
    reproduction_threshold: float
    mutation_rate: float
    # Игра (матрица выигрышей)
    R: float
    S: float
    T: float
    P: float
    # Симуляция
    max_steps: int
    seed: int
    # Имитация (Пункт 1.3) — поля с дефолтами ОБЯЗАТЕЛЬНО в конце
    imitation_protocol: str = "none"   # "none", "fermi", "proportional", "pairwise"
    selection_intensity: float = 1.0   # параметр m (интенсивность отбора)

    @property
    def game(self) -> GamePayoffs:
        return GamePayoffs(self.R, self.S, self.T, self.P)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
