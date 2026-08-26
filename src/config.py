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
    width: int
    height: int
    max_resource: float
    regen_rate: float
    season_period: int
    season_amplitude: float
    catastrophe_prob: float
    catastrophe_duration: int
    catastrophe_severity: float
    initial_agents: int
    min_vision: int
    max_vision: int
    min_metabolism: float
    max_metabolism: float
    initial_resource: float
    max_age: int
    memory_size: int
    reproduction_threshold: float
    mutation_rate: float
    R: float
    S: float
    T: float
    P: float
    max_steps: int
    seed: int

    # === НОВЫЕ ПАРАМЕТРЫ ДЛЯ ПУНКТА 1.3 ===
    min_imitation_intensity: float = 0.1   # Минимальный параметр m
    max_imitation_intensity: float = 10.0  # Максимальный параметр m
    initial_imitation_rate: float = 0.2    # Начальная вероятность имитации

    @property
    def game(self) -> GamePayoffs:
        return GamePayoffs(self.R, self.S, self.T, self.P)

    @property
    def max_payoff_difference(self) -> float:
        """Максимально возможная разница payoff — нужна для pairwise_diff."""
        return self.T - self.S  # max(T, R, P, S) - min(...)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
