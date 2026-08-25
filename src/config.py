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
    reproduction_threshold: float
    mutation_rate: float
    R: float
    S: float
    T: float
    P: float
    max_steps: int
    seed: int

    @property
    def game(self) -> GamePayoffs:
        return GamePayoffs(self.R, self.S, self.T, self.P)

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
