from dataclasses import dataclass, fields
import yaml

@dataclass
class GamePayoffs:
    R: float
    S: float
    T: float
    P: float

    def payoff(self, my_action: str, other_action: str) -> float:
        table = {
            ('C', 'C'): self.R,
            ('C', 'D'): self.S,
            ('D', 'C'): self.T,
            ('D', 'D'): self.P,
        }
        return table[(my_action, other_action)]

@dataclass
class Config:
    # =========================================================
    # Сначала обязательные поля без значений по умолчанию
    # =========================================================
    width: int
    height: int
    max_resource: float
    regen_rate: float
    season_period: int
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

    # =========================================================
    # Поля, которых может не быть в config.yaml
    # =========================================================
    initial_agents: int = 100
    season_amplitude: float = 0.0
    catastrophe_prob: float = 0.0
    catastrophe_duration: int = 0
    catastrophe_severity: float = 0.0

    max_spice: float = 4.0
    regen_rate_spice: float = 1.0
    min_metabolism_spice: float = 1.0
    max_metabolism_spice: float = 4.0
    initial_spice: float = 15.0

    min_imitation_intensity: float = 0.1
    max_imitation_intensity: float = 10.0
    initial_imitation_rate: float = 0.2

    network_type: str = "none"
    network_param_m: int = 3
    network_param_k: int = 4
    network_param_p: float = 0.1

    network_payoff_mode: str = "mean"
    offspring_network_edges: int = 0
    max_network_degree: int = 0

    use_food_carrying_capacity: bool = True
    max_population: int = 0
    max_runtime_seconds: float = 0.0

    group_selection_enabled: bool = False
    num_groups: int = 10
    group_selection_intensity: float = 0.3
    group_migration_rate: float = 0.02
    group_competition_step: int = 50

    trade_enabled: bool = True

    # === ЗАГРЯЗНЕНИЕ / ИСТОЩЕНИЕ (Пункт 2.2) ===
    pollution_enabled: bool = True
    pollution_production_rate: float = 0.15
    pollution_consumption_rate: float = 0.25
    pollution_diffusion_rate: float = 0.20
    pollution_decay_rate: float = 0.05
    pollution_capacity_impact: float = 1.5

    # =========================================================
    # Свойства
    # =========================================================
    @property
    def game(self) -> GamePayoffs:
        return GamePayoffs(self.R, self.S, self.T, self.P)

    @property
    def max_payoff_difference(self) -> float:
        return self.T - self.S

    @property
    def target_offspring_edges(self) -> int:
        if self.offspring_network_edges > 0:
            return max(1, int(self.offspring_network_edges))
        if self.network_type == "barabasi_albert":
            return max(1, int(2 * self.network_param_m))
        if self.network_type == "watts_strogatz":
            return max(1, int(self.network_param_k))
        if self.network_type == "random":
            expected_degree = self.network_param_p * self.initial_agents
            return max(1, int(round(expected_degree)))
        return max(1, int(self.network_param_m))

    @property
    def population_capacity(self) -> int:
        if self.max_population > 0:
            return int(self.max_population)
        if not self.use_food_carrying_capacity:
            return 10**9
        if self.min_metabolism <= 0:
            return 10**9
        if self.regen_rate > 0:
            energy_flow = self.width * self.height * self.regen_rate
        else:
            energy_flow = self.width * self.height * self.max_resource
        return max(1, int(energy_flow / self.min_metabolism))

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(
            "config.yaml должен содержать словарь параметров. "
            "Названия секций нужно закомментировать через #."
        )

    known_fields = {f.name for f in fields(Config)}
    unknown_fields = sorted(set(data.keys()) - known_fields)

    if unknown_fields:
        print(f"Предупреждение: неизвестные ключи в {path}: {unknown_fields}")

    filtered_data = {
        key: value
        for key, value in data.items()
        if key in known_fields
    }

    return Config(**filtered_data)
