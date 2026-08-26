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

    # === ПАРАМЕТРЫ СОЦИАЛЬНОЙ СЕТИ ===
    network_type: str = "none"
    network_param_m: int = 3
    network_param_k: int = 4
    network_param_p: float = 0.1

    # === СОЦИАЛЬНОЕ ОБУЧЕНИЕ ===
    min_imitation_intensity: float = 0.1
    max_imitation_intensity: float = 10.0
    initial_imitation_rate: float = 0.2

    # === ПРОИЗВОДИТЕЛЬНОСТЬ И УСТОЙЧИВОСТЬ ===
    # "mean" — payoff за взаимодействие, консистентно с клеточным режимом.
    # "sum"  — суммарный payoff, как было; используйте осторожно.
    network_payoff_mode: str = "mean"

    # Сколько рёбер наследует ребёнок.
    # 0 означает автоматический выбор по типу сети.
    offspring_network_edges: int = 0

    # Максимальная степень узла.
    # 0 — без ограничения.
    max_network_degree: int = 0

    # Экологическая ёмкость по пищевому потоку.
    use_food_carrying_capacity: bool = True

    # Ручное перекрытие ёмкости.
    # 0 — автоматически.
    max_population: int = 0

    # Ограничение времени выполнения всей симуляции в секундах.
    # 0 — без ограничения.
    max_runtime_seconds: float = 0.0

    @property
    def game(self) -> GamePayoffs:
        return GamePayoffs(self.R, self.S, self.T, self.P)

    @property
    def max_payoff_difference(self) -> float:
        """Максимально возможная разница payoff — нужна для pairwise_diff."""
        return self.T - self.S

    @property
    def target_offspring_edges(self) -> int:
        """
        Сколько социальных связей разумно передавать потомку.

        Значение выводится из параметров сети:
        - barabasi_albert: средний градус ≈ 2 * m;
        - watts_strogatz: средний градус ≈ k;
        - random: ожидаемый средний градус ≈ p * initial_agents.

        Это не магическое число, а оценка исходной плотности графа.
        """
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
        """
        Консервативная экологическая ёмкость.

        Идея:
        - среда восстанавливает width * height * regen_rate ресурса за шаг;
        - минимально возможный метаболизм — min_metabolism;
        - значит, устойчиво поддерживаемая численность не выше:
          (width * height * regen_rate) / min_metabolism.

        Если regen_rate == 0, используем запас max_resource.
        """
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
    return Config(**data)
