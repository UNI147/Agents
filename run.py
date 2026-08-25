"""Точка входа."""
from config import Config
from simulation import Simulation
from visualization import plot_results


def main():
    cfg = Config()
    sim = Simulation(cfg)
    print("=" * 60)
    print("Прототип эволюционной мультиагентной системы")
    print(f"  Агентов: {cfg.initial_agents}")
    print(f"  Сетка: {cfg.width}x{cfg.height}")
    print(f"  Сезон: период={cfg.season_period}, "
          f"амплитуда={cfg.season_amplitude}")
    print(f"  Катастрофы: p={cfg.catastrophe_prob}")
    print("=" * 60)

    sim.run()
    plot_results(sim.datacollector, save_path='results.png')


if __name__ == '__main__':
    main()
