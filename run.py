import os
import shutil
import pandas as pd
from datetime import datetime

from src.config import load_config
from src.model import AgentsModel
from src.visualization import plot_results


def main():
    try:
        cfg = load_config("config.yaml")
    except FileNotFoundError:
        print("Ошибка: файл config.yaml не найден!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    output_dir = os.path.join("out", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Симуляция «Среда / Мультиагентная Система» (С/МАС)")
    print(f"  Агентов: {cfg.initial_agents}")
    print(f"  Сетка: {cfg.width}x{cfg.height}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Запуск: {run_name}")
    print(f"  Папка вывода: {output_dir}")
    print("=" * 60)

    model = AgentsModel(cfg=cfg, seed=cfg.seed)

    failed = False

    try:
        max_runtime = getattr(cfg, "max_runtime_seconds", 0.0)

        model.run_model(
            cfg.max_steps,
            max_seconds=max_runtime if max_runtime > 0 else None
        )

    except KeyboardInterrupt:
        failed = True
        print("\nСимуляция остановлена пользователем. Сохраняю частичные результаты.")

    except Exception as e:
        failed = True
        print(f"\nОшибка симуляции: {e}")
        print("Сохраняю частичные результаты.")

    finally:
        try:
            df_model = model.datacollector.get_model_vars_dataframe()
            df_model.reset_index(names="Step", inplace=True)
        except Exception as e:
            print(f"Не удалось собрать данные: {e}")
            return

        suffix = "_partial" if failed else ""

        csv_path = os.path.join(output_dir, f"model_data{suffix}.csv")
        parquet_path = os.path.join(output_dir, f"model_data{suffix}.parquet")
        plot_path = os.path.join(output_dir, f"results{suffix}.png")
        comparison_path = os.path.join(output_dir, f"imitation_comparison{suffix}.png")
        config_backup_path = os.path.join(output_dir, "config_used.yaml")

        try:
            df_model.to_csv(csv_path, index=False)
            print(f"Данные сохранены: {csv_path}")
        except Exception as e:
            print(f"Не удалось сохранить CSV: {e}")

        try:
            df_model.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        try:
            plot_results(df_model, save_path=plot_path)
        except Exception as e:
            print(f"Не удалось построить основной график: {e}")

        try:
            from src.visualization import plot_imitation_comparison
            plot_imitation_comparison(df_model, save_path=comparison_path)
        except Exception as e:
            print(f"Не удалось построить график сравнения имитации: {e}")

        try:
            if os.path.exists("config.yaml"):
                shutil.copy2("config.yaml", config_backup_path)
                print(f"Конфигурация сохранена: {config_backup_path}")
        except Exception as e:
            print(f"Не удалось сохранить копию конфига: {e}")

    print("\n" + "=" * 60)
    if failed:
        print("[ЗАВЕРШЕНО С ПРЕДУПРЕЖДЕНИЕМ] Сохранены частичные результаты.")
    else:
        print("[ГОТОВО] Работа завершена успешно.")
    print(f"Результаты в папке: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
