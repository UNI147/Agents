import os
import shutil
import pandas as pd
from datetime import datetime
from src.config import load_config
from src.model import AgentsModel
from src.visualization import plot_results

def main():
    # 1. Загрузка конфигурации
    try:
        cfg = load_config("config.yaml")
    except FileNotFoundError:
        print("Ошибка: файл config.yaml не найден!")
        return

    # 2. Генерация уникальной метки времени
    # Формат: YYYYMMDD_HHMMSS (удобно для сортировки по имени файла)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    
    # 3. Создание директории для результатов конкретного запуска
    output_dir = os.path.join("out", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Эволюционная мультиагентная система")
    print(f"  Агентов: {cfg.initial_agents}, Сетка: {cfg.width}x{cfg.height}, Seed: {cfg.seed}")
    print(f"  Запуск: {run_name}")
    print(f"  Папка вывода: {output_dir}")
    print("=" * 60)

    # 4. Инициализация и запуск модели
    model = AgentsModel(cfg=cfg, seed=cfg.seed)
    model.run_model(cfg.max_steps)

    # 5. Сбор данных
    df_model = model.datacollector.get_model_vars_dataframe()
    df_model.reset_index(names='Step', inplace=True)

    # Пути к файлам
    csv_path = os.path.join(output_dir, "model_data.csv")
    parquet_path = os.path.join(output_dir, "model_data.parquet")
    plot_path = os.path.join(output_dir, "results.png")
    config_backup_path = os.path.join(output_dir, "config_used.yaml")

    # 6. Сохранение данных
    df_model.to_csv(csv_path, index=False)
    print(f"Данные сохранены: {csv_path}")

    try:
        df_model.to_parquet(parquet_path, index=False)
    except ImportError:
        pass # Игнорируем, если нет pyarrow

    # 7. Сохранение графика
    plot_results(df_model, save_path=plot_path)

    # === Сравнение успешности типов имитаторов ===
    from src.visualization import plot_imitation_comparison
    comparison_path = os.path.join(output_dir, "imitation_comparison.png")
    plot_imitation_comparison(df_model, save_path=comparison_path)

    # 8. Бэкап конфигурации (очень полезно для воспроизводимости)
    if os.path.exists("config.yaml"):
        shutil.copy2("config.yaml", config_backup_path)
        print(f"Конфигурация сохранена: {config_backup_path}")

    print("\n" + "=" * 60)
    print(f"[ГОТОВО] Работа завершена успешно.")
    print(f"Результаты в папке: {os.path.abspath(output_dir)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
