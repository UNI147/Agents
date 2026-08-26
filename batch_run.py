import os
import pandas as pd
from src.config import load_config
from src.model import AgentsModel
from mesa.batchrunner import batch_run # Изменено для Mesa 3.x

def run_batch():
    cfg = load_config("config.yaml")
    
    params = vars(cfg).copy()
    
    # Базовые параметры свипа
    params["mutation_rate"] = [0.05, 0.1, 0.2] 
    params["season_period"] = [50, 100, 200]
    
    # === НОВЫЕ ПАРАМЕТРЫ ДЛЯ СВИПА (Социальное обучение и среда) ===
    params["initial_imitation_rate"] = [0.1, 0.3, 0.5]  # Вероятность попытки имитации
    params["memory_size"] = [5, 10, 20]                 # Размер памяти агентов
    params["catastrophe_prob"] = [0.0, 0.005, 0.02]     # Вероятность катастрофы
    
    params["max_steps"] = 100 
    
    print("Запуск batch_run (может занять время)...")
    
    results = batch_run(
        AgentsModel,
        parameters=params,
        iterations=2, # Можно вернуть 3 для большей стат. значимости
        max_steps=params["max_steps"],
        number_processes=1, # Поставьте None для multiprocessing
        data_collection_period=10,
        display_progress=True,
    )
    
    results_df = pd.DataFrame(results)
    os.makedirs("out", exist_ok=True)
    results_df.to_csv("out/batch_results.csv", index=False)
    print(f"Batch results сохранены в out/batch_results.csv ({len(results_df)} строк)")

if __name__ == '__main__':
    run_batch()
