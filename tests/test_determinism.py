import pytest
from src.config import Config
from src.model import AgentsModel

def get_test_config():
    """Возвращает полный набор параметров для теста, включая новые поля."""
    params = {
        "width": 10, "height": 10, "max_resource": 4.0, "regen_rate": 0.0,
        "season_period": 100, "season_amplitude": 0.0, "catastrophe_prob": 0.0,
        "catastrophe_duration": 0, "catastrophe_severity": 0.0,
        "initial_agents": 10, "min_vision": 1, "max_vision": 2,
        "min_metabolism": 1.0, "max_metabolism": 2.0, "initial_resource": 15.0,
        "max_age": 100, "reproduction_threshold": 40.0, "mutation_rate": 0.0, # Отключаем мутации
        "R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0, "max_steps": 10, "seed": 42,
        "memory_size": 10, 
        "min_imitation_intensity": 0.1, 
        "max_imitation_intensity": 10.0, 
        "initial_imitation_rate": 1.0 # 100% имитация для теста
    }
    return Config(**params)

def test_determinism():
    """Два независимых запуска с одинаковым сидом должны давать бит-в-бит идентичные результаты."""
    cfg = get_test_config()
    
    model1 = AgentsModel(cfg=cfg, seed=42)
    model1.run_model(5)
    df1 = model1.datacollector.get_model_vars_dataframe()
    
    model2 = AgentsModel(cfg=cfg, seed=42)
    model2.run_model(5)
    df2 = model2.datacollector.get_model_vars_dataframe()
    
    assert df1.equals(df2), "Модель не детерминирована при одинаковых сидах!"
