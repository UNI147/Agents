from src.config import Config
from src.model import AgentsModel

def test_determinism():
    # Минимальные параметры для быстрого теста
    params = {
        "width": 10, "height": 10, "max_resource": 4.0, "regen_rate": 0.0,
        "season_period": 100, "season_amplitude": 0.0, "catastrophe_prob": 0.0,
        "catastrophe_duration": 0, "catastrophe_severity": 0.0,
        "initial_agents": 10, "min_vision": 1, "max_vision": 2,
        "min_metabolism": 1.0, "max_metabolism": 2.0, "initial_resource": 15.0,
        "max_age": 100, "reproduction_threshold": 40.0, "mutation_rate": 0.1,
        "R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0, "max_steps": 10, "seed": 42
    }
    cfg = Config(**params)
    
    # Два независимых запуска с одинаковым сидом
    model1 = AgentsModel(cfg=cfg, seed=42)
    model1.run_model(5)
    df1 = model1.datacollector.get_model_vars_dataframe()
    
    model2 = AgentsModel(cfg=cfg, seed=42)
    model2.run_model(5)
    df2 = model2.datacollector.get_model_vars_dataframe()
    
    # Данные должны совпадать бит-в-бит
    assert df1.equals(df2)