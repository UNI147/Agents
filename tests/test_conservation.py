import pytest
import numpy as np
from src.environment import DynamicEnvironment

def test_resource_conservation_harvest():
    """Проверка точного уменьшения ресурса при сборе."""
    rng = np.random.default_rng(42)
    env = DynamicEnvironment(10, 10, 4.0, 0.0, 100, 0.0, 0.0, 15, 0.3, rng)
    
    initial_total = env.total_resource
    x, y = 5, 5
    
    # Агент собирает ресурс
    harvested = env.harvest(x, y, 2.0)
    new_total = env.total_resource
    
    assert initial_total - new_total == pytest.approx(harvested)

def test_resource_conservation_overharvest():
    """Проверка, что нельзя собрать больше, чем есть в ячейке."""
    rng = np.random.default_rng(42)
    env = DynamicEnvironment(10, 10, 4.0, 0.0, 100, 0.0, 0.0, 15, 0.3, rng)
    
    x, y = 5, 5
    available = env.get_resource(x, y)
    
    # Пытаемся собрать в 10 раз больше
    harvested = env.harvest(x, y, available * 10)
    
    assert harvested == pytest.approx(available)
    assert env.get_resource(x, y) == pytest.approx(0.0)

def test_regeneration_conservation():
    """Проверка, что регенерация не превышает capacity."""
    rng = np.random.default_rng(42)
    env = DynamicEnvironment(10, 10, 4.0, 1.0, 100, 0.0, 0.0, 15, 0.3, rng)
    
    # Собираем всё в одной ячейке
    x, y = 5, 5
    env.harvest(x, y, 100.0)
    
    # Делаем много шагов регенерации
    for _ in range(50):
        env.step()
        
    assert env.get_resource(x, y) <= env.capacity[y, x] + 1e-6
