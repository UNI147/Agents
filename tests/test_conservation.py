import pytest
import numpy as np
from src.environment import DynamicEnvironment

def test_resource_conservation():
    rng = np.random.default_rng(42)
    env = DynamicEnvironment(10, 10, 4.0, 0.0, 100, 0.0, 0.0, 15, 0.3, rng)
    
    initial_total = env.total_resource
    x, y = 5, 5
    
    # Агент собирает ресурс
    harvested = env.harvest(x, y, 2.0)
    new_total = env.total_resource
    
    # Ресурс среды должен уменьшиться ровно на величину сбора агента
    assert initial_total - new_total == pytest.approx(harvested)