import pytest
import numpy as np
from src.agent import Genome, IMITATION_TYPES

def test_genome_mutation_bounds():
    """Проверка, что мутации не выходят за заданные границы."""
    rng = np.random.default_rng(42)
    g = Genome(vision=3, metabolism=2.0, strategy="TFT", max_age=100, 
               imitation_type="fermi_m", imitation_intensity=5.0, imitation_rate=0.5)
    
    # Мутируем 1000 раз с вероятностью 1.0
    for _ in range(1000):
        g = g.mutate(
            rate=1.0, 
            min_vision=1, max_vision=6, 
            min_metabolism=1.0, max_metabolism=4.0,
            min_intensity=0.1, max_intensity=10.0,
            rng=rng
        )
        
        assert 1 <= g.vision <= 6
        assert 1.0 <= g.metabolism <= 4.0
        assert 0.1 <= g.imitation_intensity <= 10.0
        assert 0.0 <= g.imitation_rate <= 1.0
        assert g.max_age >= 50

def test_genome_mutation_types():
    """Проверка, что тип имитации мутирует в допустимые значения."""
    rng = np.random.default_rng(42)
    g = Genome(vision=3, metabolism=2.0, strategy="TFT", max_age=100)
    
    mutated_types = set()
    for _ in range(500):
        g = g.mutate(1.0, 1, 6, 1.0, 4.0, 0.1, 10.0, rng)
        mutated_types.add(g.imitation_type)
        
    # Все типы должны быть из допустимого списка
    assert mutated_types.issubset(set(IMITATION_TYPES))