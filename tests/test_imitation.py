import pytest
import numpy as np
from src.config import Config
from src.model import AgentsModel
from src.agent import EcoAgent, Genome

@pytest.fixture
def setup_model():
    params = {
        "width": 5, "height": 5, "max_resource": 4.0, "regen_rate": 0.0,
        "season_period": 100, "season_amplitude": 0.0, "catastrophe_prob": 0.0,
        "catastrophe_duration": 0, "catastrophe_severity": 0.0,
        "initial_agents": 0, "min_vision": 1, "max_vision": 2,
        "min_metabolism": 1.0, "max_metabolism": 2.0, "initial_resource": 15.0,
        "max_age": 100, "reproduction_threshold": 40.0, "mutation_rate": 0.0,
        "R": 3.0, "S": 0.0, "T": 5.0, "P": 1.0, "max_steps": 10, "seed": 42,
        "memory_size": 10, "min_imitation_intensity": 1.0, 
        "max_imitation_intensity": 1.0, "initial_imitation_rate": 1.0
    }
    cfg = Config(**params)
    model = AgentsModel(cfg=cfg, seed=42)
    return model

def create_agent(model, strategy, payoff, imit_type="best_neighbor"):
    genome = Genome(
        vision=1, metabolism=1.0, strategy=strategy, max_age=100,
        imitation_type=imit_type, imitation_intensity=1.0, imitation_rate=1.0
    )
    agent = EcoAgent(model, genome)
    agent.resource = 10.0
    agent.last_payoff = payoff
    return agent

def test_imitate_best_neighbor(setup_model):
    """Агент должен скопировать стратегию соседа с наибольшим payoff."""
    model = setup_model
    model.rng = np.random.default_rng(42)
    
    agent = create_agent(model, "AlwaysC", payoff=1.0, imit_type="best_neighbor")
    neighbor1 = create_agent(model, "AlwaysD", payoff=5.0)
    neighbor2 = create_agent(model, "TFT", payoff=2.0)
    
    agent.try_imitate([agent, neighbor1, neighbor2])
    assert agent.genome.strategy == "AlwaysD"

def test_imitate_none(setup_model):
    """Консерватор (none) не должен менять стратегию."""
    model = setup_model
    model.rng = np.random.default_rng(42)
    
    agent = create_agent(model, "AlwaysC", payoff=1.0, imit_type="none")
    neighbor = create_agent(model, "AlwaysD", payoff=10.0)
    
    agent.try_imitate([agent, neighbor])
    assert agent.genome.strategy == "AlwaysC"

def test_imitate_pairwise_diff_negative(setup_model):
    """Если сосед хуже, агент не должен его копировать."""
    model = setup_model
    model.rng = np.random.default_rng(42)
    
    agent = create_agent(model, "AlwaysC", payoff=5.0, imit_type="pairwise_diff")
    neighbor = create_agent(model, "AlwaysD", payoff=1.0)
    
    agent.try_imitate([agent, neighbor])
    assert agent.genome.strategy == "AlwaysC"

def test_imitate_fermi_logit(setup_model):
    """Проверка, что Ферми-правило работает (смена стратегии при большой разнице)."""
    model = setup_model
    model.rng = np.random.default_rng(42) 
    
    agent = create_agent(model, "AlwaysC", payoff=0.0, imit_type="fermi_m")
    agent.genome.imitation_intensity = 10.0 # Высокая чувствительность
    neighbor = create_agent(model, "AlwaysD", payoff=10.0)
    
    # При m=10 и diff=10, prob = 1 / (1 + exp(-100)) ~ 1.0
    agent.try_imitate([agent, neighbor])
    assert agent.genome.strategy == "AlwaysD"