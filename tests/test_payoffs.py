import pytest
from src.config import GamePayoffs

def test_payoffs_matrix():
    """Проверка базовой матрицы выигрышей (Дилемма заключенного)."""
    game = GamePayoffs(R=3.0, S=0.0, T=5.0, P=1.0)
    assert game.payoff('C', 'C') == 3.0
    assert game.payoff('C', 'D') == 0.0
    assert game.payoff('D', 'C') == 5.0
    assert game.payoff('D', 'D') == 1.0

def test_payoffs_symmetry():
    """Проверка корректности возврата выигрышей для ролей."""
    game = GamePayoffs(R=3.0, S=0.0, T=5.0, P=1.0)
    assert game.payoff('C', 'D') == 0.0  # S (наказание за кооперацию)
    assert game.payoff('D', 'C') == 5.0  # T (искушение предать)

def test_payoffs_order():
    """Проверка условия дилеммы заключенного: T > R > P > S."""
    game = GamePayoffs(R=3.0, S=0.0, T=5.0, P=1.0)
    assert game.T > game.R > game.P > game.S
