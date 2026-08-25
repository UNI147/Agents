import pytest
from src.config import GamePayoffs

def test_payoffs_matrix():
    game = GamePayoffs(R=3.0, S=0.0, T=5.0, P=1.0)
    assert game.payoff('C', 'C') == 3.0
    assert game.payoff('C', 'D') == 0.0
    assert game.payoff('D', 'C') == 5.0
    assert game.payoff('D', 'D') == 1.0