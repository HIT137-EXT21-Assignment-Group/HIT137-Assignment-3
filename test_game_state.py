import pytest
from game_state import GameState

# This acts as our fake data for all tests
@pytest.fixture
def sample_regions():
    # Region 1 center: (70, 70). Region 2 center: (215, 215)
    return [(50, 50, 40, 40), (200, 200, 30, 30)]

@pytest.fixture
def game(sample_regions):
    return GameState(sample_regions)

def test_correct_click_returns_found(game):
    # Click exactly in the middle of the first region
    status, region = game.register_click(70, 70)
    assert status in ["found", "level_complete"]

def test_wrong_click_counts_as_mistake(game):
    # Click at 0,0 (far away from any region)
    status, region = game.register_click(0, 0)
    assert status == "mistake"
    assert game.mistakes == 1

def test_three_mistakes_triggers_game_over(game):
    game.register_click(0, 0) # Mistake 1
    game.register_click(0, 0) # Mistake 2
    status, region = game.register_click(0, 0) # Mistake 3
    assert status == "game_over"
    assert game.is_game_over() is True

def test_reveal_all_marks_all_unfound(game):
    revealed = game.reveal_all()
    assert len(revealed) == 2
    assert len(game.unfound_regions) == 0

def test_score_increments_on_find(game):
    game.register_click(70, 70)
    assert game.cumulative_score == 1