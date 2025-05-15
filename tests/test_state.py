import pytest
from state import SokobanState

def test_initial_state():
    level = [
        "########",
        "#      #",
        "#  .$  #",
        "#  @   #",
        "#  .$  #",
        "#      #",
        "########"
    ]
    state = SokobanState(level)
    
    # Test player position
    assert state.player_pos == (3, 3)
    
    # Test boxes
    assert len(state.boxes) == 2
    assert (2, 3) in state.boxes
    assert (4, 3) in state.boxes
    
    # Test goals
    assert len(state.goals) == 2
    assert (2, 3) in state.goals
    assert (4, 3) in state.goals

def test_valid_moves():
    level = [
        "########",
        "#      #",
        "#  .$  #",
        "#  @   #",
        "#  .$  #",
        "#      #",
        "########"
    ]
    state = SokobanState(level)
    
    # Test valid moves
    moves = state.get_valid_moves()
    assert len(moves) > 0
    assert all(isinstance(move, tuple) for move in moves)
    assert all(len(move) == 2 for move in moves)

def test_move():
    level = [
        "########",
        "#      #",
        "#  .$  #",
        "#  @   #",
        "#  .$  #",
        "#      #",
        "########"
    ]
    state = SokobanState(level)
    
    # Test moving up
    new_state = state.move((-1, 0))
    assert new_state.player_pos == (2, 3)
    
    # Test moving box
    new_state = new_state.move((0, 1))
    assert new_state.player_pos == (2, 4)
    assert (2, 4) in new_state.boxes

def test_goal_state():
    # Create a state where all boxes are on goals
    level = [
        "########",
        "#      #",
        "#  *   #",
        "#  @   #",
        "#  *   #",
        "#      #",
        "########"
    ]
    state = SokobanState(level)
    assert state.is_goal_state()
    
    # Create a state where not all boxes are on goals
    level = [
        "########",
        "#      #",
        "#  .$  #",
        "#  @   #",
        "#  .$  #",
        "#      #",
        "########"
    ]
    state = SokobanState(level)
    assert not state.is_goal_state() 