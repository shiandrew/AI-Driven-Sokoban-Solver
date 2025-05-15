import pytest
from state import SokobanState
from search import bfs, dfs, evaluate_search_algorithm

def test_bfs_simple_level():
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
    
    result = bfs(state)
    assert result is not None
    assert result.path is not None
    assert len(result.path) > 0
    assert result.nodes_expanded > 0
    assert result.max_depth > 0

def test_dfs_simple_level():
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
    
    result = dfs(state)
    assert result is not None
    assert result.path is not None
    assert len(result.path) > 0
    assert result.nodes_expanded > 0
    assert result.max_depth > 0

def test_impossible_level():
    level = [
        "########",
        "#      #",
        "#  .$  #",
        "#  @   #",
        "#  .$  #",
        "#  #   #",
        "########"
    ]
    state = SokobanState(level)
    
    # This level should be impossible to solve
    bfs_result = bfs(state)
    assert bfs_result is None
    
    dfs_result = dfs(state)
    assert dfs_result is None

def test_evaluation():
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
    
    # Test BFS evaluation
    bfs_metrics = evaluate_search_algorithm(bfs, state)
    assert 'success' in bfs_metrics
    assert 'nodes_expanded' in bfs_metrics
    assert 'max_depth' in bfs_metrics
    assert 'solution_length' in bfs_metrics
    
    # Test DFS evaluation
    dfs_metrics = evaluate_search_algorithm(dfs, state)
    assert 'success' in dfs_metrics
    assert 'nodes_expanded' in dfs_metrics
    assert 'max_depth' in dfs_metrics
    assert 'solution_length' in dfs_metrics 