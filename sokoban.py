from state import SokobanState
from search import bfs, dfs, evaluate_search_algorithm

def load_level(level_number: int) -> SokobanState:
    """
    Load a predefined Sokoban level.
    
    Args:
        level_number: The number of the level to load
        
    Returns:
        SokobanState representing the level
    """
    # Example level (you can add more levels)
    levels = {
        1: [
            "########",
            "#      #",
            "#  .$  #",
            "#  @   #",
            "#  .$  #",
            "#      #",
            "########"
        ]
    }
    
    if level_number not in levels:
        raise ValueError(f"Level {level_number} not found")
        
    return SokobanState(levels[level_number])

def print_state(state: SokobanState):
    """Print the current state of the game."""
    print("\n".join(state.level))
    print()

def main():
    # Load the first level
    initial_state = load_level(1)
    print("Initial state:")
    print_state(initial_state)
    
    # Try solving with BFS
    print("Solving with BFS...")
    bfs_result = bfs(initial_state)
    if bfs_result:
        print(f"Solution found! Path length: {len(bfs_result.path)}")
        print(f"Nodes expanded: {bfs_result.nodes_expanded}")
        print(f"Max depth: {bfs_result.max_depth}")
    else:
        print("No solution found with BFS")
        
    # Try solving with DFS
    print("\nSolving with DFS...")
    dfs_result = dfs(initial_state)
    if dfs_result:
        print(f"Solution found! Path length: {len(dfs_result.path)}")
        print(f"Nodes expanded: {dfs_result.nodes_expanded}")
        print(f"Max depth: {dfs_result.max_depth}")
    else:
        print("No solution found with DFS")
        
    # Evaluate both algorithms
    print("\nEvaluating algorithms...")
    bfs_metrics = evaluate_search_algorithm(bfs, initial_state)
    dfs_metrics = evaluate_search_algorithm(dfs, initial_state)
    
    print("\nBFS Metrics:")
    for key, value in bfs_metrics.items():
        print(f"{key}: {value}")
        
    print("\nDFS Metrics:")
    for key, value in dfs_metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 