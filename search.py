from typing import List, Tuple, Set, Dict, Optional
from collections import deque
from state import SokobanState

class SearchResult:
    def __init__(self, path: List[Tuple[int, int]], nodes_expanded: int, max_depth: int):
        self.path = path
        self.nodes_expanded = nodes_expanded
        self.max_depth = max_depth

def bfs(initial_state: SokobanState) -> Optional[SearchResult]:
    """
    Breadth-First Search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0)
        
    # Initialize data structures
    queue = deque([(initial_state, [])])  # (state, path)
    visited = {initial_state}
    nodes_expanded = 0
    max_depth = 0
    
    while queue:
        current_state, path = queue.popleft()
        nodes_expanded += 1
        max_depth = max(max_depth, len(path))
        
        # Get all valid moves
        for move in current_state.get_valid_moves():
            next_state = current_state.move(move)
            
            # Check if we've reached the goal
            if next_state.is_goal_state():
                return SearchResult(path + [move], nodes_expanded, max_depth)
                
            # Add new state to queue if not visited
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
                
    return None

def dfs(initial_state: SokobanState, max_depth: int = 100) -> Optional[SearchResult]:
    """
    Depth-First Search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        max_depth: Maximum depth to search (to prevent infinite recursion)
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0)
        
    # Initialize data structures
    stack = [(initial_state, [])]  # (state, path)
    visited = {initial_state}
    nodes_expanded = 0
    current_max_depth = 0
    
    while stack:
        current_state, path = stack.pop()
        nodes_expanded += 1
        current_max_depth = max(current_max_depth, len(path))
        
        # Skip if we've reached max depth
        if len(path) >= max_depth:
            continue
            
        # Get all valid moves
        for move in current_state.get_valid_moves():
            next_state = current_state.move(move)
            
            # Check if we've reached the goal
            if next_state.is_goal_state():
                return SearchResult(path + [move], nodes_expanded, current_max_depth)
                
            # Add new state to stack if not visited
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [move]))
                
    return None

def evaluate_search_algorithm(algorithm, initial_state: SokobanState) -> Dict:
    """
    Evaluate a search algorithm's performance on a given initial state.
    
    Args:
        algorithm: The search algorithm function to evaluate
        initial_state: The initial state to test
        
    Returns:
        Dictionary containing performance metrics
    """
    result = algorithm(initial_state)
    
    if result is None:
        return {
            'success': False,
            'nodes_expanded': 0,
            'max_depth': 0,
            'solution_length': 0
        }
        
    return {
        'success': True,
        'nodes_expanded': result.nodes_expanded,
        'max_depth': result.max_depth,
        'solution_length': len(result.path)
    } 