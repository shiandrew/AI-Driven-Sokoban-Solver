from typing import List, Tuple, Set, Dict, Optional
from collections import deque
import heapq
import time
from state import SokobanState

class SearchResult:
    def __init__(self, path: List[Tuple[int, int]], nodes_expanded: int, max_depth: int, time_taken: float):
        self.path = path
        self.nodes_expanded = nodes_expanded
        self.max_depth = max_depth
        self.time_taken = time_taken

def bfs(initial_state: SokobanState, time_limit: float = 60.0) -> Optional[SearchResult]:
    """
    Breadth-First Search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        time_limit: Maximum time in seconds to search (default: 60 seconds)
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0, 0.0)
        
    # Initialize data structures
    queue = deque([(initial_state, [])])  # (state, path)
    visited = {initial_state}
    nodes_expanded = 0
    max_depth = 0
    start_time = time.time()
    
    while queue:
        # Check time limit
        if time.time() - start_time > time_limit:
            return None
            
        current_state, path = queue.popleft()
        nodes_expanded += 1
        max_depth = max(max_depth, len(path))
        
        # Get all valid moves
        for move in current_state.get_valid_moves():
            next_state = current_state.move(move)
            
            # Check if we've reached the goal
            if next_state.is_goal_state():
                time_taken = time.time() - start_time
                return SearchResult(path + [move], nodes_expanded, max_depth, time_taken)
                
            # Add new state to queue if not visited
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
                
    time_taken = time.time() - start_time
    return None

def dfs(initial_state: SokobanState, max_depth: int = 100, time_limit: float = 60.0) -> Optional[SearchResult]:
    """
    Depth-First Search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        max_depth: Maximum depth to search (to prevent infinite recursion)
        time_limit: Maximum time in seconds to search (default: 60 seconds)
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0, 0.0)
        
    # Initialize data structures
    stack = [(initial_state, [])]  # (state, path)
    visited = {initial_state}
    nodes_expanded = 0
    current_max_depth = 0
    start_time = time.time()
    
    while stack:
        # Check time limit
        if time.time() - start_time > time_limit:
            return None
            
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
                time_taken = time.time() - start_time
                return SearchResult(path + [move], nodes_expanded, current_max_depth, time_taken)
                
            # Add new state to stack if not visited
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [move]))
                
    time_taken = time.time() - start_time
    return None

def astar(initial_state: SokobanState, time_limit: float = 60.0) -> Optional[SearchResult]:
    """
    A* search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        time_limit: Maximum time in seconds to search (default: 60 seconds)
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def min_box_to_goal_distance(box_pos: Tuple[int, int], goals: Set[Tuple[int, int]]) -> int:
        """Calculate minimum Manhattan distance from a box to any goal."""
        return min(manhattan_distance(box_pos, goal) for goal in goals)
    
    def heuristic(state: SokobanState) -> int:
        """
        Calculate heuristic value for a state.
        Uses sum of minimum distances from each box to its nearest goal.
        """
        total_distance = 0
        for box in state.boxes:
            total_distance += min_box_to_goal_distance(box, state.goals)
        return total_distance
    
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0, 0.0)
    
    # Priority queue for A* search
    # Format: (f_score, state_hash, state, moves)
    frontier = [(0, hash(initial_state), initial_state, [])]
    heapq.heapify(frontier)
    
    # Set to keep track of visited states
    visited = set()
    nodes_expanded = 0
    max_depth = 0
    start_time = time.time()
    
    while frontier:
        # Check time limit
        if time.time() - start_time > time_limit:
            return None
            
        _, _, current_state, moves = heapq.heappop(frontier)
        nodes_expanded += 1
        max_depth = max(max_depth, len(moves))
        
        # Check if we've reached the goal state
        if current_state.is_goal_state():
            time_taken = time.time() - start_time
            return SearchResult(moves, nodes_expanded, max_depth, time_taken)
        
        # Skip if we've seen this state before
        state_hash = hash(current_state)
        if state_hash in visited:
            continue
        visited.add(state_hash)
        
        # Try all valid moves
        for direction in current_state.get_valid_moves():
            next_state = current_state.move(direction)
            next_hash = hash(next_state)
            
            if next_hash not in visited:
                # Calculate f_score = g_score + h_score
                g_score = len(moves) + 1
                h_score = heuristic(next_state)
                f_score = g_score + h_score
                
                # Add to frontier
                heapq.heappush(frontier, (f_score, next_hash, next_state, moves + [direction]))
    
    time_taken = time.time() - start_time
    return None

def ids(initial_state: SokobanState, max_depth: int = 100, time_limit: float = 60.0) -> Optional[SearchResult]:
    """
    Iterative Deepening Search implementation for Sokoban.
    
    Args:
        initial_state: The initial state of the game
        max_depth: Maximum depth to search
        time_limit: Maximum time in seconds to search (default: 60 seconds)
        
    Returns:
        SearchResult containing the solution path and statistics, or None if no solution exists
    """
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0, 0.0)
    
    start_time = time.time()
    total_nodes_expanded = 0
    
    for depth_limit in range(1, max_depth + 1):
        # Check time limit
        if time.time() - start_time > time_limit:
            return None
            
        # Perform depth-limited search
        result = _depth_limited_search(initial_state, depth_limit, start_time, time_limit)
        
        if result is not None:
            total_nodes_expanded += result.nodes_expanded
            time_taken = time.time() - start_time
            return SearchResult(result.path, total_nodes_expanded, result.max_depth, time_taken)
        
        # Add nodes expanded from this iteration
        total_nodes_expanded += result.nodes_expanded if result else 0
    
    time_taken = time.time() - start_time
    return None

def _depth_limited_search(initial_state: SokobanState, depth_limit: int, 
                         start_time: float, time_limit: float) -> Optional[SearchResult]:
    """
    Helper function for iterative deepening search.
    Performs depth-limited search up to the specified depth.
    """
    stack = [(initial_state, [], 0)]  # (state, path, depth)
    visited_at_depth = {}  # Track states at specific depths
    nodes_expanded = 0
    max_depth_reached = 0
    
    while stack:
        # Check time limit
        if time.time() - start_time > time_limit:
            return None
            
        current_state, path, depth = stack.pop()
        nodes_expanded += 1
        max_depth_reached = max(max_depth_reached, depth)
        
        # Check if we've reached the goal
        if current_state.is_goal_state():
            return SearchResult(path, nodes_expanded, max_depth_reached, 0.0)
        
        # Skip if we've reached depth limit
        if depth >= depth_limit:
            continue
        
        # Skip if we've seen this state at a shallower or equal depth
        state_hash = hash(current_state)
        if state_hash in visited_at_depth and visited_at_depth[state_hash] <= depth:
            continue
        visited_at_depth[state_hash] = depth
        
        # Get all valid moves
        for move in current_state.get_valid_moves():
            next_state = current_state.move(move)
            stack.append((next_state, path + [move], depth + 1))
    
    return SearchResult([], nodes_expanded, max_depth_reached, 0.0)

def evaluate_search_algorithm(algorithm, initial_state: SokobanState, time_limit: float = 60.0, **kwargs) -> Dict:
    """
    Evaluate a search algorithm's performance on a given initial state.
    
    Args:
        algorithm: The search algorithm function to evaluate
        initial_state: The initial state to test
        time_limit: Maximum time in seconds to search (default: 60 seconds)
        **kwargs: Additional arguments to pass to the algorithm
        
    Returns:
        Dictionary containing performance metrics
    """
    result = algorithm(initial_state, time_limit=time_limit, **kwargs)
    
    if result is None:
        return {
            'success': False,
            'nodes_expanded': 0,
            'max_depth': 0,
            'solution_length': 0,
            'time_taken': time_limit
        }
        
    return {
        'success': True,
        'nodes_expanded': result.nodes_expanded,
        'max_depth': result.max_depth,
        'solution_length': len(result.path),
        'time_taken': result.time_taken
    }

# Convenience function for backward compatibility
def visualize_solution(initial_state: SokobanState, moves: List[Tuple[int, int]]) -> None:
    """
    Visualize the solution by printing each state in the solution path.
    
    Args:
        initial_state: The starting state
        moves: List of moves to apply
    """
    current_state = initial_state
    print("\nInitial State:")
    print(current_state)
    
    for i, move in enumerate(moves, 1):
        current_state = current_state.move(move)
        print(f"\nStep {i}:")
        print(current_state)