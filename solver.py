from typing import List, Tuple, Set, Dict, Optional
import heapq
from state import SokobanState

class SokobanSolver:
    def __init__(self):
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def min_box_to_goal_distance(self, box_pos: Tuple[int, int], goals: Set[Tuple[int, int]]) -> int:
        """Calculate minimum Manhattan distance from a box to any goal."""
        return min(self.manhattan_distance(box_pos, goal) for goal in goals)
    
    def heuristic(self, state: SokobanState) -> int:
        """
        Calculate heuristic value for a state.
        Uses sum of minimum distances from each box to its nearest goal.
        """
        total_distance = 0
        for box in state.boxes:
            total_distance += self.min_box_to_goal_distance(box, state.goals)
        return total_distance
    
    def a_star_search(self, initial_state: SokobanState) -> Optional[List[Tuple[int, int]]]:
        """
        A* search implementation for Sokoban.
        
        Returns:
            List of moves (directions) to solve the puzzle, or None if no solution exists
        """
        # Priority queue for A* search
        # Format: (f_score, state_hash, state, moves)
        frontier = [(0, hash(initial_state), initial_state, [])]
        heapq.heapify(frontier)
        
        # Set to keep track of visited states
        visited = set()
        
        while frontier:
            _, _, current_state, moves = heapq.heappop(frontier)
            
            # Check if we've reached the goal state
            if current_state.is_goal_state():
                return moves
            
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
                    h_score = self.heuristic(next_state)
                    f_score = g_score + h_score
                    
                    # Add to frontier
                    heapq.heappush(frontier, (f_score, next_hash, next_state, moves + [direction]))
        
        return None  # No solution found

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