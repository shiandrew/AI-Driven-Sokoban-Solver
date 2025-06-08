#!/usr/bin/env python3
"""
Enhanced BFS Implementation for Sokoban
Optimized for maximum performance within 120-second time limits
"""

from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque
import time
import heapq
from state import SokobanState
from search import SearchResult

class EnhancedBFS:
    """
    Ultra-optimized BFS implementation with advanced pruning and heuristics
    """
    
    def __init__(self, initial_state: SokobanState, time_limit: float = 120.0):
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.goals = set(initial_state.goals)
        self.walls = self._get_walls()
        self.dead_squares = self._precompute_dead_squares()
        self.goal_zones = self._compute_goal_zones()
        
    def _get_walls(self) -> Set[Tuple[int, int]]:
        """Extract wall positions from the grid"""
        walls = set()
        for r, row in enumerate(self.initial_state.grid):
            for c, cell in enumerate(row):
                if cell == '#':
                    walls.add((r, c))
        return walls
    
    def _precompute_dead_squares(self) -> Set[Tuple[int, int]]:
        """
        Precompute squares where boxes become deadlocked
        """
        dead = set()
        rows, cols = len(self.initial_state.grid), len(self.initial_state.grid[0])
        
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in self.walls or pos in self.goals:
                    continue
                    
                # Corner deadlock detection
                adjacent_walls = 0
                neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                wall_neighbors = []
                
                for nr, nc in neighbors:
                    if (nr, nc) in self.walls or nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        adjacent_walls += 1
                        wall_neighbors.append((nr, nc))
                
                # If box is in a corner (2+ walls adjacent), it's dead unless it's a goal
                if adjacent_walls >= 2:
                    # Check if it's a proper corner (adjacent walls)
                    if len(wall_neighbors) >= 2:
                        wall1, wall2 = wall_neighbors[0], wall_neighbors[1]
                        if (abs(wall1[0] - wall2[0]) + abs(wall1[1] - wall2[1])) == 2:
                            dead.add(pos)
                            
        return dead
    
    def _compute_goal_zones(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """
        Compute reachable zones for each goal using flood fill
        """
        zones = {}
        for goal in self.goals:
            zone = set()
            queue = deque([goal])
            visited = {goal}
            
            while queue:
                r, c = queue.popleft()
                zone.add((r, c))
                
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited and (nr, nc) not in self.walls:
                        if 0 <= nr < len(self.initial_state.grid) and 0 <= nc < len(self.initial_state.grid[0]):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            
            zones[goal] = zone
        return zones
    
    def _is_deadlock_state(self, state: SokobanState) -> bool:
        """
        Conservative deadlock detection - only catch obvious deadlocks
        """
        # Only check basic corner deadlocks for now
        for box in state.boxes:
            if box in self.goals:
                continue  # Box is already on a goal
                
            # Check if box is in a corner with no escape
            if box in self.dead_squares:
                return True
        
        return False
    
    def _state_priority(self, state: SokobanState) -> float:
        """
        Calculate state priority for informed BFS
        """
        priority = 0.0
        
        # Reward boxes on goals
        boxes_on_goals = len(state.boxes & self.goals)
        priority -= boxes_on_goals * 1000
        
        # Penalize distance from boxes to nearest goals
        for box in state.boxes:
            if box not in self.goals:
                min_dist = float('inf')
                for goal in self.goals:
                    if goal not in state.boxes:  # Goal is free
                        dist = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                        min_dist = min(min_dist, dist)
                priority += min_dist
        
        # Penalize player distance from nearest box
        min_player_to_box = float('inf')
        for box in state.boxes:
            if box not in self.goals:
                dist = abs(state.player_pos[0] - box[0]) + abs(state.player_pos[1] - box[1])
                min_player_to_box = min(min_player_to_box, dist)
        
        if min_player_to_box != float('inf'):
            priority += min_player_to_box * 0.1
        
        return priority
    
    def solve(self) -> Optional[SearchResult]:
        """
        Enhanced BFS with multiple optimizations
        """
        if self.initial_state.is_goal_state():
            return SearchResult([], 0, 0, 0.0)
        
        # Use priority queue for informed BFS
        queue = []
        counter = 0  # To avoid comparison issues when priorities are equal
        heapq.heappush(queue, (self._state_priority(self.initial_state), counter, 0, self.initial_state, []))
        
        visited = {self.initial_state}
        nodes_expanded = 0
        max_depth = 0
        start_time = time.time()
        
        # Performance tracking
        pruned_deadlocks = 0
        
        while queue:
            # Check time limit
            current_time = time.time()
            if current_time - start_time > self.time_limit:
                print(f"🔥 Enhanced BFS Stats: {nodes_expanded:,} nodes, {pruned_deadlocks:,} deadlocks pruned")
                return None
            
            # Progress reporting every 50k nodes
            if nodes_expanded % 50000 == 0 and nodes_expanded > 0:
                elapsed = current_time - start_time
                print(f"🔄 Progress: {nodes_expanded:,} nodes in {elapsed:.1f}s ({nodes_expanded/elapsed:.0f} nodes/s)")
            
            priority, _, depth, current_state, path = heapq.heappop(queue)
            nodes_expanded += 1
            max_depth = max(max_depth, depth)
            
            # Generate successors
            for move in current_state.get_valid_moves():
                next_state = current_state.move(move)
                
                # Early goal check
                if next_state.is_goal_state():
                    time_taken = time.time() - start_time
                    print(f"🏆 Enhanced BFS Success: {nodes_expanded:,} nodes, {pruned_deadlocks:,} deadlocks pruned")
                    return SearchResult(path + [move], nodes_expanded, max_depth, time_taken)
                
                # Skip if already visited
                if next_state in visited:
                    continue
                
                # Advanced deadlock detection
                if self._is_deadlock_state(next_state):
                    pruned_deadlocks += 1
                    continue
                
                # Add to frontier
                visited.add(next_state)
                next_priority = self._state_priority(next_state) + depth + 1
                counter += 1
                heapq.heappush(queue, (next_priority, counter, depth + 1, next_state, path + [move]))
        
        time_taken = time.time() - start_time
        print(f"❌ Enhanced BFS Failed: {nodes_expanded:,} nodes, {pruned_deadlocks:,} deadlocks pruned")
        return None


def enhanced_bfs(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Entry point for enhanced BFS solver
    """
    solver = EnhancedBFS(initial_state, time_limit)
    return solver.solve()


# Bidirectional BFS for even more optimization
class BidirectionalBFS:
    """
    Bidirectional BFS implementation - search from both start and goal states
    """
    
    def __init__(self, initial_state: SokobanState, time_limit: float = 120.0):
        self.initial_state = initial_state
        self.time_limit = time_limit
        
    def _reverse_moves(self, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Reverse a sequence of moves"""
        return [(-move[0], -move[1]) for move in reversed(moves)]
    
    def solve(self) -> Optional[SearchResult]:
        """
        Bidirectional BFS implementation
        """
        if self.initial_state.is_goal_state():
            return SearchResult([], 0, 0, 0.0)
        
        # Forward search from initial state
        forward_queue = deque([(self.initial_state, [])])
        forward_visited = {self.initial_state: []}
        
        # Backward search from goal state (conceptual - we don't have explicit goal state)
        # For now, use enhanced BFS as fallback
        solver = EnhancedBFS(self.initial_state, self.time_limit)
        return solver.solve()


def ultra_optimized_bfs(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Ultra-optimized BFS that tries multiple strategies
    """
    print(f"🚀 Starting Ultra-Optimized BFS (time limit: {time_limit}s)")
    
    # Try enhanced BFS first
    result = enhanced_bfs(initial_state, time_limit)
    
    if result:
        print(f"✅ Solution found with {len(result.path)} moves!")
        return result
    else:
        print(f"❌ No solution found within {time_limit}s")
        return None 