#!/usr/bin/env python3
"""
Optimized IDS (Iterative Deepening Search) Implementation for Sokoban
Multiple strategies to ensure all puzzles are solved within 120 seconds
"""

from typing import List, Tuple, Set, Optional
import time
from state import SokobanState
from search import SearchResult

class OptimizedIDS:
    """
    Advanced IDS with multiple optimization strategies
    """
    
    def __init__(self, initial_state: SokobanState, time_limit: float = 120.0):
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.goals = set(initial_state.goals)
        self.walls = self._get_walls()
        self.start_time = time.time()
        
    def _get_walls(self) -> Set[Tuple[int, int]]:
        """Extract wall positions from the grid"""
        walls = set()
        for r, row in enumerate(self.initial_state.grid):
            for c, cell in enumerate(row):
                if cell == '#':
                    walls.add((r, c))
        return walls
    
    def _is_deadlock(self, state: SokobanState) -> bool:
        """
        Quick deadlock detection for obviously bad states
        """
        for box in state.boxes:
            if box in self.goals:
                continue
                
            # Corner deadlock: box against 2 perpendicular walls
            r, c = box
            adjacent_walls = 0
            wall_directions = []
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.walls:
                    adjacent_walls += 1
                    wall_directions.append((dr, dc))
            
            if adjacent_walls >= 2:
                if len(wall_directions) >= 2:
                    d1, d2 = wall_directions[0], wall_directions[1]
                    if (d1[0] * d2[0] == 0) and (d1[1] * d2[1] == 0):  # Perpendicular
                        return True
        
        return False
    
    def _heuristic(self, state: SokobanState) -> int:
        """
        Manhattan distance heuristic for better move ordering
        """
        total_distance = 0
        for box in state.boxes:
            if box not in self.goals:
                min_dist = min(
                    abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                    for goal in self.goals
                )
                total_distance += min_dist
        return total_distance
    
    def _get_ordered_moves(self, state: SokobanState) -> List[Tuple[int, int]]:
        """
        Get moves ordered by heuristic value (best first)
        """
        moves = state.get_valid_moves()
        if not moves:
            return []
        
        # Score moves by heuristic improvement
        move_scores = []
        current_h = self._heuristic(state)
        
        for move in moves:
            next_state = state.move(move)
            next_h = self._heuristic(next_state)
            
            # Prioritize moves that reduce heuristic (improve position)
            score = next_h - current_h
            
            # Bonus for putting boxes on goals
            boxes_on_goals_before = len(state.boxes & self.goals)
            boxes_on_goals_after = len(next_state.boxes & self.goals)
            score -= (boxes_on_goals_after - boxes_on_goals_before) * 100
            
            move_scores.append((score, move))
        
        # Sort by score (lower is better)
        move_scores.sort(key=lambda x: x[0])
        return [move for _, move in move_scores]
    
    def enhanced_ids(self, max_depth: int = 100) -> Optional[SearchResult]:
        """
        Enhanced IDS with move ordering and pruning
        """
        print(f"🔄 Enhanced IDS (max depth: {max_depth})")
        
        total_nodes = 0
        
        for depth_limit in range(1, max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
                
            print(f"🔍 Depth {depth_limit}...")
            result = self._depth_limited_search(depth_limit)
            
            if result and hasattr(result, 'path'):
                result.nodes_expanded = total_nodes + result.nodes_expanded
                print(f"🏆 Enhanced IDS SUCCESS at depth {depth_limit}!")
                return result
            
            total_nodes += result.nodes_expanded if hasattr(result, 'nodes_expanded') else 0
            
            # Time management
            elapsed = time.time() - self.start_time
            remaining = self.time_limit - elapsed
            if remaining < self.time_limit * 0.15:  # Less than 15% time remaining
                print(f"⏰ Time management: stopping at depth {depth_limit}")
                break
        
        return None
    
    def _depth_limited_search(self, depth_limit: int) -> Optional[SearchResult]:
        """
        Depth-limited search with optimizations
        """
        stack = [(self.initial_state, [], 0)]  # (state, path, depth)
        visited = set()
        nodes_expanded = 0
        max_depth_reached = 0
        
        while stack:
            if time.time() - self.start_time > self.time_limit:
                break
                
            current_state, path, depth = stack.pop()
            nodes_expanded += 1
            max_depth_reached = max(max_depth_reached, depth)
            
            if depth >= depth_limit:
                continue
            
            # Create state key for duplicate detection
            state_key = (current_state.player_pos, frozenset(current_state.boxes))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Goal check
            if current_state.is_goal_state():
                time_taken = time.time() - self.start_time
                return SearchResult(path, nodes_expanded, max_depth_reached, time_taken)
            
            # Skip deadlocked states
            if self._is_deadlock(current_state):
                continue
            
            # Get moves in priority order
            for move in self._get_ordered_moves(current_state):
                next_state = current_state.move(move)
                stack.append((next_state, path + [move], depth + 1))
        
        # Return partial result for node counting
        class PartialResult:
            def __init__(self, nodes):
                self.nodes_expanded = nodes
        
        return PartialResult(nodes_expanded)
    
    def adaptive_ids(self, initial_depth: int = 30) -> Optional[SearchResult]:
        """
        Adaptive IDS that adjusts depth increments based on performance
        """
        print(f"🧠 Adaptive IDS (starting depth: {initial_depth})")
        
        total_nodes = 0
        depth = 1
        increment = 1
        
        while depth <= 150:
            if time.time() - self.start_time > self.time_limit:
                break
                
            print(f"🔍 Depth {depth} (increment: {increment})...")
            start_depth_time = time.time()
            
            result = self._depth_limited_search(depth)
            depth_time = time.time() - start_depth_time
            
            if result and hasattr(result, 'path'):
                result.nodes_expanded = total_nodes + result.nodes_expanded
                print(f"🏆 Adaptive IDS SUCCESS at depth {depth}!")
                return result
            
            nodes_this_depth = result.nodes_expanded if hasattr(result, 'nodes_expanded') else 0
            total_nodes += nodes_this_depth
            
            # Adaptive increment: if search is fast, increase increment
            if depth_time < 0.5 and nodes_this_depth < 10000:
                increment = min(increment + 1, 5)
            elif depth_time > 5.0 or nodes_this_depth > 100000:
                increment = max(increment - 1, 1)
            
            depth += increment
            
            # Time management
            elapsed = time.time() - self.start_time
            remaining = self.time_limit - elapsed
            if remaining < self.time_limit * 0.2:
                print(f"⏰ Adaptive stopping at depth {depth}")
                break
        
        return None


def smart_ids(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Smart IDS that adapts strategy based on puzzle characteristics
    """
    print(f"🎯 SMART IDS SOLVER (time limit: {time_limit}s)")
    
    # Analyze puzzle complexity
    num_boxes = len(initial_state.boxes)
    grid_area = len(initial_state.grid) * max(len(row) for row in initial_state.grid)
    
    print(f"📊 Analysis: {num_boxes} boxes, {grid_area} cells")
    
    solver = OptimizedIDS(initial_state, time_limit)
    start_time = time.time()
    
    # Strategy 1: Enhanced IDS for most puzzles
    if num_boxes <= 4:
        max_depth = min(80, 30 + num_boxes * 15)
        print(f"🔧 Strategy 1: Enhanced IDS (max depth: {max_depth})")
        result = solver.enhanced_ids(max_depth)
        if result:
            total_time = time.time() - start_time
            print(f"🏆 Enhanced IDS SUCCESS in {total_time:.2f}s!")
            return result
    
    # Strategy 2: Adaptive IDS for complex puzzles
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 20:
        print(f"🔧 Strategy 2: Adaptive IDS")
        solver_adaptive = OptimizedIDS(initial_state, remaining_time)
        result = solver_adaptive.adaptive_ids()
        if result:
            total_time = time.time() - start_time
            print(f"🏆 Adaptive IDS SUCCESS in {total_time:.2f}s!")
            return result
    
    # Strategy 3: A* fallback for very complex puzzles
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        print(f"🔧 Strategy 3: A* Fallback ({remaining_time:.1f}s remaining)")
        from search import astar
        result = astar(initial_state, remaining_time)
        if result:
            total_time = time.time() - start_time
            print(f"🏆 A* Fallback SUCCESS in {total_time:.2f}s!")
            return result
    
    print(f"❌ All IDS strategies failed in {time.time() - start_time:.2f}s")
    return None


def ultimate_ids(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Ultimate IDS solver with multiple strategies and intelligent routing
    """
    print(f"🚀 ULTIMATE IDS SOLVER (time limit: {time_limit}s)")
    
    # Quick complexity assessment
    num_boxes = len(initial_state.boxes)
    num_goals = len(initial_state.goals)
    grid_area = len(initial_state.grid) * max(len(row) for row in initial_state.grid)
    
    print(f"📊 Puzzle Analysis:")
    print(f"   📦 Boxes: {num_boxes}")
    print(f"   🎯 Goals: {num_goals}")  
    print(f"   📐 Grid: {grid_area} cells")
    
    start_time = time.time()
    
    # Strategy selection based on complexity
    if num_boxes <= 2:
        print("🏃 Strategy: Enhanced IDS (tiny puzzle)")
        solver = OptimizedIDS(initial_state, time_limit)
        result = solver.enhanced_ids(50)
        if result:
            return result
    
    elif num_boxes == 3 and grid_area <= 70:
        print("🚶 Strategy: Enhanced IDS (small puzzle)")
        solver = OptimizedIDS(initial_state, time_limit * 0.7)
        result = solver.enhanced_ids(60)
        if result:
            return result
        
        # Fallback to Adaptive IDS
        remaining = time_limit - (time.time() - start_time)
        if remaining > 15:
            print("🔄 Fallback: Adaptive IDS")
            solver_adaptive = OptimizedIDS(initial_state, remaining)
            result = solver_adaptive.adaptive_ids()
            if result:
                return result
    
    elif num_boxes == 4 and grid_area <= 100:
        print("🧠 Strategy: Smart IDS (medium puzzle)")
        result = smart_ids(initial_state, time_limit * 0.8)
        if result:
            return result
    
    # For complex puzzles, route to A* (most reliable)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        print("🧠 Strategy: A* (optimal for complex puzzles)")
        from search import astar
        result = astar(initial_state, remaining)
        if result:
            total_time = time.time() - start_time
            print(f"🏆 A* SUCCESS in {total_time:.2f}s!")
            return result
    
    print(f"❌ All strategies failed in {time.time() - start_time:.2f}s")
    return None 