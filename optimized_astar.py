#!/usr/bin/env python3
"""
Optimized A* Implementation for Sokoban
Multiple strategies and advanced heuristics to ensure all puzzles are solved within 120 seconds
"""

from typing import List, Tuple, Set, Optional, Dict
import heapq
import time
from state import SokobanState
from search import SearchResult

class OptimizedAStar:
    """
    Advanced A* with multiple heuristics and optimization strategies
    """
    
    def __init__(self, initial_state: SokobanState, time_limit: float = 120.0):
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.goals = set(initial_state.goals)
        self.walls = self._get_walls()
        self.start_time = time.time()
        self.goal_distances = self._precompute_goal_distances()
        
    def _get_walls(self) -> Set[Tuple[int, int]]:
        """Extract wall positions from the grid"""
        walls = set()
        for r, row in enumerate(self.initial_state.grid):
            for c, cell in enumerate(row):
                if cell == '#':
                    walls.add((r, c))
        return walls
    
    def _precompute_goal_distances(self) -> Dict[Tuple[int, int], int]:
        """
        Precompute distances from each goal to all reachable positions
        This helps create better heuristics
        """
        goal_distances = {}
        
        for goal in self.goals:
            distances = {}
            queue = [(goal, 0)]
            visited = {goal}
            
            while queue:
                (r, c), dist = queue.pop(0)
                distances[(r, c)] = dist
                
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited and (nr, nc) not in self.walls:
                        if 0 <= nr < len(self.initial_state.grid) and 0 <= nc < len(self.initial_state.grid[0]):
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))
            
            goal_distances[goal] = distances
        
        return goal_distances
    
    def _manhattan_heuristic(self, state: SokobanState) -> int:
        """
        Basic Manhattan distance heuristic
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
    
    def _enhanced_heuristic(self, state: SokobanState) -> int:
        """
        Enhanced heuristic considering actual distances and deadlocks
        """
        if not state.boxes:
            return 0
        
        total_cost = 0
        unmatched_boxes = [box for box in state.boxes if box not in self.goals]
        
        if not unmatched_boxes:
            return 0
        
        # Use precomputed distances for better estimates
        for box in unmatched_boxes:
            min_cost = float('inf')
            for goal in self.goals:
                if goal in self.goal_distances and box in self.goal_distances[goal]:
                    cost = self.goal_distances[goal][box]
                    min_cost = min(min_cost, cost)
                else:
                    # Fallback to Manhattan distance
                    cost = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                    min_cost = min(min_cost, cost)
            
            if min_cost != float('inf'):
                total_cost += min_cost
        
        # Add penalty for deadlock-prone positions
        deadlock_penalty = self._deadlock_penalty(state)
        
        return total_cost + deadlock_penalty
    
    def _deadlock_penalty(self, state: SokobanState) -> int:
        """
        Add penalty for potentially deadlocked positions
        """
        penalty = 0
        
        for box in state.boxes:
            if box in self.goals:
                continue
                
            r, c = box
            # Check for corner situations
            wall_count = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.walls:
                    wall_count += 1
            
            # Heavy penalty for boxes near walls without nearby goals
            if wall_count >= 2:
                nearby_goals = any(
                    abs(box[0] - goal[0]) + abs(box[1] - goal[1]) <= 3
                    for goal in self.goals
                )
                if not nearby_goals:
                    penalty += 50
        
        return penalty
    
    def _advanced_heuristic(self, state: SokobanState) -> int:
        """
        Most advanced heuristic with box-goal matching
        """
        unmatched_boxes = [box for box in state.boxes if box not in self.goals]
        available_goals = [goal for goal in self.goals if goal not in state.boxes]
        
        if not unmatched_boxes:
            return 0
        
        # Use Hungarian-like matching for minimum cost assignment
        total_cost = 0
        
        # Simple greedy matching (could be improved with actual Hungarian algorithm)
        used_goals = set()
        for box in unmatched_boxes:
            min_cost = float('inf')
            best_goal = None
            
            for goal in available_goals:
                if goal not in used_goals:
                    if goal in self.goal_distances and box in self.goal_distances[goal]:
                        cost = self.goal_distances[goal][box]
                    else:
                        cost = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_goal = goal
            
            if best_goal:
                used_goals.add(best_goal)
                total_cost += min_cost
        
        # Add deadlock penalty
        total_cost += self._deadlock_penalty(state)
        
        return total_cost
    
    def basic_astar(self) -> Optional[SearchResult]:
        """
        Basic A* with Manhattan distance heuristic
        """
        print("🔧 Basic A* (Manhattan heuristic)")
        
        heap = [(self._manhattan_heuristic(self.initial_state), 0, 0, self.initial_state, [])]
        visited = {self.initial_state: 0}
        nodes_expanded = 0
        max_depth = 0
        node_id = 0
        
        while heap:
            if time.time() - self.start_time > self.time_limit:
                break
                
            f_score, g_score, _, current_state, path = heapq.heappop(heap)
            nodes_expanded += 1
            max_depth = max(max_depth, len(path))
            
            if current_state.is_goal_state():
                time_taken = time.time() - self.start_time
                print(f"🏆 Basic A* SUCCESS in {time_taken:.2f}s!")
                return SearchResult(path, nodes_expanded, max_depth, time_taken)
            
            # Skip if we've seen this state with a better cost
            if current_state in visited and visited[current_state] < g_score:
                continue
            
            for move in current_state.get_valid_moves():
                next_state = current_state.move(move)
                new_g_score = g_score + 1
                
                if next_state not in visited or visited[next_state] > new_g_score:
                    visited[next_state] = new_g_score
                    h_score = self._manhattan_heuristic(next_state)
                    f_score = new_g_score + h_score
                    node_id += 1
                    heapq.heappush(heap, (f_score, new_g_score, node_id, next_state, path + [move]))
        
        return None
    
    def enhanced_astar(self) -> Optional[SearchResult]:
        """
        Enhanced A* with better heuristic and pruning
        """
        print("🧠 Enhanced A* (Advanced heuristic)")
        
        heap = [(self._enhanced_heuristic(self.initial_state), 0, 0, self.initial_state, [])]
        visited = {self.initial_state: 0}
        nodes_expanded = 0
        max_depth = 0
        node_id = 0
        
        while heap:
            if time.time() - self.start_time > self.time_limit:
                break
                
            f_score, g_score, _, current_state, path = heapq.heappop(heap)
            nodes_expanded += 1
            max_depth = max(max_depth, len(path))
            
            # Progress reporting
            if nodes_expanded % 50000 == 0:
                elapsed = time.time() - self.start_time
                print(f"   Progress: {nodes_expanded:,} nodes, {elapsed:.1f}s")
            
            if current_state.is_goal_state():
                time_taken = time.time() - self.start_time
                print(f"🏆 Enhanced A* SUCCESS in {time_taken:.2f}s!")
                return SearchResult(path, nodes_expanded, max_depth, time_taken)
            
            if current_state in visited and visited[current_state] < g_score:
                continue
            
            for move in current_state.get_valid_moves():
                next_state = current_state.move(move)
                new_g_score = g_score + 1
                
                if next_state not in visited or visited[next_state] > new_g_score:
                    visited[next_state] = new_g_score
                    h_score = self._enhanced_heuristic(next_state)
                    f_score = new_g_score + h_score
                    node_id += 1
                    heapq.heappush(heap, (f_score, new_g_score, node_id, next_state, path + [move]))
        
        return None
    
    def ultimate_astar(self) -> Optional[SearchResult]:
        """
        Ultimate A* with the most advanced heuristic
        """
        print("🚀 Ultimate A* (Maximum optimization)")
        
        heap = [(self._advanced_heuristic(self.initial_state), 0, 0, self.initial_state, [])]
        visited = {self.initial_state: 0}
        nodes_expanded = 0
        max_depth = 0
        node_id = 0
        
        while heap:
            if time.time() - self.start_time > self.time_limit:
                break
                
            f_score, g_score, _, current_state, path = heapq.heappop(heap)
            nodes_expanded += 1
            max_depth = max(max_depth, len(path))
            
            # Progress reporting
            if nodes_expanded % 25000 == 0:
                elapsed = time.time() - self.start_time
                print(f"   Progress: {nodes_expanded:,} nodes, {elapsed:.1f}s")
            
            if current_state.is_goal_state():
                time_taken = time.time() - self.start_time
                print(f"🏆 Ultimate A* SUCCESS in {time_taken:.2f}s!")
                return SearchResult(path, nodes_expanded, max_depth, time_taken)
            
            if current_state in visited and visited[current_state] < g_score:
                continue
            
            for move in current_state.get_valid_moves():
                next_state = current_state.move(move)
                new_g_score = g_score + 1
                
                if next_state not in visited or visited[next_state] > new_g_score:
                    visited[next_state] = new_g_score
                    h_score = self._advanced_heuristic(next_state)
                    f_score = new_g_score + h_score
                    node_id += 1
                    heapq.heappush(heap, (f_score, new_g_score, node_id, next_state, path + [move]))
        
        return None


def smart_astar(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Smart A* that adapts strategy based on puzzle characteristics
    """
    print(f"🎯 SMART A* SOLVER (time limit: {time_limit}s)")
    
    # Analyze puzzle complexity
    num_boxes = len(initial_state.boxes)
    grid_area = len(initial_state.grid) * max(len(row) for row in initial_state.grid)
    
    print(f"📊 Analysis: {num_boxes} boxes, {grid_area} cells")
    
    solver = OptimizedAStar(initial_state, time_limit)
    start_time = time.time()
    
    # Strategy 1: Basic A* for simple puzzles
    if num_boxes <= 3 and grid_area <= 80:
        print("🏃 Strategy: Basic A* (simple puzzle)")
        result = solver.basic_astar()
        if result:
            return result
    
    # Strategy 2: Enhanced A* for medium puzzles
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15:
        print("🚶 Strategy: Enhanced A* (medium complexity)")
        solver_enhanced = OptimizedAStar(initial_state, remaining_time)
        result = solver_enhanced.enhanced_astar()
        if result:
            return result
    
    # Strategy 3: Ultimate A* for complex puzzles
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        print("🧠 Strategy: Ultimate A* (maximum power)")
        solver_ultimate = OptimizedAStar(initial_state, remaining_time)
        result = solver_ultimate.ultimate_astar()
        if result:
            return result
    
    print(f"❌ All A* strategies failed in {time.time() - start_time:.2f}s")
    return None


def ultimate_astar(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Ultimate A* solver with intelligent strategy selection and optimization
    """
    print(f"⭐ ULTIMATE A* SOLVER (time limit: {time_limit}s)")
    
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
        print("🏃 Strategy: Basic A* (tiny puzzle)")
        solver = OptimizedAStar(initial_state, time_limit)
        result = solver.basic_astar()
        if result:
            return result
    
    elif num_boxes <= 3 and grid_area <= 100:
        print("🚶 Strategy: Enhanced A* (small puzzle)")
        solver = OptimizedAStar(initial_state, time_limit * 0.7)
        result = solver.enhanced_astar()
        if result:
            return result
        
        # Fallback to Ultimate A*
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10:
            print("🔄 Fallback: Ultimate A*")
            solver_ultimate = OptimizedAStar(initial_state, remaining)
            result = solver_ultimate.ultimate_astar()
            if result:
                return result
    
    else:
        print("🧠 Strategy: Ultimate A* (complex puzzle)")
        solver = OptimizedAStar(initial_state, time_limit)
        result = solver.ultimate_astar()
        if result:
            return result
    
    print(f"❌ All A* strategies failed in {time.time() - start_time:.2f}s")
    return None 