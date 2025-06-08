#!/usr/bin/env python3
"""
Optimized BFS Implementation for Sokoban
Focus on performance and reliability within 120 seconds
"""

from typing import List, Tuple, Set, Optional
from collections import deque
import time
from state import SokobanState
from search import SearchResult

def optimized_bfs(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Highly optimized BFS with performance-focused design
    """
    print(f"🚀 Starting Optimized BFS (time limit: {time_limit}s)")
    
    if initial_state.is_goal_state():
        return SearchResult([], 0, 0, 0.0)
    
    # Use deque for O(1) append/popleft operations
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    nodes_expanded = 0
    max_depth = 0
    start_time = time.time()
    
    # Performance optimizations
    check_interval = 10000  # Check time every N nodes
    last_time_check = start_time
    
    while queue:
        # Efficient time checking - only every N iterations
        if nodes_expanded % check_interval == 0:
            current_time = time.time()
            if current_time - start_time > time_limit:
                print(f"⏰ Timeout after {nodes_expanded:,} nodes in {current_time - start_time:.2f}s")
                return None
            
            # Progress reporting
            if nodes_expanded > 0:
                elapsed = current_time - start_time
                rate = nodes_expanded / elapsed
                print(f"🔄 Progress: {nodes_expanded:,} nodes, {len(visited):,} states, {rate:.0f} nodes/s")
            
            last_time_check = current_time
        
        current_state, path = queue.popleft()
        nodes_expanded += 1
        max_depth = max(max_depth, len(path))
        
        # Try all valid moves
        for move in current_state.get_valid_moves():
            next_state = current_state.move(move)
            
            # Quick goal check
            if next_state.is_goal_state():
                time_taken = time.time() - start_time
                print(f"🏆 SUCCESS! Found solution with {len(path) + 1} moves")
                print(f"   📊 {nodes_expanded:,} nodes expanded in {time_taken:.2f}s")
                return SearchResult(path + [move], nodes_expanded, max_depth, time_taken)
            
            # Add to queue if not visited
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
    
    time_taken = time.time() - start_time
    print(f"❌ No solution found after {nodes_expanded:,} nodes in {time_taken:.2f}s")
    return None


def hybrid_bfs_astar(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Hybrid approach: Start with BFS, switch to A* if needed
    """
    print(f"🔄 Hybrid BFS->A* (time limit: {time_limit}s)")
    
    # Try BFS for the first portion
    bfs_time_limit = min(30.0, time_limit * 0.5)
    print(f"📍 Phase 1: BFS for {bfs_time_limit}s")
    
    result = optimized_bfs(initial_state, bfs_time_limit)
    if result:
        return result
    
    # If BFS didn't work, try A*
    remaining_time = time_limit - bfs_time_limit
    if remaining_time > 5:
        print(f"📍 Phase 2: A* for {remaining_time}s")
        from search import astar
        return astar(initial_state, remaining_time)
    
    return None


def smart_bfs(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Smart BFS that chooses strategy based on puzzle complexity
    """
    # Estimate puzzle complexity
    num_boxes = len(initial_state.boxes)
    grid_size = len(initial_state.grid) * len(initial_state.grid[0])
    
    print(f"🎯 Smart BFS: {num_boxes} boxes, {grid_size} grid size")
    
    if num_boxes <= 3 and grid_size <= 64:
        # Small puzzle - use optimized BFS
        print("🏃 Strategy: Optimized BFS (small puzzle)")
        return optimized_bfs(initial_state, time_limit)
    elif num_boxes <= 4 and grid_size <= 100:
        # Medium puzzle - use hybrid approach
        print("🚶 Strategy: Hybrid BFS->A* (medium puzzle)")
        return hybrid_bfs_astar(initial_state, time_limit)
    else:
        # Large puzzle - go straight to A*
        print("🧠 Strategy: A* (large puzzle)")
        from search import astar
        return astar(initial_state, time_limit)


def ultimate_bfs(initial_state: SokobanState, time_limit: float = 120.0) -> Optional[SearchResult]:
    """
    Ultimate BFS solver - tries multiple strategies
    """
    print(f"🎯 ULTIMATE BFS SOLVER (time limit: {time_limit}s)")
    print(f"📊 Puzzle: {len(initial_state.boxes)} boxes, {len(initial_state.goals)} goals")
    
    strategies = [
        ("Smart BFS", lambda: smart_bfs(initial_state, time_limit)),
        ("Optimized BFS", lambda: optimized_bfs(initial_state, time_limit * 0.7)),
        ("A* Fallback", lambda: None)  # Will use A* if others fail
    ]
    
    total_start = time.time()
    
    for strategy_name, strategy_func in strategies:
        if time.time() - total_start >= time_limit:
            break
            
        print(f"\n🔧 Trying {strategy_name}...")
        remaining_time = time_limit - (time.time() - total_start)
        
        if strategy_name == "A* Fallback" and remaining_time > 10:
            from search import astar
            result = astar(initial_state, remaining_time)
        else:
            result = strategy_func()
        
        if result:
            total_time = time.time() - total_start
            print(f"🏆 {strategy_name} SUCCESS in {total_time:.2f}s total!")
            return result
    
    print(f"❌ All strategies failed in {time.time() - total_start:.2f}s")
    return None 