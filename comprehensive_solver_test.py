from deadlock import quick_deadlock_check
from sokoban import load_grid_from_file
from search import bfs, dfs, astar, ids
import time
import sys

def comprehensive_puzzle_test(puzzle_file, time_limit=60):
    """
    Comprehensive puzzle test with deadlock detection pre-filtering
    
    Workflow:
    1. Load puzzle and run deadlock detection
    2. If deadlock detected -> Stop immediately and report
    3. If solvable -> Run all four algorithms with time limit
    4. Report best solution and performance comparison
    """
    
    print(f"🎯 COMPREHENSIVE SOKOBAN SOLVER TEST")
    print(f"{'='*60}")
    print(f"📂 Loading map from: {puzzle_file}")
    
    try:
        # Load puzzle
        state = load_grid_from_file(puzzle_file)
        
        # Display initial state
        print(f"\n📋 Initial state:")
        for row in state.grid:
            print(f"  {row}")
        
        print(f"\n📊 Puzzle Info:")
        print(f"  Grid: {state.height}x{state.width}")
        print(f"  Boxes: {len(state.boxes)}")
        print(f"  Goals: {len(state.goals)}")
        print(f"  Player: {state.player_pos}")
        
        # STEP 1: DEADLOCK DETECTION (Pre-filter)
        print(f"\n🔍 STEP 1: DEADLOCK DETECTION")
        print("-" * 40)
        
        deadlock_start = time.time()
        is_solvable, deadlock_reasons = quick_deadlock_check(state)
        deadlock_time = time.time() - deadlock_start
        
        print(f"Detection time: {deadlock_time:.4f}s")
        print(f"Result: {'✅ SOLVABLE' if is_solvable else '❌ UNSOLVABLE'}")
        
        if not is_solvable:
            print(f"\n🚫 DEADLOCKS DETECTED:")
            for i, reason in enumerate(deadlock_reasons, 1):
                print(f"  {i}. {reason}")
            
            print(f"\n⚠️  STOPPING: Puzzle is unsolvable - no need to run search algorithms")
            print(f"💡 Time saved: Up to {time_limit * 4}s (4 algorithms × {time_limit}s each)")
            return False
        
        # STEP 2: SEARCH ALGORITHMS (Only if solvable)
        print(f"\n🤖 STEP 2: SEARCH ALGORITHMS")
        print(f"Running all algorithms with {time_limit}s time limit each...")
        print("-" * 40)
        
        algorithms = [
            ("A*", astar),
            ("BFS", bfs), 
            ("DFS", dfs),
            ("IDS", ids)
        ]
        
        results = {}
        total_search_time = 0
        
        for name, algorithm in algorithms:
            print(f"\n🔍 Solving with {name}...")
            
            try:
                start_time = time.time()
                result = algorithm(state, time_limit=time_limit)
                solve_time = time.time() - start_time
                total_search_time += solve_time
                
                if result:
                    path_length = len(result.path)
                    nodes_expanded = result.nodes_expanded
                    max_depth = result.max_depth
                    
                    print(f"  ✅ Solution found! Path length: {path_length}")
                    print(f"     Nodes expanded: {nodes_expanded}")
                    print(f"     Max depth: {max_depth}")
                    print(f"     Time taken: {solve_time:.2f} seconds")
                    
                    results[name] = {
                        'solved': True,
                        'moves': path_length,
                        'nodes': nodes_expanded,
                        'depth': max_depth,
                        'time': solve_time
                    }
                else:
                    print(f"  ❌ No solution found (timeout after {solve_time:.2f}s)")
                    results[name] = {
                        'solved': False,
                        'time': solve_time
                    }
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[name] = {'error': str(e)}
        
        # STEP 3: ANALYSIS AND SUMMARY
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        solved_algorithms = [name for name, res in results.items() if res.get('solved', False)]
        
        if solved_algorithms:
            print(f"✅ Solved by: {', '.join(solved_algorithms)}")
            
            # Find best solution (fewest moves)
            best_algo = min(solved_algorithms, key=lambda name: results[name]['moves'])
            best_result = results[best_algo]
            
            print(f"\n🏆 Best solution: {best_algo} with {best_result['moves']} moves")
            
            # Detailed comparison table
            print(f"\n📈 Algorithm Comparison:")
            print(f"{'Algorithm':<8} {'Status':<10} {'Moves':<6} {'Nodes':<8} {'Time':<8} {'Efficiency'}")
            print("-" * 65)
            
            for name in ["A*", "BFS", "DFS", "IDS"]:
                if name in results:
                    res = results[name]
                    if res.get('solved'):
                        efficiency = res['nodes'] / res['time'] if res['time'] > 0 else 0
                        print(f"{name:<8} {'✅ Solved':<10} {res['moves']:<6} {res['nodes']:<8} {res['time']:<8.2f} {efficiency:.0f} n/s")
                    else:
                        print(f"{name:<8} {'❌ Failed':<10} {'-':<6} {'-':<8} {res['time']:<8.2f} -")
            
            print(f"\n🎉 Puzzle solved successfully in {best_result['moves']} moves!")
            
        else:
            print(f"❌ No algorithm found a solution within {time_limit}s time limit")
            print(f"💡 This puzzle may require longer search time or advanced techniques")
        
        # Time analysis
        print(f"\n⏱️  TIME ANALYSIS:")
        print(f"  Deadlock detection: {deadlock_time:.4f}s")
        print(f"  Search algorithms: {total_search_time:.2f}s") 
        print(f"  Total time: {deadlock_time + total_search_time:.2f}s")
        print(f"  Time saved by deadlock detection: ~0s (puzzle was solvable)")
        
        return len(solved_algorithms) > 0
        
    except Exception as e:
        print(f"❌ Error loading puzzle: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_puzzles():
    """Test multiple puzzles to demonstrate the system"""
    puzzles = [
        'puzzles/input-01.txt',
        'puzzles/input-05a.txt', 
        'puzzles/input-05b.txt',  # This should be caught by deadlock detection
        'puzzles/input38.txt',
        'puzzles/input46.txt'
    ]
    
    print("🎯 BATCH TESTING MULTIPLE PUZZLES")
    print("=" * 60)
    
    results_summary = {}
    
    for puzzle in puzzles:
        try:
            print(f"\n\n" + "🔸" * 60)
            success = comprehensive_puzzle_test(puzzle, time_limit=30)  # Shorter for batch testing
            results_summary[puzzle] = success
        except Exception as e:
            print(f"Failed to test {puzzle}: {e}")
            results_summary[puzzle] = False
    
    # Final summary
    print(f"\n\n🏆 BATCH TESTING SUMMARY")
    print("=" * 60)
    
    for puzzle, success in results_summary.items():
        status = "✅ SUCCESS" if success else "❌ FAILED/UNSOLVABLE"
        print(f"{puzzle}: {status}")
    
    solved_count = sum(1 for success in results_summary.values() if success)
    total_count = len(results_summary)
    print(f"\nOverall: {solved_count}/{total_count} puzzles solved ({solved_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific puzzle
        puzzle_file = sys.argv[1]
        time_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        comprehensive_puzzle_test(puzzle_file, time_limit)
    else:
        # Test multiple puzzles
        test_multiple_puzzles() 