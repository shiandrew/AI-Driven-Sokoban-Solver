import sys
import os
from state import SokobanState
from search import bfs, dfs, astar, ids, evaluate_search_algorithm, visualize_solution

def parse_input(file_content: str) -> SokobanState:
    """
    Parse a Sokoban grid from a text file format.
    
    The function handles two formats:
    1. Direct map representation:
       Each line represents a row of the map using characters:
       '#' for walls, '$' for boxes, '.' for goals, '@' for player,
       '*' for box on goal, '+' for player on goal
    
    2. Coordinate-based format:
       Line 1: rows cols
       Line 2: number_of_walls x1 y1 x2 y2 ... (wall coordinates)
       Line 3: number_of_boxes x1 y1 x2 y2 ... (box coordinates)
       Line 4: number_of_goals x1 y1 x2 y2 ... (goal coordinates)
       Line 5: player_x player_y (player position)
    
    Args:
        file_content: String containing the grid description
        
    Returns:
        SokobanState representing the grid
    """
    lines = file_content.strip().split('\n')
    
    # Check if this is a direct map representation
    if any(c in lines[0] for c in SokobanState.WALL + SokobanState.BOX + SokobanState.GOAL + SokobanState.PLAYER):
        return SokobanState(lines)
    
    # Otherwise, parse coordinate-based format
    # Parse dimensions
    rows, cols = map(int, lines[0].split())
    
    # Initialize empty grid
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # Parse walls
    wall_data = lines[1].split()
    num_walls = int(wall_data[0])
    for i in range(1, len(wall_data), 2):
        x, y = int(wall_data[i]), int(wall_data[i+1])
        grid[y-1][x-1] = SokobanState.WALL
    
    # Parse boxes
    box_data = lines[2].split()
    num_boxes = int(box_data[0])
    for i in range(1, len(box_data), 2):
        x, y = int(box_data[i]), int(box_data[i+1])
        grid[y-1][x-1] = SokobanState.BOX
    
    # Parse goals
    goal_data = lines[3].split()
    num_goals = int(goal_data[0])
    for i in range(1, len(goal_data), 2):
        x, y = int(goal_data[i]), int(goal_data[i+1])
        if grid[y-1][x-1] == SokobanState.BOX:
            grid[y-1][x-1] = SokobanState.BOX_ON_GOAL
        else:
            grid[y-1][x-1] = SokobanState.GOAL
    
    # Parse player position
    player_x, player_y = map(int, lines[4].split())
    if grid[player_y-1][player_x-1] == SokobanState.GOAL:
        grid[player_y-1][player_x-1] = SokobanState.PLAYER_ON_GOAL
    else:
        grid[player_y-1][player_x-1] = SokobanState.PLAYER
    
    # Convert grid to list of strings
    grid_data = [''.join(row) for row in grid]
    return SokobanState(grid_data)

def load_grid_from_file(filename: str) -> SokobanState:
    """
    Load a Sokoban grid from a text file.
    
    Args:
        filename: Path to the file containing the grid
        
    Returns:
        SokobanState representing the grid
    """
    try:
        with open(filename, 'r') as file:
            file_content = file.read()
        return parse_input(file_content)
    except FileNotFoundError:
        raise FileNotFoundError(f"Map file '{filename}' not found")
    except Exception as e:
        raise ValueError(f"Error loading map from '{filename}': {e}")

def load_predefined_grid(grid_number: int) -> SokobanState:
    """
    Load a predefined Sokoban grid.
    
    Args:
        grid_number: The number of the grid to load
        
    Returns:
        SokobanState representing the grid
    """
    # Example grids (you can add more grids)
    predefined_grids = {
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
    
    if grid_number not in predefined_grids:
        raise ValueError(f"Grid {grid_number} not found")
        
    return SokobanState(predefined_grids[grid_number])

def print_state(state: SokobanState):
    """Print the current state of the game."""
    print("\n".join(state.grid))
    print()

def solve_and_visualize(initial_state: SokobanState, time_limit: float = 30.0, show_visualization: bool = True):
    """
    Solve the puzzle using multiple algorithms and optionally visualize the solution.
    
    Args:
        initial_state: The initial state of the puzzle
        time_limit: Maximum time in seconds to search
        show_visualization: Whether to show step-by-step visualization
    """
    print("Initial state:")
    print_state(initial_state)
    
    # Create algorithms list
    algorithms = [
        ("A*", lambda state: astar(state, time_limit=time_limit)),
        ("BFS", lambda state: bfs(state, time_limit=time_limit)),
        ("DFS", lambda state: dfs(state, time_limit=time_limit)),
        ("IDS", lambda state: ids(state, time_limit=time_limit))
    ]
    
    solutions = {}
    
    # Try each algorithm
    for name, algorithm in algorithms:
        print(f"Solving with {name}...")
        
        # All algorithms now return SearchResult objects
        result = algorithm(initial_state)
        if result:
            solutions[name] = {
                'moves': result.path,
                'length': len(result.path),
                'nodes_expanded': result.nodes_expanded,
                'max_depth': result.max_depth,
                'time_taken': result.time_taken,
                'success': True
            }
            print(f"✅ Solution found! Path length: {len(result.path)}")
            print(f"   Nodes expanded: {result.nodes_expanded}")
            print(f"   Max depth: {result.max_depth}")
            print(f"   Time taken: {result.time_taken:.2f} seconds")
        else:
            solutions[name] = {'success': False}
            print(f"❌ No solution found with {name} within {time_limit} seconds")
        print()
    
    # Find the best solution (shortest path among successful solutions)
    successful_solutions = {name: sol for name, sol in solutions.items() if sol['success']}
    
    if successful_solutions:
        best_algorithm = min(successful_solutions.keys(), 
                           key=lambda name: successful_solutions[name]['length'])
        best_solution = successful_solutions[best_algorithm]
        
        print(f"🏆 Best solution: {best_algorithm} with {best_solution['length']} moves")
        
        if show_visualization:
            print("\n" + "="*50)
            print(f"VISUALIZING SOLUTION ({best_algorithm})")
            print("="*50)
            visualize_solution(initial_state, best_solution['moves'])
        
        return best_solution['moves']
    else:
        print("❌ No solution found with any algorithm!")
        return None

def main():
    """Main function that handles command line arguments and runs the solver."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sokoban.py <map_file>                    # Load from file")
        print("  python sokoban.py <map_file> --no-viz           # Load from file without visualization")
        print("  python sokoban.py <map_file> --time-limit <sec> # Set custom time limit")
        print("  python sokoban.py --default                     # Use default grid")
        print()
        print("Examples:")
        print("  python sokoban.py grid1.txt")
        print("  python sokoban.py maps/puzzle.txt --no-viz")
        print("  python sokoban.py puzzle.txt --time-limit 30")
        print("  python sokoban.py --default --time-limit 5")
        print()
        print("Default time limit: 30 seconds")
        return
    
    # Parse arguments
    show_visualization = True
    time_limit = 30.0  # Default time limit in seconds
    
    # Handle --no-viz flag
    if "--no-viz" in sys.argv:
        show_visualization = False
        sys.argv.remove("--no-viz")
    
    # Handle --time-limit flag
    if "--time-limit" in sys.argv:
        time_limit_index = sys.argv.index("--time-limit")
        if time_limit_index + 1 < len(sys.argv):
            try:
                time_limit = float(sys.argv[time_limit_index + 1])
                if time_limit <= 0:
                    print("Error: Time limit must be positive")
                    return
                # Remove both the flag and the value
                sys.argv.pop(time_limit_index + 1)  # Remove the value first
                sys.argv.pop(time_limit_index)      # Then remove the flag
                print(f"Using custom time limit: {time_limit} seconds")
            except (ValueError, IndexError):
                print("Error: --time-limit requires a valid number")
                return
        else:
            print("Error: --time-limit requires a value")
            return
    
    if "--default" in sys.argv:
        # Use default grid
        print("Loading default grid...")
        initial_state = load_predefined_grid(1)
    else:
        # Load from file
        filename = sys.argv[1]
        
        # Normalize the path (handles both forward and backward slashes)
        filename = os.path.normpath(filename)
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            return
        
        print(f"Loading map from: {filename}")
        try:
            initial_state = load_grid_from_file(filename)
        except Exception as e:
            print(f"Error: {e}")
            return
    
    # Solve and optionally visualize
    solution = solve_and_visualize(initial_state, time_limit, show_visualization)
    
    if solution:
        print(f"\n🎉 Puzzle solved successfully in {len(solution)} moves!")
    else:
        print("\n😞 Could not solve the puzzle.")

if __name__ == "__main__":
    main()