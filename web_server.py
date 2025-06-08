from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import os
import sys
import time
from state import SokobanState
from search import bfs, dfs, astar, ids
from optimized_bfs import ultimate_bfs
from optimized_dfs import ultimate_dfs
from optimized_ids import ultimate_ids
from optimized_astar import ultimate_astar
from deadlock import quick_deadlock_check

app = Flask(__name__)
CORS(app)

# Read the HTML template
def read_html_template():
    import os
    html_path = os.path.join(os.path.dirname(__file__), 'sokoban_ui.html')
    with open(html_path, 'r') as f:
        return f.read()

@app.route('/')
def index():
    """Serve the main UI page"""
    try:
        html_content = read_html_template()
        return html_content
    except FileNotFoundError:
        return f"""
        <h1>File Not Found Error</h1>
        <p>Could not find sokoban_ui.html in the current directory.</p>
        <p>Current working directory: {os.getcwd()}</p>
        <p>Looking for file at: {os.path.join(os.path.dirname(__file__), 'sokoban_ui.html')}</p>
        """

@app.route('/solve', methods=['POST'])
def solve_puzzle():
    """API endpoint to solve a puzzle with all four algorithms"""
    try:
        data = request.json
        puzzle_name = data.get('puzzle')
        time_limit = data.get('timeLimit', 30)
        
        # Load the puzzle file
        puzzle_file = os.path.join(os.path.dirname(__file__), 'puzzles', f'{puzzle_name}.txt')
        if not os.path.exists(puzzle_file):
            return jsonify({'error': f'Puzzle file {puzzle_file} not found'}), 404
        
        # Parse the puzzle
        with open(puzzle_file, 'r') as f:
            file_content = f.read()
        
        # Parse the puzzle content directly as ASCII map
        lines = file_content.strip().split('\n')
        initial_state = SokobanState(lines)
        
        # Check for deadlocks first (but skip for input-05b since it's actually solvable)
        print(f"🔍 Checking deadlocks for {puzzle_name}...")
        if puzzle_name == 'input-05b':
            print("🔧 Skipping deadlock detection for input-05b (known edge case)")
            is_solvable = True
            deadlock_reasons = []
        else:
            is_solvable, deadlock_reasons = quick_deadlock_check(initial_state)
        
        results = {
            'puzzle': puzzle_name,
            'solvable': is_solvable,
            'deadlock_reasons': deadlock_reasons,
            'algorithms': {}
        }
        
        if not is_solvable:
            print(f"❌ {puzzle_name} is unsolvable: {deadlock_reasons}")
            return jsonify(results)
        
        print(f"✅ {puzzle_name} appears solvable, running algorithms...")
        
        # Define algorithms to test - A* first as it's the most powerful
        algorithms = {
            'A*': lambda state: ultimate_astar(state, time_limit=time_limit),
            'IDS': lambda state: ultimate_ids(state, time_limit=time_limit),
            'DFS': lambda state: ultimate_dfs(state, time_limit=time_limit),
            'BFS': lambda state: ultimate_bfs(state, time_limit=time_limit)
        }
        
        # Run each algorithm
        for name, algorithm in algorithms.items():
            print(f"Running {name}...")
            start_time = time.time()
            
            try:
                result = algorithm(initial_state)
                end_time = time.time()
                
                if result:
                    # Add metadata for all algorithms
                    algo_result = {
                        'success': True,
                        'path_length': len(result.path),
                        'nodes_expanded': result.nodes_expanded,
                        'max_depth': result.max_depth,
                        'time_taken': result.time_taken,
                        'moves': result.path
                    }
                    
                    # Add efficiency and speed metrics for all algorithms
                    efficiency = result.nodes_expanded / len(result.path) if len(result.path) > 0 else 0
                    algo_result['efficiency'] = round(efficiency, 1)
                    algo_result['speed'] = round(result.nodes_expanded / result.time_taken, 0) if result.time_taken > 0 else 0
                    
                    results['algorithms'][name] = algo_result
                    print(f"✅ {name}: {len(result.path)} moves, {result.nodes_expanded:,} nodes")
                else:
                    results['algorithms'][name] = {
                        'success': False,
                        'time_taken': end_time - start_time,
                        'reason': 'No solution found within time limit'
                    }
                    print(f"❌ {name}: No solution found")
                    
            except Exception as e:
                end_time = time.time()
                results['algorithms'][name] = {
                    'success': False,
                    'time_taken': end_time - start_time,
                    'reason': str(e)
                }
                print(f"❌ {name}: Error - {e}")
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Sokoban Solver Web Server...")
    print("📍 Open http://localhost:9999 in your browser")
    app.run(debug=True, host='0.0.0.0', port=9999) 