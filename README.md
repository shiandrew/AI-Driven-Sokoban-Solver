# AI-Driven Sokoban Solver

This project implements a comprehensive Sokoban puzzle solver using various search algorithms. Sokoban is a classic puzzle game where the player must push boxes to designated storage locations.

## Project Structure

- `sokoban.py`: Main application with command-line interface
- `state.py`: Game state representation and mechanics
- `search.py`: Implementation of search algorithms (BFS, DFS, A*, IDS)
- `tests/`: Test cases and evaluation framework
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules

## Features

### Game Engine
- Robust game state representation with irregular grid support
- Complete game mechanics (movement, box pushing, goal detection)
- Support for multiple input formats (ASCII maps and coordinate-based)

### Search Algorithms
- **BFS (Breadth-First Search)**: Guarantees optimal solution, high memory usage
- **DFS (Depth-First Search)**: Fast but finds suboptimal solutions
- **A* (A-Star)**: Optimal with Manhattan distance heuristic, efficient
- **IDS (Iterative Deepening Search)**: Optimal with low memory usage

### Advanced Features
- Configurable time limits (default: 30 seconds)
- Performance metrics (nodes expanded, search depth, execution time)
- Solution visualization with step-by-step playback
- Command-line interface with multiple options
- Automatic algorithm comparison and best solution selection

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Solve a puzzle file with default settings (30-second time limit)
python sokoban.py puzzle.txt

# Solve without showing step-by-step visualization
python sokoban.py puzzle.txt --no-viz

# Use a custom time limit
python sokoban.py puzzle.txt --time-limit 60

# Test with the built-in default puzzle
python sokoban.py --default
```

### Advanced Usage
```bash
# Quick 5-second test
python sokoban.py puzzle.txt --time-limit 5 --no-viz

# Longer search for difficult puzzles
python sokoban.py hard-puzzle.txt --time-limit 120

# Combine options
python sokoban.py puzzle.txt --time-limit 45 --no-viz
```

### Command-Line Options
- `<map_file>`: Path to puzzle file
- `--no-viz`: Disable step-by-step solution visualization
- `--time-limit <seconds>`: Set custom time limit (default: 30 seconds)
- `--default`: Use built-in test puzzle

## Input Formats

### ASCII Map Format
```
########
#  .# .#
# $   @#
# $## #
#     .#
########
```

### Coordinate-Based Format
```
6 8
16 1 1 1 2 1 3 1 4 1 5 1 6 1 7 1 8 2 1 2 8 3 1 3 8 4 1 4 8 5 1 5 8 6 1 6 2 6 3 6 4 6 5 6 6 6 7 6 8
3 3 2 4 2 4 3
3 3 4 6 4 7 4
6 7
```

## Game Symbols
- `#`: Wall
- `$`: Box
- `.`: Goal/Storage location
- `@`: Player
- `*`: Box on goal
- `+`: Player on goal
- ` `: Empty floor

## Example Output
```
Loading map from: puzzle.txt
Initial state:
########
#  .# .#
# $   @#
# $$## #
#     .#
########

Solving with A*...
✅ Solution found! Path length: 37
   Nodes expanded: 17472
   Max depth: 37
   Time taken: 1.11 seconds

Solving with BFS...
✅ Solution found! Path length: 37
   Nodes expanded: 15751
   Max depth: 36
   Time taken: 0.96 seconds

Solving with DFS...
✅ Solution found! Path length: 47
   Nodes expanded: 1529
   Max depth: 100
   Time taken: 0.10 seconds

Solving with IDS...
✅ Solution found! Path length: 37
   Nodes expanded: 23891
   Max depth: 37
   Time taken: 2.34 seconds

🏆 Best solution: BFS with 37 moves

🎉 Puzzle solved successfully in 37 moves!
```

## Game Rules

- The player can move in four directions (up, down, left, right)
- Boxes can only be pushed, not pulled
- Only one box can be moved at a time
- The player cannot walk through walls or boxes
- The goal is to push all boxes to designated storage locations
- The puzzle is solved when all boxes are on goal positions

## Technical Details

### State Representation
- Efficient hash-based state comparison for duplicate detection
- Compact representation using player position and box positions
- Support for irregular grid shapes with automatic normalization

### Search Implementation
- Time-bounded search with configurable limits
- Memory-efficient algorithms with visited state tracking
- Comprehensive performance metrics collection
- Robust error handling and edge case management

### Heuristics
- Manhattan distance heuristic for A* search
- Sum of minimum distances from each box to nearest goal
- Admissible heuristic ensuring optimal solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request