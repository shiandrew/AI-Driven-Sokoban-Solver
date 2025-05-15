# Sokoban Solver

This project implements a Sokoban puzzle solver using various search algorithms. Sokoban is a classic puzzle game where the player must push boxes to designated storage locations.

## Project Structure

- `sokoban.py`: Main game engine implementation
- `state.py`: State representation and game state management
- `search.py`: Implementation of search algorithms (BFS, DFS)
- `tests/`: Test cases and evaluation framework

## Features

- Game state representation
- Basic game mechanics (movement, box pushing)
- Uninformed search algorithms (BFS, DFS)
- Evaluation framework for algorithm performance

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

Run the main game:
```bash
python sokoban.py
```

Run tests:
```bash
pytest tests/
```

## Game Rules

- The player can move in four directions (up, down, left, right)
- Boxes can only be pushed, not pulled
- Only one box can be moved at a time
- The player cannot walk through walls or boxes
- The goal is to push all boxes to designated storage locations 