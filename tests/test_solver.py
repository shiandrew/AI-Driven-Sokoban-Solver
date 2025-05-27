import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import SokobanState
from solver import SokobanSolver, visualize_solution

def test_simple_puzzle():
    # Simple puzzle: One box, one goal
    level = [
        "#####",
        "#   #",
        "# $ #",
        "# . #",
        "#@  #",
        "#####"
    ]
    
    initial_state = SokobanState(level)
    solver = SokobanSolver()
    
    print("Testing simple puzzle...")
    solution = solver.a_star_search(initial_state)
    
    if solution:
        print(f"Solution found! Number of moves: {len(solution)}")
        visualize_solution(initial_state, solution)
    else:
        print("No solution found!")

def test_two_box_puzzle():
    # Two boxes, two goals
    level = [
        "#######",
        "#     #",
        "# $ $ #",
        "# . . #",
        "#@    #",
        "#######"
    ]
    
    initial_state = SokobanState(level)
    solver = SokobanSolver()
    
    print("\nTesting two-box puzzle...")
    solution = solver.a_star_search(initial_state)
    
    if solution:
        print(f"Solution found! Number of moves: {len(solution)}")
        visualize_solution(initial_state, solution)
    else:
        print("No solution found!")

if __name__ == "__main__":
    test_simple_puzzle()
    test_two_box_puzzle() 