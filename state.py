from typing import List, Tuple, Set
import numpy as np

class SokobanState:
    # Constants for the game elements
    WALL = '#'
    BOX = '$'
    PLAYER = '@'
    GOAL = '.'
    FLOOR = ' '
    BOX_ON_GOAL = '*'
    PLAYER_ON_GOAL = '+'

    def __init__(self, level: List[str]):
        """
        Initialize the Sokoban state from a level representation.
        
        Args:
            level: List of strings representing the level, where each string is a row
        """
        self.level = level
        self.height = len(level)
        self.width = len(level[0])
        self.player_pos = self._find_player()
        self.boxes = self._find_boxes()
        self.goals = self._find_goals()
        
    def _find_player(self) -> Tuple[int, int]:
        """Find the player's position in the level."""
        for y in range(self.height):
            for x in range(self.width):
                if self.level[y][x] in [self.PLAYER, self.PLAYER_ON_GOAL]:
                    return (y, x)
        raise ValueError("No player found in level")

    def _find_boxes(self) -> Set[Tuple[int, int]]:
        """Find all boxes in the level."""
        boxes = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.level[y][x] in [self.BOX, self.BOX_ON_GOAL]:
                    boxes.add((y, x))
        return boxes

    def _find_goals(self) -> Set[Tuple[int, int]]:
        """Find all goal positions in the level."""
        goals = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.level[y][x] in [self.GOAL, self.BOX_ON_GOAL, self.PLAYER_ON_GOAL]:
                    goals.add((y, x))
        return goals

    def is_goal_state(self) -> bool:
        """Check if all boxes are on goals."""
        return all(box in self.goals for box in self.boxes)

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves for the player."""
        valid_moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dy, dx in directions:
            new_y, new_x = self.player_pos[0] + dy, self.player_pos[1] + dx
            
            # Check if the new position is within bounds
            if not (0 <= new_y < self.height and 0 <= new_x < self.width):
                continue
                
            # Check if the new position is a wall
            if self.level[new_y][new_x] == self.WALL:
                continue
                
            # Check if the new position has a box
            if (new_y, new_x) in self.boxes:
                # Check if we can push the box
                box_new_y, box_new_x = new_y + dy, new_x + dx
                if not (0 <= box_new_y < self.height and 0 <= box_new_x < self.width):
                    continue
                if self.level[box_new_y][box_new_x] == self.WALL:
                    continue
                if (box_new_y, box_new_x) in self.boxes:
                    continue
                    
            valid_moves.append((dy, dx))
            
        return valid_moves

    def move(self, direction: Tuple[int, int]) -> 'SokobanState':
        """
        Create a new state by moving the player in the given direction.
        
        Args:
            direction: Tuple (dy, dx) representing the direction to move
            
        Returns:
            New SokobanState after the move
        """
        dy, dx = direction
        new_y, new_x = self.player_pos[0] + dy, self.player_pos[1] + dx
        
        # Create a new level representation
        new_level = [list(row) for row in self.level]
        
        # Update player position
        old_y, old_x = self.player_pos
        if (old_y, old_x) in self.goals:
            new_level[old_y][old_x] = self.GOAL
        else:
            new_level[old_y][old_x] = self.FLOOR
            
        # Check if we're moving a box
        if (new_y, new_x) in self.boxes:
            box_new_y, box_new_x = new_y + dy, new_x + dx
            # Move the box
            if (box_new_y, box_new_x) in self.goals:
                new_level[box_new_y][box_new_x] = self.BOX_ON_GOAL
            else:
                new_level[box_new_y][box_new_x] = self.BOX
                
        # Update player position
        if (new_y, new_x) in self.goals:
            new_level[new_y][new_x] = self.PLAYER_ON_GOAL
        else:
            new_level[new_y][new_x] = self.PLAYER
            
        return SokobanState([''.join(row) for row in new_level])

    def __eq__(self, other: 'SokobanState') -> bool:
        """Check if two states are equal."""
        if not isinstance(other, SokobanState):
            return False
        return (self.player_pos == other.player_pos and 
                self.boxes == other.boxes)

    def __hash__(self) -> int:
        """Create a hash of the state for use in sets/dictionaries."""
        return hash((self.player_pos, frozenset(self.boxes)))

    def __str__(self) -> str:
        """String representation of the state."""
        return '\n'.join(self.level) 