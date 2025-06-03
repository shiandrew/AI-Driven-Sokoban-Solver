"""
Professional Sokoban Solver with Hungarian Algorithm Heuristic and IDA*
Complete implementation with A*, IDA*, BFS algorithms and advanced deadlock detection
"""

import heapq
import time
from collections import deque
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

try:
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PriorityQueue:
    """Efficient priority queue for A* search"""

    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}
        self.counter = 0  # Tie-breaker for equal priorities

    def update(self, state, new_priority):
        """Update priority or insert new state"""
        old_priority = self.priorities.get(state)
        if old_priority is None or new_priority < old_priority:
            self.priorities[state] = new_priority
            # Use counter as tie-breaker to avoid comparing SokobanState objects
            heapq.heappush(self.heap, (new_priority, self.counter, state))
            self.counter += 1
            return True
        return False

    def remove_min(self):
        """Remove and return state with minimum priority"""
        while len(self.heap) > 0:
            priority, _, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return state, priority
        return None, None

    def empty(self):
        """Check if queue is empty"""
        return len(self.heap) == 0


class SokobanState:
    """Represents a Sokoban game state"""

    def __init__(self, width: int, height: int, walls: Set[Tuple[int, int]],
                 player_pos: Tuple[int, int], boxes: List[Tuple[int, int]],
                 targets: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.walls = walls
        self.player_pos = player_pos
        self.boxes = tuple(sorted(boxes))  # Keep sorted for consistent hashing
        self.targets = set(targets)
        self.h = 0  # Heuristic value

    def __str__(self) -> str:
        """String representation for state comparison"""
        return f"P{self.player_pos}B{self.boxes}"

    def __hash__(self) -> int:
        """Hash for efficient state storage"""
        return hash((self.player_pos, self.boxes))

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        return (self.player_pos == other.player_pos and
                self.boxes == other.boxes)

    def is_wall(self, x: int, y: int) -> bool:
        """Check if position is a wall"""
        return (x, y) in self.walls

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (within bounds and not a wall)"""
        return (0 <= x < self.width and 0 <= y < self.height and
                not self.is_wall(x, y))

    def is_success(self) -> bool:
        """Check if all boxes are on targets"""
        return set(self.boxes) == self.targets

    def is_corner_deadlock(self, box_pos: Tuple[int, int]) -> bool:
        """Check if box is stuck in a corner"""
        if box_pos in self.targets:
            return False

        x, y = box_pos
        # Check if box is blocked horizontally and vertically
        horizontal_blocked = (self.is_wall(x - 1, y) or self.is_wall(x + 1, y))
        vertical_blocked = (self.is_wall(x, y - 1) or self.is_wall(x, y + 1))

        return horizontal_blocked and vertical_blocked

    def is_edge_deadlock(self, box_pos: Tuple[int, int]) -> bool:
        """
        Checks if a box is deadlocked against an edge (wall or boundary)
        and cannot be moved away from the edge, with no targets on the edge line.
        """
        if box_pos in self.targets:
            return False  # On target is always fine

        x, y = box_pos

        # Define edge directions: (dx, dy, is_edge)
        edges = [
            (-1, 0, lambda x, y: x == 0 or self.is_wall(x - 1, y)),  # Left edge
            (1, 0, lambda x, y: x == self.width - 1 or self.is_wall(x + 1, y)),  # Right edge
            (0, -1, lambda x, y: y == 0 or self.is_wall(x, y - 1)),  # Bottom edge
            (0, 1, lambda x, y: y == self.height - 1 or self.is_wall(x, y + 1)),  # Top edge
        ]

        for dx, dy, is_edge in edges:
            if is_edge(x, y):
                # Check: Is there a target on this edge line?
                if dx != 0:
                    target_on_edge = any(t[0] == x for t in self.targets)
                else:
                    target_on_edge = any(t[1] == y for t in self.targets)
                if target_on_edge:
                    continue  # Not a deadlock, can still push to target

                # Check if box can be pushed AWAY from edge
                away_x, away_y = x + dx, y + dy
                if self.is_valid_position(away_x, away_y) and (away_x, away_y) not in self.boxes:
                    return False  # Can escape the edge, not deadlocked

                # Also check moving perpendicular to the wall (i.e., along the wall)
                for perp_dx, perp_dy in [(-dy, dx), (dy, -dx)]:
                    check_x, check_y = x + perp_dx, y + perp_dy
                    if self.is_valid_position(check_x, check_y) and (check_x, check_y) not in self.boxes:
                        return False  # Can move along edge, so not deadlocked

                # If all moves blocked, this is really an edge deadlock
                return True
        return False

    def is_box_cluster_deadlock(self, box_pos: Tuple[int, int]) -> bool:
        """Detect cluster deadlocks: 2x2, L-shape, and stuck lines, with minimal false positives."""
        if box_pos in self.targets:
            return False

        x, y = box_pos
        boxes = set(self.boxes)
        targets = self.targets

        # 1. Check for 2x2 cluster (classic)
        square_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            square = [(x + ox, y + oy), (x + ox - 1, y + oy), (x + ox, y + oy - 1), (x + ox - 1, y + oy - 1)]
            in_boxes = [pos in boxes for pos in square]
            # If all 4 present and at least 3 not on targets: stuck
            if sum(in_boxes) == 4:
                off_targets = [pos not in targets for pos in square]
                if sum(off_targets) >= 3:
                    return True

        # 2. Check for "L" shape clusters (three boxes in a corner)
        l_shapes = [
            [(x, y), (x + 1, y), (x, y + 1)],
            [(x, y), (x - 1, y), (x, y + 1)],
            [(x, y), (x + 1, y), (x, y - 1)],
            [(x, y), (x - 1, y), (x, y - 1)],
        ]
        for cluster in l_shapes:
            if all(pos in boxes for pos in cluster):
                off_targets = [pos not in targets for pos in cluster]
                if sum(off_targets) >= 2:
                    # Check if L is up against wall(s)
                    wall_touch = (
                            self.is_wall(cluster[1][0], cluster[1][1]) or
                            self.is_wall(cluster[2][0], cluster[2][1])
                    )
                    if wall_touch:
                        return True

        # 3. Check for horizontal/vertical lines of 3+ boxes against a wall and not on targets
        for dx, dy in [(1, 0), (0, 1)]:  # horizontal and vertical
            for d in range(-2, 1):
                line = [(x + d * dx, y + d * dy), (x + (d + 1) * dx, y + (d + 1) * dy),
                        (x + (d + 2) * dx, y + (d + 2) * dy)]
                if all(pos in boxes and pos not in targets for pos in line):
                    # Check if line is against a wall
                    for pos in line:
                        wall_pos = (pos[0] + dy, pos[1] + dx)  # perpendicular to line
                        if self.is_wall(*wall_pos):
                            return True

        # 4. Check for "bends" (three boxes in an L that cannot be separated, each touching a wall)
        bends = [
            [(x, y), (x + 1, y), (x + 1, y + 1)],
            [(x, y), (x - 1, y), (x - 1, y + 1)],
            [(x, y), (x + 1, y), (x + 1, y - 1)],
            [(x, y), (x - 1, y), (x - 1, y - 1)],
        ]
        for cluster in bends:
            if all(pos in boxes and pos not in targets for pos in cluster):
                wall_touch = any(self.is_wall(pos[0], pos[1]) for pos in cluster)
                if wall_touch:
                    return True

        return False

    def is_freeze_deadlock(self, box_pos: Tuple[int, int]) -> bool:
        """Check for freeze deadlocks where boxes block each other - CONSERVATIVE VERSION"""
        if box_pos in self.targets:
            return False

        x, y = box_pos

        # Count how many directions the box CAN be moved
        moveable_directions = 0

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Position where box would go if pushed
            new_box_x, new_box_y = x + dx, y + dy
            # Position where player needs to be to push
            player_push_x, player_push_y = x - dx, y - dy

            # Check if this push is possible
            can_push = (
                # Box destination is valid and empty
                    self.is_valid_position(new_box_x, new_box_y) and
                    (new_box_x, new_box_y) not in self.boxes and
                    # Player push position is valid and empty (or is current player position)
                    self.is_valid_position(player_push_x, player_push_y) and
                    (((player_push_x, player_push_y) not in self.boxes) or
                     (player_push_x, player_push_y) == self.player_pos)
            )

            if can_push:
                moveable_directions += 1

        # Only consider it frozen if it can't move in ANY direction
        # AND it's completely surrounded (very conservative)
        if moveable_directions == 0:
            # Additional check: make sure it's really trapped, not just temporarily blocked
            # Check if ALL adjacent positions are walls or boxes
            adjacent_blocked = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_x, adj_y = x + dx, y + dy
                if (not self.is_valid_position(adj_x, adj_y) or
                        (adj_x, adj_y) in self.boxes):
                    adjacent_blocked += 1

            # Only freeze deadlock if completely surrounded
            return adjacent_blocked >= 3  # Very conservative - need 3+ sides blocked

        return False

    def is_failure(self, debug: bool = False) -> bool:
        """Check for comprehensive deadlock conditions with optional debugging"""
        # Quick exit if already solved
        if self.is_success():
            return False

        deadlocks_found = set()

        # Check each box for various deadlock types
        for i, box in enumerate(self.boxes):
            # Skip boxes already on targets
            if box in self.targets:
                continue

            # Check corner deadlocks
            if self.is_corner_deadlock(box):
                deadlocks_found.add("CORNER")

            # Check edge deadlocks (boxes stuck against walls)
            if self.is_edge_deadlock(box):
                deadlocks_found.add("EDGE")

            # Check 2x2 box cluster deadlocks
            if self.is_box_cluster_deadlock(box):
                deadlocks_found.add("CLUSTER")

            # Check freeze deadlocks
            if self.is_freeze_deadlock(box):
                deadlocks_found.add("FREEZE")

        if debug and has_deadlock:
            print(f"\n💀 DEADLOCK DETECTED!")
            print("=" * 50)
            self.render_board_with_deadlocks(deadlocks_found)
            print(f"Deadlocked boxes:")
            for box_idx, box_pos, deadlock_types in deadlocks_found:
                print(f"  Box {box_idx + 1} at {box_pos}: {', '.join(deadlock_types)}")
            print("=" * 50)

        return deadlocks_found

    def render_board_with_deadlocks(self, deadlocks_found):
        """Render board highlighting deadlocked boxes"""
        if not self:
            return

        print("\nBoard with deadlocks marked (X = deadlocked box):")
        deadlocked_positions = set(box_pos for _, box_pos, _ in deadlocks_found)

        for y in range(self.height - 1, -1, -1):
            display_row = ""
            for x in range(self.width):
                pos = (x, y)

                if pos in self.walls:
                    display_row += '█'
                else:
                    is_player = self.player_pos == pos
                    has_box = pos in self.boxes
                    is_target = pos in self.targets
                    is_deadlocked = pos in deadlocked_positions

                    if is_player and is_target:
                        display_row += '+'
                    elif is_player:
                        display_row += '@'
                    elif has_box and is_deadlocked:
                        display_row += 'X'  # Deadlocked box
                    elif has_box and is_target:
                        display_row += '■'
                    elif has_box:
                        display_row += '□'
                    elif is_target:
                        display_row += '○'
                    else:
                        display_row += ' '
            print(display_row)

    def get_pushable_boxes(self) -> List[Tuple[Tuple[int, int], str]]:
        """Get all boxes that can be pushed and their valid push directions"""
        from collections import deque

        def bfs_reachable(start, walls, boxes):
            """Find all positions reachable by the player"""
            visited = set()
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                if (x, y) in visited or (x, y) in walls or (x, y) in boxes:
                    continue
                visited.add((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        queue.append((nx, ny))
            return visited

        reachable = bfs_reachable(self.player_pos, self.walls, set(self.boxes))

        pushable_moves = []
        direction_map = {
            (-1, 0): 'L',  # Left
            (1, 0): 'R',  # Right
            (0, -1): 'D',  # Down
            (0, 1): 'U',  # Up
        }

        for box_pos in self.boxes:
            bx, by = box_pos

            for (dx, dy), direction in direction_map.items():
                # Position player needs to be to push box in this direction
                agent_x, agent_y = bx - dx, by - dy
                # Position box would move to
                target_x, target_y = bx + dx, by + dy

                # Check if push is valid
                if ((agent_x, agent_y) in reachable and  # Player can reach push position
                        self.is_valid_position(target_x, target_y) and  # Target is valid
                        (target_x, target_y) not in self.boxes):  # Target is empty

                    pushable_moves.append((box_pos, direction))

        return pushable_moves

    def get_possible_actions(self) -> List[Tuple[str, str]]:
        """Get all possible actions - optimized to only consider meaningful box pushes"""
        # Use pushable box analysis for much better efficiency
        pushable_moves = self.get_pushable_boxes()

        # Convert to action format
        actions = []
        for box_pos, direction in pushable_moves:
            actions.append((direction, 'Push', box_pos))

        return actions

    def successor(self, box_pos, action: str) -> 'SokobanState':
        """Generate successor state after pushing a box"""
        directions = {'U': (0, 1), 'D': (0, -1), 'L': (-1, 0), 'R': (1, 0)}
        dx, dy = directions[action]

        bx, by = box_pos

        # New box position
        new_box_x = bx + dx
        new_box_y = by + dy
        new_box_pos = (new_box_x, new_box_y)

        # New player position (where the box was)
        new_player_pos = box_pos

        # Update box list
        new_boxes = list(self.boxes)
        box_index = new_boxes.index(box_pos)
        new_boxes[box_index] = new_box_pos

        return SokobanState(
            self.width, self.height, self.walls,
            new_player_pos, new_boxes, list(self.targets)
        )

        # Should not reach here if action is valid
        raise ValueError(f"Invalid action: {action}")


class SokobanSolver:
    """Advanced Sokoban solver with multiple algorithms"""

    def __init__(self):
        self.cache = {}
        # Add these for IDA*
        self.solution_found = False
        self.solution_actions = ""
        self.ida_stats = {}
        self.transposition_table = {}
        self.nodes_explored = 0
        self.current_max_depth = 0

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def simple_heuristic(self, state: SokobanState) -> int:
        """Simple minimum distance heuristic"""
        if state.is_success():
            return 0

        boxes = list(state.boxes)
        targets = list(state.targets)

        total_cost = 0
        remaining_targets = targets[:]

        # Assign each box to closest available target
        for box in boxes:
            if box in state.targets:
                continue  # Box already on target

            min_dist = float('inf')
            best_target = None

            for target in remaining_targets:
                dist = self.manhattan_distance(box, target)
                if dist < min_dist:
                    min_dist = dist
                    best_target = target

            if best_target:
                total_cost += min_dist
                remaining_targets.remove(best_target)

        return total_cost

    def hungarian_heuristic(self, state: SokobanState) -> int:
        """Hungarian algorithm for optimal box-to-target assignment"""
        if state.is_success():
            return 0

        if not SCIPY_AVAILABLE:
            return self.simple_heuristic(state)

        boxes = [box for box in state.boxes if box not in state.targets]
        targets = [target for target in state.targets if target not in state.boxes]

        if not boxes or not targets:
            return 0

        # Create cost matrix
        cost_matrix = np.zeros((len(boxes), len(targets)))
        for i, box in enumerate(boxes):
            for j, target in enumerate(targets):
                cost_matrix[i][j] = self.manhattan_distance(box, target)

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_indices, col_indices].sum()

        return int(total_cost)

    def astar(self, start_state: SokobanState, max_cost: int = 1000,
              use_hungarian: bool = True) -> Tuple[str, int]:
        """A* search with Hungarian heuristic and deadlock detection"""
        heuristic = self.hungarian_heuristic if use_hungarian else self.simple_heuristic

        queue = PriorityQueue()
        action_map = {}
        visited = set()

        # Deadlock detection statistics
        deadlock_stats = {
            'corner': 0,
            'edge': 0,
            'cluster': 0,
            'freeze': 0,
            'total_pruned': 0
        }

        start_state.h = heuristic(start_state)
        queue.update(start_state, start_state.h)
        action_map[start_state] = ""

        iterations = 0

        print(f"A* starting with {'Hungarian' if use_hungarian else 'Simple'} heuristic: {start_state.h}")
        print("🛡️  Advanced deadlock detection enabled")

        while not queue.empty():
            state, cost = queue.remove_min()

            if not state:
                break

            iterations += 1

            if iterations % 1000 == 0:
                total_pruned = deadlock_stats['total_pruned']
                print(f"A* iterations: {iterations}, cost: {cost:.1f}, visited: {len(visited)}")
                print(
                    f"  💀 Deadlocks pruned: {total_pruned} (corner:{deadlock_stats['corner']}, edge:{deadlock_stats['edge']}, cluster:{deadlock_stats['cluster']}, freeze:{deadlock_stats['freeze']})")

            if state in visited:
                continue

            visited.add(state)
            actions = action_map[state]

            if state.is_success():
                total_pruned = deadlock_stats['total_pruned']
                print(f"🎉 Solution found in {iterations} iterations!")
                print(f"💀 Total deadlocks detected: {total_pruned}")
                return actions, len(visited)

            if cost >= max_cost:
                continue

            if state.is_failure():
                dead_found = state.is_failure()
                deadlock_stats['total_pruned'] += 1
                # Count specific deadlock types for statistics
                if 'CORNER' in dead_found:
                    deadlock_stats['corner'] += 1
                elif 'EDGE' in dead_found:
                    # s = SokobanGame()
                    # s.current_state = state
                    # s.render_board()
                    deadlock_stats['edge'] += 1
                elif 'CLUSTER' in dead_found:
                    deadlock_stats['cluster'] += 1
                elif 'FREEZE' in dead_found:
                    deadlock_stats['freeze'] += 1
                '''
                for box in state.boxes:
                    if box in state.targets:
                        continue
                    if state.is_corner_deadlock(box):
                        deadlock_stats['corner'] += 1
                    elif state.is_edge_deadlock(box):
                        deadlock_stats['edge'] += 1
                    elif state.is_box_cluster_deadlock(box):
                        deadlock_stats['cluster'] += 1
                    elif state.is_freeze_deadlock(box):
                        deadlock_stats['freeze'] += 1
                '''
                continue

            for action, cost_type, box_pos in state.get_possible_actions():
                successor = state.successor(box_pos, action)

                if successor in visited:
                    continue

                old_actions = action_map.get(successor)
                new_actions = actions + f'push {str(box_pos)} to {action}\n'

                if not old_actions or len(new_actions) < len(old_actions):
                    action_map[successor] = new_actions
                    successor.h = heuristic(successor)
                    action_cost = 2 if cost_type == 'Push' else 1
                    priority = cost + action_cost + successor.h - state.h
                    queue.update(successor, priority)

        total_pruned = deadlock_stats['total_pruned']
        print(f"A* completed {iterations} iterations without solution")
        print(f"💀 Total deadlocks detected: {total_pruned}")
        return "", 0

    def ida_star(self, start_state: SokobanState, max_cost: int = 1000,
                 use_hungarian: bool = True) -> Tuple[str, int]:
        """
        IDA* (Iterative Deepening A*) search algorithm.
        Memory-efficient variant of A* that uses depth-first search with iteratively increasing bounds.
        """
        heuristic = self.hungarian_heuristic if use_hungarian else self.simple_heuristic

        # Statistics tracking
        self.ida_stats = {
            'nodes_generated': 0,
            'nodes_revisited': 0,
            'iterations': 0,
            'deadlocks_pruned': 0,
            'deadlock_types': {'corner': 0, 'edge': 0, 'cluster': 0, 'freeze': 0}
        }

        # Calculate initial bound
        start_state.h = heuristic(start_state)
        bound = start_state.h

        print(f"IDA* starting with {'Hungarian' if use_hungarian else 'Simple'} heuristic")
        print(f"Initial bound: {bound}")
        print("🛡️  Deadlock detection enabled")

        # Store the solution path when found
        self.solution_found = False
        self.solution_actions = ""

        # Iteratively increase bound until solution found or limit reached
        while bound < max_cost and not self.solution_found:
            self.ida_stats['iterations'] += 1
            print(f"\n🔄 IDA* Iteration {self.ida_stats['iterations']}, bound: {bound}")

            # Reset path for this iteration
            path = []
            actions = ""

            # Perform depth-first search with current bound
            min_excess = self._ida_search(start_state, 0, bound, path, actions,
                                          heuristic, set())

            if self.solution_found:
                print(f"\n🎉 Solution found in {self.ida_stats['iterations']} iterations!")
                print(f"Nodes generated: {self.ida_stats['nodes_generated']}")
                print(f"Nodes revisited: {self.ida_stats['nodes_revisited']}")
                print(f"Deadlocks pruned: {self.ida_stats['deadlocks_pruned']}")
                return self.solution_actions, self.ida_stats['nodes_generated']

            # Update bound for next iteration
            if min_excess == float('inf'):
                print("No solution exists within constraints")
                break

            bound = min_excess
            print(f"Next bound will be: {bound}")

        print(f"\nIDA* completed without finding solution")
        print(f"Total iterations: {self.ida_stats['iterations']}")
        print(f"Nodes generated: {self.ida_stats['nodes_generated']}")
        return "", 0

    def _ida_search(self, state: SokobanState, g_cost: float, bound: float,
                    path: List[SokobanState], actions: str, heuristic,
                    cycle_check: Set[SokobanState]) -> float:
        """
        Recursive depth-first search for IDA*.
        Returns the minimum cost that exceeded the bound.
        """
        self.ida_stats['nodes_generated'] += 1

        # Calculate f-cost
        f_cost = g_cost + state.h

        # Check if we've exceeded bound
        if f_cost > bound:
            return f_cost

        # Check for success
        if state.is_success():
            self.solution_found = True
            self.solution_actions = actions
            return f_cost

        # Deadlock detection
        if state.is_failure():
            self.ida_stats['deadlocks_pruned'] += 1

            # Track deadlock types for statistics
            for box in state.boxes:
                if box in state.targets:
                    continue
                if state.is_corner_deadlock(box):
                    self.ida_stats['deadlock_types']['corner'] += 1
                elif state.is_edge_deadlock(box):
                    self.ida_stats['deadlock_types']['edge'] += 1
                elif state.is_box_cluster_deadlock(box):
                    self.ida_stats['deadlock_types']['cluster'] += 1
                elif state.is_freeze_deadlock(box):
                    self.ida_stats['deadlock_types']['freeze'] += 1

            return float('inf')

        # Cycle detection using the path
        if state in cycle_check:
            self.ida_stats['nodes_revisited'] += 1
            return float('inf')

        # Add state to path and cycle check
        path.append(state)
        cycle_check.add(state)

        # Try all possible actions
        min_excess = float('inf')

        # Get and sort actions by heuristic value to improve pruning
        possible_actions = []
        for action, cost_type, box_pos in state.get_possible_actions():
            successor = state.successor(box_pos, action)
            successor.h = heuristic(successor)
            action_cost = 2 if cost_type == 'Push' else 1
            possible_actions.append((successor.h, action, cost_type, box_pos, successor))

        # Sort by heuristic to try most promising moves first
        possible_actions.sort(key=lambda x: x[0])

        # Explore successors
        for _, action, cost_type, box_pos, successor in possible_actions:
            if self.solution_found:
                break

            action_cost = 2 if cost_type == 'Push' else 1
            new_g_cost = g_cost + action_cost
            new_actions = actions + f'push {str(box_pos)} to {action}\n'

            # Recursive search
            result = self._ida_search(successor, new_g_cost, bound, path,
                                      new_actions, heuristic, cycle_check)

            if result < min_excess:
                min_excess = result

        # Remove state from path and cycle check
        path.pop()
        cycle_check.remove(state)

        return min_excess

    def ida_star_with_transposition(self, start_state: SokobanState, max_cost: int = 1000,
                                    use_hungarian: bool = True) -> Tuple[str, int]:
        """
        Enhanced IDA* with transposition table for better efficiency.
        Uses a cache to avoid recomputing states reached via different paths.
        """
        heuristic = self.hungarian_heuristic if use_hungarian else self.simple_heuristic

        # Statistics
        self.ida_stats = {
            'nodes_generated': 0,
            'cache_hits': 0,
            'iterations': 0,
            'deadlocks_pruned': 0
        }

        # Transposition table: state -> (best_g_cost, actions)
        self.transposition_table = {}

        start_state.h = heuristic(start_state)
        bound = start_state.h

        print(f"IDA* with transposition table starting")
        print(f"Using {'Hungarian' if use_hungarian else 'Simple'} heuristic")
        print(f"Initial bound: {bound}")

        self.solution_found = False
        self.solution_actions = ""

        while bound < max_cost and not self.solution_found:
            self.ida_stats['iterations'] += 1
            print(f"\n🔄 IDA* Iteration {self.ida_stats['iterations']}, bound: {bound}")

            # Clear transposition table for new iteration (optional - can keep between iterations)
            # self.transposition_table.clear()

            min_excess = self._ida_search_transposition(
                start_state, 0, bound, "", heuristic, set()
            )

            if self.solution_found:
                print(f"\n🎉 Solution found!")
                print(f"Iterations: {self.ida_stats['iterations']}")
                print(f"Nodes generated: {self.ida_stats['nodes_generated']}")
                print(f"Cache hits: {self.ida_stats['cache_hits']}")
                print(f"Deadlocks pruned: {self.ida_stats['deadlocks_pruned']}")
                return self.solution_actions, self.ida_stats['nodes_generated']

            if min_excess == float('inf'):
                break

            bound = min_excess

        return "", 0

    def _ida_search_transposition(self, state: SokobanState, g_cost: float,
                                  bound: float, actions: str, heuristic,
                                  path_states: Set[SokobanState]) -> float:
        """
        IDA* search with transposition table support.
        """
        self.ida_stats['nodes_generated'] += 1

        # Check transposition table
        if state in self.transposition_table:
            cached_g, cached_actions = self.transposition_table[state]
            if cached_g <= g_cost:
                self.ida_stats['cache_hits'] += 1
                # We've seen this state with equal or better cost
                return float('inf')

        # Update transposition table
        self.transposition_table[state] = (g_cost, actions)

        f_cost = g_cost + state.h

        if f_cost > bound:
            return f_cost

        if state.is_success():
            self.solution_found = True
            self.solution_actions = actions
            return f_cost

        if state.is_failure():
            self.ida_stats['deadlocks_pruned'] += 1
            return float('inf')

        # Path-based cycle detection
        if state in path_states:
            return float('inf')

        path_states.add(state)
        min_excess = float('inf')

        # Get successors sorted by heuristic
        successors = []
        for action, cost_type, box_pos in state.get_possible_actions():
            successor = state.successor(box_pos, action)
            successor.h = heuristic(successor)
            successors.append((successor.h, action, cost_type, box_pos, successor))

        successors.sort(key=lambda x: x[0])

        for _, action, cost_type, box_pos, successor in successors:
            if self.solution_found:
                break

            action_cost = 2 if cost_type == 'Push' else 1
            new_g_cost = g_cost + action_cost
            new_actions = actions + f'push {str(box_pos)} to {action}\n'

            result = self._ida_search_transposition(
                successor, new_g_cost, bound, new_actions, heuristic, path_states
            )

            min_excess = min(min_excess, result)

        path_states.remove(state)
        return min_excess

    def bfs(self, start_state: SokobanState, max_depth: int = 100) -> Tuple[str, int]:
        """Breadth-First Search"""
        queue = deque([start_state])
        action_map = {start_state: ""}
        visited = set()

        iterations = 0

        while queue:
            state = queue.popleft()

            if state in visited:
                continue
            visited.add(state)

            iterations += 1
            if iterations % 5000 == 0:
                print(f"BFS iterations: {iterations}, queue: {len(queue)}")

            current_actions = action_map[state]

            if state.is_success():
                print(f"🎉 BFS found solution in {iterations} iterations!")
                return current_actions, len(visited)

            if len(current_actions) >= max_depth or state.is_failure():
                continue

            for action, _, box_pos in state.get_possible_actions():
                successor = state.successor(box_pos, action)

                if successor not in visited and successor not in action_map:
                    action_map[successor] = current_actions + action
                    queue.append(successor)

        print(f"BFS completed {iterations} iterations without solution")
        return "", 0


class SokobanGame:
    """Main game interface"""

    def __init__(self):
        self.current_state = None
        self.solution = ""
        self.solver = SokobanSolver()

    def load_from_file(self, filename: str) -> bool:
        """Load puzzle from file with 1-indexed coordinates"""
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            # Parse dimensions
            width, height = map(int, lines[0].split())

            # Parse walls (convert 1-indexed to 0-indexed)
            wall_data = list(map(int, lines[1].split()))
            num_walls = wall_data[0]
            walls = set()
            for i in range(num_walls):
                x, y = wall_data[1 + i * 2] - 1, wall_data[1 + i * 2 + 1] - 1
                walls.add((x, y))

            # Parse boxes
            box_data = list(map(int, lines[2].split()))
            num_boxes = box_data[0]
            boxes = []
            for i in range(num_boxes):
                x, y = box_data[1 + i * 2] - 1, box_data[1 + i * 2 + 1] - 1
                boxes.append((x, y))

            # Parse targets
            target_data = list(map(int, lines[3].split()))
            num_targets = target_data[0]
            targets = []
            for i in range(num_targets):
                x, y = target_data[1 + i * 2] - 1, target_data[1 + i * 2 + 1] - 1
                targets.append((x, y))

            # Parse player position
            player_x, player_y = map(int, lines[4].split())
            player_pos = (player_x - 1, player_y - 1)

            # Create state
            self.current_state = SokobanState(width, height, walls, player_pos, boxes, targets)

            print(f"✅ Loaded puzzle: {width}x{height}")
            print(f"Walls: {num_walls}, Boxes: {num_boxes}, Targets: {num_targets}")
            print(f"Player: {player_pos}")

            if SCIPY_AVAILABLE:
                print("🧮 Hungarian algorithm available for optimal heuristic")
            else:
                print("⚠️  Using simple heuristic (install scipy for better performance)")

            self.render_board()
            return True

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False

    def render_board(self):
        """Render the current board state"""
        if not self.current_state:
            print("No puzzle loaded!")
            return

        print("\n" + "=" * 50)
        # Render from top to bottom (flip Y axis since input uses bottom-left origin)
        for y in range(self.current_state.height - 1, -1, -1):
            display_row = ""
            for x in range(self.current_state.width):
                pos = (x, y)

                if pos in self.current_state.walls:
                    display_row += '█'
                else:
                    is_player = self.current_state.player_pos == pos
                    has_box = pos in self.current_state.boxes
                    is_target = pos in self.current_state.targets

                    if is_player and is_target:
                        display_row += '+'
                    elif is_player:
                        display_row += '@'
                    elif has_box and is_target:
                        display_row += '■'
                    elif has_box:
                        display_row += '□'
                    elif is_target:
                        display_row += '○'
                    else:
                        display_row += ' '
            print(display_row)
        print("=" * 50)

    def validate_puzzle(self) -> bool:
        """Validate puzzle for basic correctness"""
        if not self.current_state:
            return False

        boxes = list(self.current_state.boxes)
        targets = list(self.current_state.targets)

        print(f"\n🔍 PUZZLE VALIDATION:")
        print(f"Boxes: {len(boxes)} at {boxes}")
        print(f"Targets: {len(targets)} at {targets}")

        if len(boxes) != len(targets):
            print(f"❌ Mismatch: {len(boxes)} boxes vs {len(targets)} targets")
            return False

        # Check for immediate deadlocks
        deadlocked_boxes = []
        for box in boxes:
            if box not in targets and self.current_state.is_corner_deadlock(box):
                deadlocked_boxes.append(box)

        if deadlocked_boxes:
            print(f"❌ Corner deadlocks: {deadlocked_boxes}")
            return False

        print("✅ Basic validation passed")
        return True

    def solve(self, algorithm: str = 'astar_hungarian'):
        """Solve the puzzle"""
        if not self.current_state:
            print("No puzzle loaded!")
            return

        print(f"\n🚀 Solving with {algorithm.upper()}...")
        start_time = time.time()

        try:
            if algorithm == 'astar_hungarian':
                solution, states_explored = self.solver.astar(self.current_state, use_hungarian=True)
            elif algorithm == 'astar_simple':
                solution, states_explored = self.solver.astar(self.current_state, use_hungarian=False)
            elif algorithm == 'ida_star':
                solution, states_explored = self.solver.ida_star(self.current_state, use_hungarian=True)
            elif algorithm == 'ida_star_simple':
                solution, states_explored = self.solver.ida_star(self.current_state, use_hungarian=False)
            elif algorithm == 'ida_star_trans':
                solution, states_explored = self.solver.ida_star_with_transposition(self.current_state,
                                                                                    use_hungarian=True)
            elif algorithm == 'bfs':
                solution, states_explored = self.solver.bfs(self.current_state)
            else:
                print(f"Unknown algorithm: {algorithm}")
                return

            end_time = time.time()

            if solution:
                self.solution = solution
                print(f"\n🎉 SOLUTION FOUND! 🎉")
                print(f"Solution length: {len(solution.splitlines())} moves")
                print(f"States explored: {states_explored}")
                print(f"Time: {end_time - start_time:.2f} seconds")
                print(f"\nSolution moves:")
                for i, move in enumerate(solution.strip().split('\n'), 1):
                    print(f"  {i}. {move}")
            else:
                print(f"\n❌ No solution found")
                print(f"States explored: {states_explored}")
                print(f"Time: {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"❌ Error during solving: {e}")
            import traceback
            traceback.print_exc()

    def compare_algorithms(self):
        """Compare performance of different algorithms on current puzzle"""
        if not self.current_state:
            print("No puzzle loaded!")
            return

        print("\n🔬 ALGORITHM COMPARISON")
        print("=" * 60)

        algorithms = [
            ('A* + Hungarian', 'astar_hungarian'),
            ('A* + Simple', 'astar_simple'),
            ('IDA* + Hungarian', 'ida_star'),
            ('IDA* + Simple', 'ida_star_simple'),
            ('IDA* + Transposition', 'ida_star_trans'),
            # ('BFS', 'bfs')  # Usually too slow for comparison
        ]

        results = []

        for name, algo in algorithms:
            print(f"\n📊 Testing {name}...")
            start_time = time.time()

            try:
                if algo.startswith('ida_star'):
                    if algo == 'ida_star':
                        solution, states = self.solver.ida_star(self.current_state, use_hungarian=True)
                    elif algo == 'ida_star_simple':
                        solution, states = self.solver.ida_star(self.current_state, use_hungarian=False)
                    else:  # ida_star_trans
                        solution, states = self.solver.ida_star_with_transposition(self.current_state)
                elif algo.startswith('astar'):
                    solution, states = self.solver.astar(
                        self.current_state,
                        use_hungarian=(algo == 'astar_hungarian')
                    )
                else:
                    solution, states = self.solver.bfs(self.current_state)

                end_time = time.time()

                results.append({
                    'name': name,
                    'time': end_time - start_time,
                    'states': states,
                    'found': bool(solution),
                    'moves': len(solution.splitlines()) if solution else 0
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'name': name,
                    'time': 0,
                    'states': 0,
                    'found': False,
                    'moves': 0
                })

        # Display comparison table
        print("\n📈 RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Algorithm':<25} {'Time (s)':<12} {'States':<15} {'Moves':<10} {'Status'}")
        print("-" * 80)

        for r in results:
            status = "✅ Solved" if r['found'] else "❌ Failed"
            print(f"{r['name']:<25} {r['time']:<12.3f} {r['states']:<15,} {r['moves']:<10} {status}")

        print("=" * 80)

        # Find best performer
        if any(r['found'] for r in results):
            best_time = min(r['time'] for r in results if r['found'])
            best_algo = next(r['name'] for r in results if r['time'] == best_time and r['found'])
            print(f"\n🏆 Fastest: {best_algo} ({best_time:.3f}s)")

            best_memory = min(r['states'] for r in results if r['found'])
            best_mem_algo = next(r['name'] for r in results if r['states'] == best_memory and r['found'])
            print(f"💾 Most memory efficient: {best_mem_algo} ({best_memory:,} states)")

    def debug_box_analysis(self, box_pos: Tuple[int, int], box_index: int):
        """Detailed analysis of a specific box"""
        print(f"\n🔍 DETAILED ANALYSIS - Box {box_index + 1} at {box_pos}")
        print("-" * 40)

        state = self.current_state
        x, y = box_pos

        # Check if on target
        if box_pos in state.targets:
            print("✅ Box is on a target - SAFE")
            return

        print(f"❌ Box is NOT on target")

        # Check corner deadlock
        horizontal_blocked = (state.is_wall(x - 1, y) or state.is_wall(x + 1, y))
        vertical_blocked = (state.is_wall(x, y - 1) or state.is_wall(x, y + 1))
        corner_deadlock = horizontal_blocked and vertical_blocked

        print(f"Corner analysis:")
        print(f"  Left wall: {state.is_wall(x - 1, y)}, Right wall: {state.is_wall(x + 1, y)}")
        print(f"  Up wall: {state.is_wall(x, y + 1)}, Down wall: {state.is_wall(x, y - 1)}")
        print(f"  Horizontal blocked: {horizontal_blocked}, Vertical blocked: {vertical_blocked}")
        print(f"  Corner deadlock: {corner_deadlock}")

        # Check movement possibilities
        print(f"Movement analysis:")
        for direction, (dx, dy) in [('UP', (0, 1)), ('DOWN', (0, -1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]:
            new_box_x, new_box_y = x + dx, y + dy
            player_push_x, player_push_y = x - dx, y - dy

            box_dest_valid = state.is_valid_position(new_box_x, new_box_y)
            box_dest_empty = (new_box_x, new_box_y) not in state.boxes
            push_pos_valid = state.is_valid_position(player_push_x, player_push_y)
            push_pos_empty = ((player_push_x, player_push_y) not in state.boxes or
                              (player_push_x, player_push_y) == state.player_pos)

            can_push = box_dest_valid and box_dest_empty and push_pos_valid and push_pos_empty

            print(f"  {direction}: dest=({new_box_x},{new_box_y}) push_from=({player_push_x},{player_push_y})")
            print(f"    dest_valid={box_dest_valid}, dest_empty={box_dest_empty}")
            print(f"    push_valid={push_pos_valid}, push_empty={push_pos_empty}")
            print(f"    CAN_PUSH: {can_push}")

    def test_deadlock_detection(self):
        """Test deadlock detection on current state with detailed analysis"""
        if not self.current_state:
            print("No puzzle loaded!")
            return

        print(f"\n🔍 COMPREHENSIVE DEADLOCK ANALYSIS")
        print("=" * 60)

        state = self.current_state
        boxes = list(state.boxes)

        print(f"Analyzing {len(boxes)} boxes...")
        print(f"Targets are at: {list(state.targets)}")
        print(f"Player is at: {state.player_pos}")

        for i, box in enumerate(boxes):
            self.debug_box_analysis(box, i)

        # Test each deadlock type individually
        print(f"\n📊 DEADLOCK TYPE SUMMARY:")
        print("=" * 40)

        for i, box in enumerate(boxes):
            if box in state.targets:
                continue

            deadlock_results = {
                'corner': state.is_corner_deadlock(box),
                'edge': state.is_edge_deadlock(box),
                'cluster': state.is_box_cluster_deadlock(box),
                'freeze': state.is_freeze_deadlock(box)
            }

            detected_types = [t for t, detected in deadlock_results.items() if detected]

            if detected_types:
                print(f"Box {i + 1} at {box}: {', '.join(detected_types)}")
            else:
                print(f"Box {i + 1} at {box}: SAFE")

        # Overall assessment
        overall_deadlock = state.is_failure()
        print(f"\nOverall state: {'DEADLOCKED' if overall_deadlock else 'OK'}")

        if overall_deadlock:
            print("⚠️  Deadlock detection says this state is unsolvable!")

    def show_coordinate_verification(self):
        """Verify coordinate loading is working correctly"""
        if not self.current_state:
            print("No puzzle loaded!")
            return

        print(f"\n📐 COORDINATE VERIFICATION")
        print("=" * 50)

        state = self.current_state

        print(f"Grid dimensions: {state.width} x {state.height}")
        print(f"Player position (0-indexed): {state.player_pos}")
        print(f"Boxes (0-indexed): {list(state.boxes)}")
        print(f"Targets (0-indexed): {list(state.targets)}")

        print(f"\nOriginal 1-indexed input conversion:")
        print(f"Player: {state.player_pos} was ({state.player_pos[0] + 1}, {state.player_pos[1] + 1}) in file")
        print("Boxes:")
        for i, box in enumerate(state.boxes):
            print(f"  Box {i + 1}: {box} was ({box[0] + 1}, {box[1] + 1}) in file")
        print("Targets:")
        for i, target in enumerate(state.targets):
            print(f"  Target {i + 1}: {target} was ({target[0] + 1}, {target[1] + 1}) in file")

        # Show a coordinate reference grid
        print(f"\nCoordinate reference grid (showing positions):")
        print("   ", end="")
        for x in range(min(state.width, 15)):
            print(f"{x:2}", end="")
        print()

        for y in range(min(state.height, 10) - 1, -1, -1):
            print(f"{y:2} ", end="")
            for x in range(min(state.width, 15)):
                pos = (x, y)
                if pos in state.walls:
                    print(" █", end="")
                elif state.player_pos == pos:
                    print(" @", end="")
                elif pos in state.boxes:
                    print(" □", end="")
                elif pos in state.targets:
                    print(" ○", end="")
                else:
                    print("  ", end="")
            print()

        print("\nLegend: @ = Player, □ = Box, ○ = Target, █ = Wall")

    def run(self):
        """Main interface loop"""
        print("🎮 PROFESSIONAL SOKOBAN SOLVER with IDA*")
        print("=" * 50)

        while True:
            print(f"\n📋 OPTIONS:")
            print("1. Load puzzle from file")
            print("2. Solve with A* + Hungarian heuristic")
            print("3. Solve with A* + Simple heuristic")
            print("4. Solve with IDA* + Hungarian heuristic")
            print("5. Solve with IDA* + Simple heuristic")
            print("6. Solve with IDA* + Transposition table")
            print("7. Compare all algorithms")
            print("8. Solve with BFS")
            print("9. Validate puzzle")
            print("10. Show board")
            print("11. Test deadlock detection")
            print("12. Show coordinate verification")
            print("0. Quit")

            choice = input("\nEnter choice: ").strip()

            if choice == '1':
                filename = input("Enter filename: ").strip()
                self.load_from_file(filename)
            elif choice == '2':
                self.solve('astar_hungarian')
            elif choice == '3':
                self.solve('astar_simple')
            elif choice == '4':
                self.solve('ida_star')
            elif choice == '5':
                self.solve('ida_star_simple')
            elif choice == '6':
                self.solve('ida_star_trans')
            elif choice == '7':
                self.compare_algorithms()
            elif choice == '8':
                self.solve('bfs')
            elif choice == '9':
                self.validate_puzzle()
            elif choice == '10':
                self.render_board()
            elif choice == '11':
                self.test_deadlock_detection()
            elif choice == '12':
                self.show_coordinate_verification()
            elif choice == '0':
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")


def create_test_files():
    """Create test puzzle files"""

    # Your specific puzzle
    with open("sokoban01.txt", "w") as f:
        f.write("11 19\n")
        f.write(
            "70 1 5 1 6 1 7 1 8 1 9 2 5 2 9 3 5 3 9 4 3 4 4 4 5 4 9 4 10 5 3 5 10 6 1 6 2 6 3 6 5 6 7 6 8 6 10 6 14 6 15 6 16 6 17 6 18 6 19 7 1 7 5 7 7 7 8 7 10 7 11 7 12 7 13 7 14 7 19 8 1 8 19 9 1 9 2 9 3 9 4 9 5 9 7 9 8 9 9 9 11 9 13 9 14 9 19 10 5 10 11 10 12 10 13 10 14 10 15 10 16 10 17 10 18 10 19 11 5 11 6 11 7 11 8 11 9 11 10 11 11\n")
        f.write("6 3 6 4 8 5 6 5 8 8 3 8 6\n")
        f.write("6 7 17 7 18 8 17 8 18 9 17 9 18\n")
        f.write("9 12\n")

    # Simple test puzzle
    with open("simple.txt", "w") as f:
        f.write("5 5\n")
        f.write("16 1 1 1 2 1 3 1 4 1 5 2 1 2 5 3 1 3 5 4 1 4 5 5 1 5 2 5 3 5 4 5 5\n")
        f.write("1 3 3\n")
        f.write("1 3 4\n")
        f.write("2 2\n")

    # Medium complexity puzzle - IDA* shines here
    with open("ida_test.txt", "w") as f:
        f.write("7 7\n")
        f.write(
            "24 1 1 1 2 1 3 1 4 1 5 1 6 1 7 2 1 2 7 3 1 3 3 3 5 3 7 4 1 4 7 5 1 5 3 5 5 5 7 6 1 6 7 7 1 7 2 7 3 7 4 7 5 7 6 7 7\n")
        f.write("3 2 2 4 4 6 6\n")
        f.write("3 2 6 4 2 6 2\n")
        f.write("4 5\n")

    print("✅ Created test files: sokoban01.txt, simple.txt, ida_test.txt")


if __name__ == "__main__":
    create_test_files()
    game = SokobanGame()
    game.run()