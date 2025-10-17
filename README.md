# Path Planning Solution Documentation

## Problem Overview

Implement a path planner for an autonomous racing car navigating between colored cones:
- **Input**: Car pose (x, y, yaw) and up to 2 cones per side
- **Cones**: Yellow (color=0) marks right boundary, Blue (color=1) marks left boundary  
- **Output**: Sequence of waypoints forming a drivable path between boundaries

## Solution Design

### Problem Formulation

- **Workspace**: 5×5m environment → 50×50 grid (10× resolution)
- **Start/Goal**: (0,0) → (50,50) in grid coordinates
- **Step Size**: 1-5 grid units (0.1-0.5m in world coordinates)
- **Collision Threshold**: 2 grid units (0.2m) minimum distance from cones

### Algorithm: Dijkstra's Search

The solution uses Dijkstra's algorithm to find the shortest collision-free path:

1. Expand nodes from priority queue ordered by cumulative cost
2. Validate each move against three safety constraints
3. Stop when goal is reachable
4. Reconstruct path via parent pointers

**Why Dijkstra**: Simple, guarantees optimal path, handles multi-constraint validation naturally. While A* or RRT* would be more efficient, Dijkstra works well for this problem.

## Safety Constraints

Every move is validated against three constraints implemented in `check_validity()`:

### 1. Collision Avoidance
**Method**: `point_segment_distance()`  
Calculates perpendicular distance from each cone to the path segment. Move is invalid if any cone is within 2 grid units.

### 2. Boundary Crossing
**Method**: `segments_intersect()`  
Checks if path segment intersects lines between same-colored cones. Prevents crossing track boundaries.

### 3. Directional Validity  
**Method**: `is_valid_cone_position()`  
Uses cross product to verify cone positioning: yellow must be on right (negative side), blue must be on left (positive side).

**Logic**:
- Cross product determines which side of path vector a cone lies on
- Validates nearest cones of each color
- Handles cases with one or both cone colors present
- When both colors on same side, validates the nearest cone

## Implementation

### Key Functions

**`check_validity(A, B)`**: Validates move from A to B against all three constraints

**`dijkstra(start, goal, min_step, max_step)`**: Main search algorithm returning list of grid waypoints

**`generatePath()`**: Public interface that runs Dijkstra and converts grid coordinates back to world frame (÷10)

### Assumptions

- Car orients itself toward movement direction before each step
- Fixed goal at (5, 5) meters
- Static cone positions
- Grid discretization provides sufficient path quality

## Performance & Limitations

### Runtime
- **5-15 seconds** per scenario
- Bottleneck: Repeated geometric validation for each neighbor
- Inefficient constant factors in implementation

### Known Issues

**Multiple Cones**: Doesn't scale to 3+ cones per side (Part 2 unsolved) due to the limited time.

**Scenario 21 Failure**: Inverted colors from Scenario 3 produces incorrect paths. The nearest-cone heuristic in direction validation breaks when cone configurations are mirrored.

**Algorithm Efficiency**: A* or RRT* would reduce explored nodes, but Dijkstra works fine for given scenarios.

## Results

Successfully solves scenarios 1-20 with various cone configurations and car orientations. Scenario 21 (added test case with inverted colors) fails, exposing limitations in direction validation logic.
