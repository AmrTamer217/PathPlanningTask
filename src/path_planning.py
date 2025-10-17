from __future__ import annotations

from typing import List

from src.models import CarPose, Cone, Path2D

import numpy as np
import heapq
import math
from scipy.interpolate import splprep, splev
from typing import Tuple, List, Dict, Optional
from queue import Queue

class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """

    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def point_segment_distance(self, P, A, B):
        P, A, B = np.array(P), np.array(A), np.array(B)
        AB = B - A
        AP = P - A
        denom = np.dot(AB, AB)

        if denom == 0:  # A and B are the same point
            return np.linalg.norm(P - A)

        t = np.dot(AP, AB) / denom
        t = np.clip(t, 0, 1)
        C = A + t * AB
        return np.linalg.norm(P - C)
    

    def segments_intersect(self, A, B, C, D):
        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # 1 = clockwise, 2 = counterclockwise
        
        o1 = orientation(A, B, C)
        o2 = orientation(A, B, D)
        o3 = orientation(C, D, A)
        o4 = orientation(C, D, B)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special collinear cases
        if o1 == 0 and on_segment(A, C, B): 
            return True
        if o2 == 0 and on_segment(A, D, B): 
            return True
        if o3 == 0 and on_segment(C, A, D): 
            return True
        if o4 == 0 and on_segment(C, B, D): 
            return True

        return False
    
    def is_valid_cone_position(self, point, prev_point, yellow_cones: np.ndarray, blue_cones: np.ndarray) -> bool:
        """
        Check if point maintains correct position relative to nearest cones.
        Yellow (0) should be on right, Blue (1) should be on left.
        
        Args:
            point: Current point to validate
            prev_point: Previous point (to form direction vector)
            yellow_cones: Array of yellow cone positions
            blue_cones: Array of blue cone positions
        
        Returns:
            True if point is correctly positioned relative to cones
        """

        def point_side_of_line(point, line_start, line_end) -> float:
            px, py = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # Cross product: (line_end - line_start) × (point - line_start)
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        # Find nearest yellow cone
        if len(yellow_cones) > 0 and len(blue_cones) > 0:
            yellow_dists = np.linalg.norm(yellow_cones - np.array(point), axis=1)
            nearest_yellow = yellow_cones[np.argmin(yellow_dists)]
            blue_dists = np.linalg.norm(blue_cones - np.array(point), axis=1)
            nearest_blue = blue_cones[np.argmin(blue_dists)]
            
            yellow_side = point_side_of_line(nearest_yellow, prev_point, point)
            blue_side = point_side_of_line(nearest_blue, prev_point, point)

            if (yellow_side < 0 and blue_side < 0) or (yellow_side > 0 and blue_side > 0):  
                if self.point_segment_distance(nearest_blue, point, prev_point) < self.point_segment_distance(nearest_yellow, point, prev_point):
                    if blue_side < 0:  # Blue on right - INVALID
                        return False
                else:
                    if yellow_side > 0:  # Yellow on left - INVALID
                        return False
            elif yellow_side > 0 and blue_side < 0:  # Yellow on left and Blue on right - INVALID
                return False
        elif len(blue_cones) > 0:
            blue_dists = np.linalg.norm(blue_cones - np.array(point), axis=1)
            nearest_blue = blue_cones[np.argmin(blue_dists)]
            
            # Blue cone should be on the LEFT (positive side)
            blue_side = point_side_of_line(nearest_blue, prev_point, point)
            if blue_side < 0:  # Blue is on right - INVALID
                return False
        elif len(yellow_cones) > 0:
            yellow_dists = np.linalg.norm(yellow_cones - np.array(point), axis=1)
            nearest_yellow = yellow_cones[np.argmin(yellow_dists)]
            
            # Yellow cone should be on the RIGHT (negative side)
            yellow_side = point_side_of_line(nearest_yellow, prev_point, point)
            if yellow_side > 0:  # Yellow is on left - INVALID
                return False
        
        return True

    def check_validity(self, A, B):
        for i in self.cones:
            dist = self.point_segment_distance((i.x * 10, i.y * 10), A, B)
            if dist < 2:
                return False

        for j in range(len(self.cones)):
            for k in range(j + 1, len(self.cones)):
                if self.segments_intersect(A, B, (self.cones[j].x * 10, self.cones[j].y * 10), (self.cones[k].x * 10, self.cones[k].y * 10)) and self.cones[j].color == self.cones[k].color:
                    return False
        
        yellow_cones = np.array([[cone.x * 10, cone.y * 10] for cone in self.cones if cone.color == 0])
        blue_cones = np.array([[cone.x * 10, cone.y * 10] for cone in self.cones if cone.color == 1])
        
        if not self.is_valid_cone_position(B, A, yellow_cones, blue_cones):
           return False
        
        return True
    
    def dijkstra(self, start = (0, 0), goal=(50, 50), min_step=1, max_step=5):
        path: Path2D = []

        pq = [(0, start)]
        visited = {}
        parent = {}

        while pq:
            cost, current = heapq.heappop(pq)
            if current in visited and visited[current] < cost:
                continue

            if math.dist(current, goal) <= 2 and self.check_validity(current, goal) == True:
                if goal != current: 
                    parent[goal] = current
                break

            # generate neighbors
            for dx in np.arange(-max_step, max_step + min_step, min_step):
                for dy in np.arange(-max_step, max_step + min_step, min_step):
                    if (dx == 0 and dy == 0) or math.dist((0, 0), (dx, dy)) >= max_step:
                        continue
                    new_point = (current[0] + dx, current[1] + dy)
                    if(new_point[0] <= -5 or new_point[0] >= 55 or new_point[1] <= -5 or new_point[1] >= 55):
                        continue

                    if not self.check_validity(current, new_point):
                        continue

                    new_cost = cost + math.dist(current, new_point)
                    if (new_point not in visited) or new_cost < visited[new_point]:
                        parent[new_point] = current
                        visited[new_point] = new_cost
                        heapq.heappush(pq, (new_cost, new_point))

        path = [goal]
        while path[-1] != start:
            print("reconstructing...")
            print(path[-1])
            if path[-1] not in parent:
                print("Broken path reconstruction!")
                return []
            path.append(parent[path[-1]])
        path.reverse()

        return path

    def generatePath(self) -> Path2D:
        """Return a list of path points (x, y) in world frame.

        Requirements and notes:
        - Cones: color==0 (yellow) are on the RIGHT of the track; color==1 (blue) are on the LEFT.
        - You may be given 2, 1, or 0 cones on each side.
        - Use the car pose (x, y, yaw) to seed your path direction if needed.
        - Return a drivable path that stays between left (blue) and right (yellow) cones.
        - The returned path will be visualized by PathTester.

        The path can contain as many points as you like, but it should be between 5-10 meters,
        with a step size <= 0.5. Units are meters.

        Replace the placeholder implementation below with your algorithm.
        """
        path: Path2D = self.dijkstra(start=(0, 0), goal=(50, 50))
        for i in range(len(path)):
            path[i] = (path[i][0] / 10, path[i][1] / 10)
        return path