from __future__ import annotations

from typing import List

from src.models import CarPose, Cone, Path2D

import numpy as np
import heapq
import math
from scipy.interpolate import splprep, splev

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
    
    def on_segment(self, p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # 1 = clockwise, 2 = counterclockwise

    def segments_intersect(self, A, B, C, D):
        o1 = self.orientation(A, B, C)
        o2 = self.orientation(A, B, D)
        o3 = self.orientation(C, D, A)
        o4 = self.orientation(C, D, B)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special collinear cases
        if o1 == 0 and self.on_segment(A, C, B): return True
        if o2 == 0 and self.on_segment(A, D, B): return True
        if o3 == 0 and self.on_segment(C, A, D): return True
        if o4 == 0 and self.on_segment(C, B, D): return True

        return False

    def check_validity(self, A, B):
        for i in self.cones:
            dist = self.point_segment_distance((i.x, i.y), A, B)
            if dist < 0.15:
                return False

        for j in range(len(self.cones)):
            for k in range(j + 1, len(self.cones)):
                if self.segments_intersect(A, B, (self.cones[j].x, self.cones[j].y), (self.cones[k].x, self.cones[k].y)) and self.cones[j].color == self.cones[k].color:
                    return False
        return True
    
    def shortcut_smooth(self, path):
        """Remove unnecessary waypoints by checking direct valid shortcuts."""
        if not path:
            return []
        smooth = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.check_validity(path[i], path[j]):
                    break
                j -= 1
            smooth.append(path[j])
            i = j
        return smooth

    def spline_smooth(self, path, num_points=200):
        """Smooth path with cubic spline interpolation (for visualization)."""
        if len(path) < 3:
            return path
        x, y = zip(*path)
        tck, _ = splprep([x, y], s=0.2)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new, y_new))
    
    def dijkstra(self, start = (0, 0), goal=(5, 5), step=0.2, max_step=0.5):
        path: Path2D = []
        
        def round_point(p):
            return (round(p[0], 1), round(p[1], 1))

        pq = [(0, start)]
        visited = {}
        parent = {}

        while pq:
            cost, current = heapq.heappop(pq)
            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost

            if math.dist(current, goal) <= step and self.check_validity(current, goal) == True:
                parent[goal] = current
                break

            # generate neighbors
            for dx in np.arange(-max_step, max_step + step, step):
                for dy in np.arange(-max_step, max_step + step, step):
                    if dx == 0 and dy == 0:
                        continue
                    new_point = (current[0] + dx, current[1] + dy)
                    if math.dist(current, new_point) > max_step:
                        continue

                    new_point = round_point(new_point)
                    if not self.check_validity(current, new_point):
                        continue

                    new_cost = cost + math.dist(current, new_point)
                    if new_point not in visited or new_cost < visited[new_point]:
                        parent[new_point] = current
                        heapq.heappush(pq, (new_cost, new_point))

        path = [goal]
        while path[-1] != start:
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
        path: Path2D = self.dijkstra(start=(self.car_pose.x, self.car_pose.y), goal=(5, 5), step=0.2, max_step=0.5)
        path = self.shortcut_smooth(path)
        path = self.spline_smooth(path, num_points=200)
        return path
