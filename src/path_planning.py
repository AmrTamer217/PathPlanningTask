from __future__ import annotations

from typing import List

from src.models import CarPose, Cone, Path2D

import numpy as np

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

    def point_segment_distance(P, A, B):
        P, A, B = np.array(P), np.array(A), np.array(B)
        AB = B - A
        AP = P - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = np.clip(t, 0, 1)  # clamp to [0,1]
        C = A + t * AB
        return np.linalg.norm(P - C)
    
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # 1 = clockwise, 2 = counterclockwise

    def segments_intersect(A, B, C, D):
        o1 = orientation(A, B, C)
        o2 = orientation(A, B, D)
        o3 = orientation(C, D, A)
        o4 = orientation(C, D, B)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special collinear cases
        if o1 == 0 and on_segment(A, C, B): return True
        if o2 == 0 and on_segment(A, D, B): return True
        if o3 == 0 and on_segment(C, A, D): return True
        if o4 == 0 and on_segment(C, B, D): return True

        return False

    def check_validity(A, B):
        for i in self.cones:
            dist = self.point_segment_distance((i.x, i.y), A, B)
            if dist < 0.15:
                return False

        for j in range(len(self.cones)):
            for k in range(j + 1, len(self.cones)):
                if self.segments_intersect(A, B, (self.cones[j].x, self.cones[j].y), (self.cones[k].x, self.cones[k].y)) and self.cones[j].color == self.cones[k].color:
                    return False
        return True

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

        # Default: produce a short straight-ahead path from the current pose.
        # delete/replace this with your own algorithm.
        num_points = 25
        step = 0.5
        cx = self.car_pose.x
        cy = self.car_pose.y
        import math

        path: Path2D = []
        for i in range(1, num_points + 1):
            dx = math.cos(self.car_pose.yaw) * step * i
            dy = math.sin(self.car_pose.yaw) * step * i
            path.append((cx + dx, cy + dy))

        return path
