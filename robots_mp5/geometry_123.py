# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""
import math
import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        for wall in walls:
            center_point = alien.get_centroid()
            dist = alien.get_width()
            startx, starty, endx, endy = wall
            wall_line = ((startx, starty), (endx, endy))
            if point_segment_distance(center_point, wall_line) < dist:
                return True
        return False
    else:
        for wall in walls:
            head_and_tail = alien.get_head_and_tail()
            dist = alien.get_width()
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            if segment_distance(head_and_tail, wall_segment) < dist:
                return True
        return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window."""
    width, height = window
    alien_width = alien.get_width()

    if alien.is_circle():
        # For circular aliens, check distance from the centroid to the edges
        x, y = alien.get_centroid()
        if x - alien_width < 0 or x + alien_width > width or y - alien_width < 0 or y + alien_width > height:
            return False
    else:
        # For non-circular aliens, check both the head and tail positions including alien width
        head, tail = alien.get_head_and_tail()

        if alien.get_shape() == 'Horizontal':
            # For horizontal aliens, the width contributes vertically
            # Check if the body with its width is within window bounds vertically and horizontally
            if (head[0] + alien_width > width or tail[0] - alien_width < 0 or 
                head[1] - alien_width < 0 or head[1] + alien_width > height):
                return False

        elif alien.get_shape() == 'Vertical':
            # For vertical aliens, the width contributes horizontally
            if (head[1] > height or tail[1] < 0 or
                head[0] - alien_width < 0 or head[0] + alien_width > width):
                return False

    return True


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    def is_all_zero(list):
        for num in list:
            if num != 0:
                return False
        return True
    polygon_vectors = [(polygon[i][0] - polygon[i-1][0], polygon[i][1] - polygon[i-1][1]) for i in range(4)]
    
    positive = 0
    negative = 0
    for i in range(4):
        point2head_vector = (polygon[i-1][0] - point[0], polygon[i-1][1] - point[1])
        determinant = point2head_vector[0] * polygon_vectors[i][1] - point2head_vector[1] * polygon_vectors[i][0]
        if determinant > 0:
            positive += 1
        elif determinant < 0:
            negative += 1
    if positive == 0 and negative == 0:
        
        dx_list = [point[0] - polygon[i-1][0] for i in range(4)]
        dy_list = [point[1] - polygon[i-1][1] for i in range(4)]
        dx_sign_sum = sum([1 if num > 0 else -1 if num < 0 else 0 for num in dx_list])
        dy_sign_sum = sum([1 if num > 0 else -1 if num < 0 else 0 for num in dy_list])
        if ((dx_sign_sum == 4 or dx_sign_sum == -4) and (dy_sign_sum == 4 or dy_sign_sum == -4 or is_all_zero(dy_list)))\
              or ((dy_sign_sum == 4 or dy_sign_sum == -4) and (dx_sign_sum == 4 or dx_sign_sum == -4 or is_all_zero(dx_list))):
            return False
        else:
            return True
    elif positive == 0 or negative == 0:
        return True
    else:
        return False
       


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    alien_x, alien_y = alien.get_centroid()
    move_dir = (waypoint[0] - alien_x, waypoint[1] - alien_y)
    if alien.is_circle():
        move_line = ((alien_x, alien_y), (waypoint[0], waypoint[1]))
        for wall in walls:
            end_pointA_x, end_pointA_y, end_pointB_x, end_pointB_y = wall
            wall_segment = ((end_pointA_x, end_pointA_y), (end_pointB_x, end_pointB_y))
            if segment_distance(move_line, wall_segment) < alien.get_width():
                return True
    else:
        (startx, starty), (endx, endy) = alien.get_head_and_tail()
        new_startx = startx + move_dir[0]
        new_starty = starty + move_dir[1]
        new_endx = endx + move_dir[0]
        new_endy = endy + move_dir[1]
        polygon_segments = (((startx, starty), (new_startx, new_starty)), ((new_startx, new_starty), (new_endx, new_endy))
                   , ((new_endx, new_endy), (endx, endy)), ((endx, endy), (startx, starty)))
        polygon = ((startx, starty), (new_startx, new_starty), (new_endx, new_endy), (endx, endy))
        for wall in walls:
            end_pointA_x, end_pointA_y, end_pointB_x, end_pointB_y = wall
            wall_segment = ((end_pointA_x, end_pointA_y), (end_pointB_x, end_pointB_y))
            for polygon_segment in polygon_segments:
                if segment_distance(wall_segment, polygon_segment) < alien.get_width():
                    return True
            if is_point_in_polygon(wall_segment[0], polygon) or is_point_in_polygon(wall_segment[1], polygon):
                return True
            
    return False



def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x1, y1 = s[0]
    x2, y2 = s[1]

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if squared_length == 0:
        return euclidean_distance(p, s[0])
    t = ((p[0] - x1) * (x2 - x1) + (p[1] - y1) * (y2 - y1)) / squared_length
    t = max(0, min(1, t))
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    dist = euclidean_distance(p, (closest_x, closest_y))

    return dist


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    if min(point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), 
           point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1)) < 1e-10:
        return True
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]
    
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    determinant = dx1 * dy2 - dx2 * dy1
    if abs(determinant) < 1e-10:
        return False
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / determinant
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / determinant
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return True
    else:
        return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        return min(point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), 
                   point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1))


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        print(f'Alien config: {config}')
        print(f'Expected within window: {truths[2]}, Actual: {in_window_result}')
        print(f'Alien width: {alien.get_width()}, Window size: {window}')

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'

    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    print("Made it halfway!")

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    print("Made it to edge cases!")

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
