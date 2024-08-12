from grid import *
from robot import *
import time
import math
from utils import *


def get_wheel_velocities(robbie, coord, pickup_marker=False, threshold=0.1):
    """
    Helper function to determine the velocities of the robot's left and right wheels.
    Arguments:
        robbie: instance of the robot
        coord (tuple): coordinate to move to (x,y)
        pickup_marker (bool): Only set to 'True' when picking up marker
        threshold (int): Set to expected heading when trying to align robot with marker
    
    Returns: 
        vr, vl: velocities of the robot's left and right wheels
    """

    # Calculate the desired change in position
    controller = PidController()
    dx_world = coord[0] - robbie.x
    dy_world = coord[1] - robbie.y
    dx_robot, dy_robot = rotate_point(dx_world, dy_world, robbie.h)
    
    # Turn in place
    if not pickup_marker:
        angle = math.atan2(dy_robot, dx_robot)
        if angle < -threshold:
            return -0.02, 0.02
        elif angle > threshold:
            return 0.02, -0.02
    else:
        angle = robbie.h
        if angle < threshold:
            return 0.01, -0.01
        elif angle > threshold:
            return -0.01, 0.01
    
    
    robot_pose = np.array([robbie.xyh[0], robbie.xyh[1], robbie.xyh[2]])
    goalpoint = np.array([coord[0], coord[1]])
    linear_v = controller.linear_controller(robot_pose, goalpoint)
    w = controller.angular_controller(robot_pose, goalpoint) 
    vl = (linear_v - robbie.wheel_dist / 2 * w) 
    vr = (linear_v + robbie.wheel_dist / 2 * w)    
    return vr, vl


def phase2_planning(robbie,grid):
    """
    This function should move the robot from it's starting position to a marker and then 'pick up' the marker.
    Arguments:
        robbie: instance of robot class
        grid: instance of grid class
    Returns:
        robbie: 'updated' instance of robot class
    Notes:
        Markers for each grid can be accessed through grid.markers
        Sample Pseudocode (this is just to give you an idea of how to implement this function. It MAY NOT be a complete solution):
        1. Move the robot from its current position to a marker on the grid. Use the 'get_wheel_velocities' function to determine the robot's velocities. 
           Note that the 'get_wheel_velocities' function relies on your PIDController implementation in utils.py.
           You may use the 'rrt' function (see grid.py) when the robot encounters an obstacle.
        2. When the robot reaches a marker, it must orient itself in the same orientation as the marker so as to 'pick it up'.
           For example if the marker's orientation is 'R', once the robot gets to the marker, it should turn in place till it's heading is 0 degrees.
           The 'get_wheel_velocities' function may be used to accomplish this. Note that you must set the 'pickup_marker' variable to 'True' when calling it. 
           Threshold would also need to be set to the appropriate heading for each marker.
           The expected heading for each of the markers can be accessed by calling the 'parse_marker_info' function in grid.py
        3. You may keep track of rrt path (if using rrt) by storing in the 'path' function member and current marker by storing in the 'curr_marker' function member
           in the robot class (check robot.py).

    Alert:
        In this part, the robot is expected to 'pick up' all markers by going the desired locations.
        You may call 'grid.markers' to get the markers' coordinates. 
        However, modifying elements in 'robot.markers_found_or_picked' is prohibited.

    """

    all_markers = grid.markers

    robbie.curr_marker = find_closest_marker(robbie, all_markers)

    # Get information about the current marker
    marker_info = robbie.curr_marker
    marker_x, marker_y, marker_orientation = parse_marker_info(*marker_info)
    marker_pos = (marker_x, marker_y)
    robbie.next_coord = marker_pos
    
    if not robbie.marker_already_picked(robbie.curr_marker) and not reached_target(robbie, robbie.next_coord):
        print(f"Moving towards marker: {marker_info}, Current position: {robbie.xy}")
        
        robbie.next_coord = marker_pos

        if grid.is_collision_with_obstacles(robbie.xy, robbie.next_coord):
            if robbie.path and not reached_target(robbie, robbie.path[1]):
                print('Still redirecting')
                robbie.next_coord = robbie.path[1]
            else:
                print('Obstacle detected. Re-planning path...')
                robbie.path = grid.rrt(robbie.xy, marker_pos)
                robbie.next_coord = robbie.path[1]
        
        robbie.vr, robbie.vl = get_wheel_velocities(robbie, robbie.next_coord)  

    elif not robbie.marker_already_picked(robbie.curr_marker):
        print(f"Aligning with marker. Current heading: {robbie.h}, Expected: {marker_orientation}")
        robbie.vr, robbie.vl = 0.2,-0.2

    else:
        # "Pick up" the marker
        
        print(f"Picked up marker at {marker_pos} facing {marker_orientation}")
        next_marker_index = all_markers.index(robbie.curr_marker) + 1
        robbie.curr_marker = find_closest_marker(robbie, all_markers)

        
    print(f"Robbie Next Coord: {robbie.next_coord}")
    return robbie

def reached_target(robbie, next_coord):
    target_reached_threshold = 0.1  # Define how close the robot needs to be to the target

    if not next_coord:
        return False
    dist = grid_distance(robbie.x, robbie.y, next_coord[0], next_coord[1])
    print(f"Distance to target: {dist}, Threshold: {target_reached_threshold}")
    
    return dist < target_reached_threshold

def find_closest_marker(robbie, all_markers):
    closest_marker = None
    min_distance = float('inf')
    
    for marker in all_markers:
        if not robbie.marker_already_picked(marker):
            marker_x, marker_y, _ = parse_marker_info(*marker)
            distance = grid_distance(robbie.x, robbie.y, marker_x, marker_y)
            
            if distance < min_distance:
                closest_marker = marker
                min_distance = distance

    return closest_marker


