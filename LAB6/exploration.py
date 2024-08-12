from robot import Robot
from utils import *
import math

PATH_HISTORY = []

def get_wheel_velocities(robbie, coord):
    """
    Helper function to determine the velocities of the robot's left and right wheels.
    Arguments:
        robbie: instance of the robot
        coord (tuple): coordinate to move to (x,y)
    
    Returns: 
        vr, vl: velocities of the robot's left and right wheels
    """

    # Calculate the desired change in position
    dx_world = coord[0] - robbie.x
    dy_world = coord[1] - robbie.y
    dx_robot, dy_robot = rotate_point(dx_world, dy_world, robbie.h)
    dist_to_coord = math.sqrt(dx_robot**2 + dy_robot**2)
    
    # Turn in place first
    angle = math.atan2(dy_robot, dx_robot)
    threshold = 0.1
    if angle < -threshold:
        return -0.01, 0.01
    elif angle > threshold:
        return 0.01, -0.01
    
    # Using desired linear velocity, set left and right wheel velocity
    linear_v = 0.05 * dist_to_coord
    w = 0.3 * math.atan2(dy_robot, dx_robot)
    vl = (linear_v - robbie.wheel_dist / 2 * w) 
    vr = (linear_v + robbie.wheel_dist / 2 * w)    
    return vr, vl

def frontier_planning(robbie, grid):
    """
        OPTIONAL: Function for defining frontier planning.

        Arguments:
            robbie: instance of the robot
            grid: instance of the grid

        Returns:
            robbie: 'updated' instance of the robot
            OPTIONAL: robbie.next_coord: new destination coordinate

        Notes:
            The lecture notes should provide you with an ample description of frontier planning.
            You will also find many of the functions declared in 'grid.py' and 'utils.py' useful.

    """
    print('Frontier planning entered')
    frontiers = find_frontiers(robbie, grid) # Identify frontiers
    frontier_clusters = separate_adjacent_coordinates(frontiers, grid) # Group frontiers into clusters
    target_frontier = select_target_frontier(frontier_clusters, robbie) # Choose a target frontier centroid

    return target_frontier


def find_frontiers(robbie, grid):
    frontiers = set()
    
    for cell in robbie.explored_cells:
        x, y = cell
        neighbors = [(x-1, y), (x-1, y-1), (x+1, y), (x+1, y+1), (x, y-1), (x-1, y+1), (x, y+1), (x+1,y-1)]
        
        for n in neighbors:
            # A frontier cell is one that is a neighbor of an explored cell, has not been explored itself, and is not occupied and in the grid.
            if n not in robbie.explored_cells and grid.is_in(*n) and grid.is_free(*n):
                frontiers.add(n)

    return list(frontiers)


def select_target_frontier(frontier_clusters, robbie):

    selected_frontier = None
    largest = 0

    for frontier in frontier_clusters:
        centroid = find_centroid(frontier)
        num = len(frontier)

        if num > largest:
            largest = num
            selected_frontier = centroid

    return selected_frontier


def exploration_state_machine(robbie, grid):
    """
    Use frontier planning, or another exploration algorithm, to explore the grid.

    Arguments:
        robbie: instance of the robot
        grid: instance of the grid

    Returns: 
        robbie: 'updated' instance of the robot

    Notes:
        Robot is considered as Point object located at the center of the traingle.
        You may use the 'rrt' function (see grid.py) to find a new path whenever the robot encounters an obstacle.
        You can use 'grid.is_collision_with_obstacles()' to check if the robot encounters an obstacle.
        Please note that the use of rrt slows down your code, so it should be used sparingly.
        The 'get_wheel_velocities' functions is useful in setting the robot's velocities.
        You will also find many of the functions declared in 'grid.py' and 'utils.py' useful.
        Feel free to create other helper functions (in this file) as necessary.

    Alert:
        In this part, the task is to let the robot find all markers by exploring the map,
        which means using 'grid.markers' will lead  cause zero point on GraderScope.

    """

    if robbie.next_coord is None or reached_target(robbie):
        robbie.next_coord = frontier_planning(robbie, grid)

    # Check if the robot has reached its target or if the path list is empty
    if grid.is_collision_with_obstacles(robbie.xy, robbie.next_coord):
        print('Obstacle detected. Re-planning path...')
        robbie.path = grid.rrt(robbie.xy, robbie.next_coord)
        robbie.next_coord = robbie.path[1]

    # if not reached_target(robbie):
    robbie.vr, robbie.vl = get_wheel_velocities(robbie, robbie.next_coord)
    robbie.move_diff_drive(grid, robbie.vl, robbie.vr, robbie.TIMESTEP)

    return robbie


def reached_target(robbie):
    """
    Check if the robot has reached its target.
    """
    target_reached_threshold = 0.3  # Define how close the robot needs to be to the target
    dist = grid_distance(robbie.x, robbie.y, robbie.next_coord[0], robbie.next_coord[1])
    return dist < target_reached_threshold