import random
import math
import numpy as np

# Node object for RRT
class Node(object):
    """Class representing a node in RRT
    """

    def __init__(self, coord, parent=None):
        super(Node, self).__init__()
        self.coord = coord    # 2D coordinate of the node in the map
        self.parent = parent  # parent node in the RRT tree

    @property
    def x(self):
        return self.coord[0]

    @property
    def y(self):
        return self.coord[1]
    
    @property
    def xy(self):
        return (self.coord[0], self.coord[1])

    def __getitem__(self, key):
        assert (key == 0 or key == 1)
        return self.coord[key]


class PidController():
    """
    Controller to tune. You may reuse code from project 5 and tune the controller as necessary.
    """
    def __init__(self):
        # TODO: tune the gains for this PID controller
        self.linear_error = 0
        self.previous_linear_error = 0
        self.integral_linear_error = 0

        self.angular_error = 0
        self.integral_angular_error = 0
        self.previous_angular_error = 0
        
        self.linear_kp = 0.05
        self.linear_ki = 0.0
        self.linear_kd = 0.0
        
        self.angular_kp = 0.1
        self.angular_ki = 0
        self.angular_kd = 0

        self.stopped = False

    def compute_angular_error(self, current_pose, next_waypoint):
        x_robot, y_robot, theta_robot = current_pose
        dx_world = next_waypoint[0] - current_pose[0]
        dy_world = next_waypoint[1] - current_pose[1]
        dx_robot, dy_robot = rotate_point(dx_world, dy_world, theta_robot)
        angular_error = math.atan2(dy_robot, dx_robot)

        return angular_error
    
    def compute_linear_error(self, current_pose, goal_point):
        x_robot, y_robot, theta_robot = current_pose
        dx_world = goal_point[0] - current_pose[0]
        dy_world = goal_point[1] - current_pose[1]

        dx_robot, dy_robot = rotate_point(dx_world, dy_world, theta_robot)
        dist_to_coord = math.sqrt(dx_robot**2 + dy_robot**2)

        return dist_to_coord
    
    def linear_controller(self, pose, goal_point):
        """
        Set the linear velocity based on the robot's current pose and goal_point.
        
        Arguments:
            pose (np.array): Current pose (x, y, theta)
            goal_point (np.array): Goal pose at the end of the trajectory (x, y)

        Returns: linear_velocity (float) 
        """
        # Calculate the linear distance to the goal
        self.linear_error = self.compute_linear_error(pose, goal_point)
        PID_p = self.linear_kp * self.linear_error
        
        # Calculate integral of error
        self.integral_linear_error += self.linear_error
        PID_i = self.linear_ki * self.integral_linear_error
        
        # Calculate derivative of error
        linear_derivative = self.linear_error - self.previous_linear_error
        self.previous_linear_error = self.linear_error
        PID_d = self.linear_kd * linear_derivative

        # PID output for linear velocity
        linear_velocity = PID_p + PID_i + PID_d
        return linear_velocity 

    
    def angular_controller(self, pose, waypoint):
        """
        Set the angular velocity based on the robot's current pose and next waypoint.
        
        Arguments:
            pose (np.array): Current pose (x, y, theta)
            waypoint (np.array): Next waypoint pose to navigate to (x, y)

        Returns: angular_velocity (float) 
        """
        #TODO: return the angular velocity based on the robot's current pose and next waypoint.

        # Calculate the angle to the waypoint
        self.angular_error = self.compute_angular_error(pose, waypoint)
        
        # Normalize the angular error to the range [-pi, pi]
        # self.angular_error = (self.angular_error + np.pi) % (2 * np.pi) - np.pi
        PID_p = self.angular_kp * self.angular_error
        
        # Calculate integral of error
        self.integral_angular_error += self.angular_error
        PID_i = self.angular_ki * self.integral_angular_error
        
        # Calculate derivative of error
        angular_derivative = self.angular_error - self.previous_angular_error
        self.previous_angular_error = self.angular_error
        PID_d = self.angular_kd * angular_derivative
        
        # PID output for angular velocity
        angular_velocity = PID_p + PID_i + PID_d
        
        return angular_velocity        



""" Some math utilies, feel free to use any of these!!!
"""

# euclian distance in grid world
def grid_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def grid_node_distance(node_a, node_b):
    return math.sqrt((node_a.x - node_b.x) ** 2 + (node_a.y - node_b.y) ** 2)


# utils for 2d rotation, given frame \theta
def rotate_point(x, y, heading_deg):
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    xr = x * c + y * s
    yr = -x * s + y * c
    return xr, yr


def diff_heading_deg(heading1, heading2):
    """
    Arguments:
        heading1: Angle (degrees)
        heading2: Angle (degrees)

    Returns:
        dh: Difference in heading1 and headin2 in range (-180,180] (degrees)
    """
    dh = heading1 - heading2
    while dh > 180:
        dh -= 360
    while dh <= -180:
        dh += 360
    return dh


def compute_mean_pose(particles, confident_dist=1):
    """ 
    Compute the mean pose for all particles
    This is not part of the particle filter algorithm but rather an
    addition to show the "best belief" for current pose
    
    """
    m_x, m_y, m_count = 0, 0, 0
    # for rotation average
    m_hx, m_hy = 0, 0
    for p in particles:
        m_count += 1
        m_x += p.x
        m_y += p.y
        m_hx += math.sin(math.radians(p.h))
        m_hy += math.cos(math.radians(p.h))

    if m_count == 0:
        return -1, -1, 0, False

    m_x /= m_count
    m_y /= m_count

    # average rotation
    m_hx /= m_count
    m_hy /= m_count
    m_h = math.degrees(math.atan2(m_hx, m_hy))

    # Now compute how good that mean is -- check how many particles
    # actually are in the immediate vicinity
    m_count = 0
    for p in particles:
        if grid_distance(p.x, p.y, m_x, m_y) < 1:
            m_count += 1

    return m_x, m_y, m_h, m_count > len(particles) * 0.95


def find_line(p1, p2):
    """ Find the line that connects two points p1 and p2 in the form y=mx+c
    """
    m = 0 if p2[0] == p1[0] else (p2[1]-p1[1])/(p2[0]-p1[0])
    c = p2[1] - m*p2[0]
    return m, c


def find_dist(m, c, p):
    return abs(m*p[0]-p[1]+c)/math.sqrt(m**2 + 1)


def find_centroid(points):
    """
    Finds centroid of a set of 2D coords
    """
    sum_x = sum([p[0] for p in points])
    sum_y = sum([p[1] for p in points])
    return sum_x/len(points), sum_y/len(points)


def separate_adjacent_coordinates(coordinates , grid):
    """
    Separates out a list of cells into a list of frontiers
    """
    def is_adjacent(coord1, coord2, grid):
        x1, y1 = coord1
        x2, y2 = coord2
        flag = grid.is_free(x1,y2) and grid.is_free(x2,y1)
        return (abs(x1 - x2) <= 1) and (abs(y1 - y2) <= 1) and flag

    def dfs(coord, grid):
        visited.add(coord)
        component.append(coord)
        for neighbor in [c for c in coordinates if c not in visited and is_adjacent(coord, c, grid)]:
            dfs(neighbor,grid)

    visited = set()
    components = []
    for coord in coordinates:
        if coord not in visited:
            component = []
            dfs(coord, grid)
            components.append(component)

    return components