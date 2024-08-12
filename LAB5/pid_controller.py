import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

class PidController():
    def __init__(self):
        self.linear_kp = 0.5
        self.linear_ki = 0
        self.linear_kd = 0

        self.angular_kp = 12
        self.angular_ki = 0.3
        self.angular_kd = 8

        # Initialize terms + set parameters
        self.linear_error_sum = 0
        self.linear_previous_error = 0
        self.angular_error_sum = 0
        self.angular_previous_error = 0

        self.braking_threshold = 0.05 # Distance at which to brake

    def linear_controller(self, pose, goal_point):
        """
        Set the linear velocity based on the robot's current pose and goal_point.
        """
        # Calculate the linear distance to the goal
        linear_error = np.linalg.norm(goal_point - pose[:2])
        PID_p = self.linear_kp * linear_error
        
        # Calculate integral of error
        self.linear_error_sum += linear_error
        PID_i = self.linear_ki * self.linear_error_sum
        
        # Calculate derivative of error
        linear_derivative = linear_error - self.linear_previous_error
        self.linear_previous_error = linear_error
        PID_d = self.linear_kd * linear_derivative

        # Check if within braking threshold
        if linear_error < self.braking_threshold:
            PID_p = 0

        linear_velocity = PID_p + PID_i + PID_d
        return linear_velocity

    def angular_controller(self, pose, waypoint):
        """
        Set the angular velocity based on the robot's current pose and next waypoint.
        """
        # Calculate the angle to the waypoint
        angle_to_waypoint = np.arctan2(waypoint[1] - pose[1], waypoint[0] - pose[0])
        angular_error = angle_to_waypoint - pose[2]
        
        # Normalize the angular error to the range [-pi, pi]
        angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi
        PID_p = self.angular_kp * angular_error
        
        # Calculate integral of error
        self.angular_error_sum += angular_error
        PID_i = self.angular_ki * self.angular_error_sum
        
        # Calculate derivative of error
        angular_derivative = angular_error - self.angular_previous_error
        self.angular_previous_error = angular_error
        PID_d = self.angular_kd * angular_derivative
        
        # PID output for angular velocity
        angular_velocity = PID_p + PID_i + PID_d
        
        return angular_velocity
    
    def set_velocity(self, pose, waypoint, goal_point):
        """
        Set the linear and angular velocities based on the robot's current pose, next waypoint, 
        and goal pose.
        
        Parameters:
            pose (np.array): Current pose (x, y, theta)
            waypoint (np.array): Next waypoint pose to navigate to (x, y)
            goal_point (np.array): Goal pose at the end of the trajectory (x, y)

        Returns: np.array([linear_velocity, angular_velocity])
        """

        linear_velocity = self.linear_controller(pose, goal_point)
        angular_velocity = self.angular_controller(pose, waypoint)
        
        return np.array([linear_velocity, angular_velocity])
    
    
    