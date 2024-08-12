import sys
import time
from map import Map
from gui import *
from utils import *
from robot_sim import *

MAX_NODES = 20000

def RRT(map):
    """ Builds RRT given a map
    """
    map.add_node(map.get_start())
    map_width, map_height = map.get_size()
    while (map.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use Map.node_generator() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node using map.add_path
        # Note: you do not need to call the add_node function, since add_path
        #       will add the node to the map as well.
        
        rand_node = map.node_generator()
        nearest_node = None
        min_dist = 99999
        for node in map.get_nodes():
            dist = get_dist(node, rand_node)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        limited_node = map.step_from_to(nearest_node, rand_node)
        map.add_path(nearest_node, limited_node)
        ########################################################################
        time.sleep(0.01)
        
        if map.is_solved():
            break

    path = map.get_path()
    smoothed_path = map.get_smooth_path()

    if map.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", map.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")


def robot_planning_with_exploration(robot, map):
    map.check_new_obstacle(robot, DDRobot.VISION_DISTANCE)
    RRT(map)
    path = map.get_smooth_path()


    #while the current robot position is not at the goal:
    while get_dist(robot, map.get_goals()[0]) > 1:
        ########################################################################
        # TODO: please enter your code below.
        # Description of function provided in instructions. Potential pseudcode is below
        
        # Get the next node from the path

        # drive the robot to next node in path. First turn to the appropriate angle, which
        # you can calculate using a trigonometric function

        # while robot has not reached the next node in the path

            # detect any visible obstacles and update cmap

            # if new obstacles are detected, reset the cmap with the current robot location as 
            # the start node, re-plan using RRT to generate a new path

            # otherwise, drive straight towards the next node within vision distance

        # Hint: feel free to use function robot.turn_in_place(), robot.move_forward()
        # map.check_new_obstacle(), map.reset(), etc.

        # Get the next node from the path
        next_node = path.pop(0)
        
        # Drive the robot to the next node in path
        delta_x = next_node.x - robot.x
        delta_y = next_node.y - robot.y
        angle_rad = np.arctan2(delta_y, delta_x)
        
        # First, turn the robot to the appropriate angle
        robot.turn_in_place(angle_rad - robot.theta)
        
        # While the robot has not reached the next node in the path
        while get_dist(Node((robot.x, robot.y)), next_node) > 1:

            # Detect any visible obstacles and update map
            has_obstacle = map.check_new_obstacle(robot, DDRobot.VISION_DISTANCE)
            
            # If new obstacles are detected, reset the map with the current robot location as the start node
            if has_obstacle:
                map.reset(Node((robot.x, robot.y)))
                RRT(map)
                path = map.get_smooth_path()
                next_node = path.pop(0)

                delta_x = next_node.x - robot.x
                delta_y = next_node.y - robot.y
                angle_rad = np.arctan2(delta_y, delta_x)
                robot.turn_in_place(angle_rad - robot.theta)
            
            # Otherwise, drive straight towards the next node within vision distance
            drive_distance = min(DDRobot.VISION_DISTANCE, get_dist(Node((robot.x, robot.y)), next_node))
            robot.move_forward(drive_distance)


class RobotThread(threading.Thread):
    """Thread to run vector code separate from main thread
    """

    def __init__(self, robot, map ):
        threading.Thread.__init__(self, daemon=True)
        self.robot = robot
        self.map = map

    def run(self):
        robot_planning_with_exploration(self.robot, self.map)
        time.sleep(5)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self, map):
        threading.Thread.__init__(self, daemon=True)
        self.map = map

    def run(self):
        RRT(self.map)
        time.sleep(5)
        self.map.reset_paths()
        stopevent.set()


if __name__ == '__main__':
    global stopevent
    stopevent = threading.Event()
    exploration = False
    for i in range(0,len(sys.argv)): 
        #reads input whether we are running the exploration version or not
        if (sys.argv[i] == "-explore"):
            exploration = True

    map = Map("maps/map2.json", exploration)
    
    if exploration:
        r = DDRobot(map.get_start().x, map.get_start().y, map)
        robot_thread = RobotThread(robot=r, map=map)
        visualizer = Visualizer(map, r, stopevent, exploration)
        robot_thread.start()
        visualizer.start()
    else:
        rrt_thread = RRTThread(map=map)
        visualizer = Visualizer(map, None, stopevent, exploration)
        rrt_thread.start()
        visualizer.start()
