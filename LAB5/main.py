from copy import deepcopy
from matplotlib import patches
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import argparse
import numpy as np
import random
import sys

from pid_controller import PidController

class Project5():
    def __init__(self, args):
        self.args = args
        self.max_delta_vel = .002 #max delta that the linear velocity can be in a single timestep
        self.max_delta_omega = .4
        self.standard_deviation = .01 #standard deviation of the angular velocity if the noise isn't 0

    #To draw the curved road
    def cubic_bezier(self,t, p0, p1, p2, p3):
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    
    #To check if the robot's distance from the nearest point on the road exceeds given threshold
    def line_center_check(self,x,road_points):
        nearest_road_distance=np.min([np.linalg.norm(x-road_point) for road_point in road_points])
        if abs(nearest_road_distance) > 0.025: #change threshold accordingly
            return True
        else:
            return False
        
    def drive(self, angular_noise=0, wind=False, curve=False, start_angle=0,display_text=""):
        controller = PidController()
        goal_points = np.array([1.4,0])
        y_init = 0
        if curve:
            y_init = -.9
        init_conditions = np.array(np.mat(f'{random.random()/3 - 1.5}; {y_init}; {start_angle}'))
        r = robotarium.Robotarium(number_of_robots=1, show_figure=self.args.visualize, initial_conditions=init_conditions, sim_in_real_time=self.args.real_time)
        r.axes.add_patch(patches.Rectangle( xy=(-1.5, -1), width=3, height=2, facecolor='#051650', zorder = -2)) #Ocean
        r.axes.text(0, 0.9, display_text, fontsize=14, color="white",ha="center")
        waypoints=[]
        road_points = []
        if not curve:
            r.axes.add_patch(patches.Rectangle( xy=(-1.5, -.06), width=2.85, height=.12, facecolor='#86d6d8', zorder = -1)) #Icy road      
            r.axes.add_patch(patches.Rectangle( xy=(1.35, -.06), width=.1, height=.12, facecolor='red', zorder = 0)) #Goal zone
        else:
            plt.plot([-1.425, 0.1], [-0.9, -0.9], color='#86d6d8', linewidth=24, zorder=-1)
            #Points for curved road
            control_points = np.array([[0.1, -0.9], [0.6, -0.9], [1, -0.9], [1.2, 0.2]])
            
            t_values = np.linspace(0, 1, 100)
            road_points = np.array([self.cubic_bezier(t, *control_points) for t in t_values])
            plt.plot(road_points[:, 0], road_points[:, 1], color='#86d6d8', linewidth=24, zorder=-1)  #curved road

            new_road_points = []
            for roadpoint in road_points:
                if  np.all(roadpoint > control_points[0]):
                    new_road_points.append(roadpoint)

            hop_step=10
            waypoints = new_road_points[::hop_step]
            waypoints.append([1.2, 0.22])
            goal_points = waypoints[-1]

            # Calculate the position for the new goal zone at the tip of the road
            goal_x = waypoints[-1][0] - 0.09 
            goal_y = waypoints[-1][1] - 0.036
            r2=patches.Rectangle(xy=(goal_x, goal_y), width=0.175, height=0.1, facecolor='red', zorder=0, angle=-8)
            r.axes.add_patch(r2)

            for point in waypoints:
                r.axes.add_patch(patches.Circle( xy=point, radius=0.01, color='black'))
            

        x = r.get_poses()
        r.step()
        prev_vels = []
        counter = 0
        waypoints_i = 0
        while(True):
            x = np.transpose(r.get_poses())[0]
            if not curve and (abs(x[1]) > .02 or x[0] > 1.45) or\
                curve and (x[0] < .1 and abs(x[1]+.9) > .02 or x[0] >= .1 and self.line_center_check(x[:2], road_points)):
                
                print("ROBOT OFF ROAD. EXITING")
                return 0, counter
            
            if (curve):
                if waypoints_i < len(waypoints):
                    r.axes.add_patch(patches.Circle( xy=waypoints[waypoints_i], radius=0.01, color='green'))
                    vels = (controller.set_velocity(x, waypoints[waypoints_i], waypoints[-1])).astype('float') #get their desired velocity
                else:
                     vels = (controller.set_velocity(x, waypoints[-1], waypoints[-1])).astype('float') #get their desired velocity                   
            else:
                vels = (controller.set_velocity(x, goal_points, goal_points)).astype('float') #get their desired velocity
            
            #Limits linear velocity to 20cm/second
            if vels[0] > .2:
                vels[0] = .2
            elif vels[0] < -.2:
                vels[0] = -.2

            #limits angular velocity to 3.6
            if vels[1] > 3.6:
                vels[1] = 3.6
            elif vels[1] < -3.6:
                vels[1] = -3.6

            if len(prev_vels) == 0: #if first loop, set prev vel to desired velocity
                prev_vels = vels
            
            #makes sure change in velocity isn't greater than max_delta
            if abs(vels[0] - prev_vels[0]) > self.max_delta_vel:
                vels[0] = prev_vels[0] + self.max_delta_vel * np.sign(vels[0] - prev_vels[0])

            if abs(vels[1] - prev_vels[1]) > self.max_delta_omega:
                vels[1] = prev_vels[1] + self.max_delta_omega * np.sign(vels[1] - prev_vels[1])

            vels[0] += np.random.normal(0, .002) #Add noise to the linear velocity
            if vels[0] > .2:
                vels[0] = .2
            elif vels[0] < -.2:
                vels[0] = -.2

            if angular_noise != 0:
                if counter % 10 == 0:
                    curr_noise = np.random.normal(angular_noise, self.standard_deviation)
                    wind_noise = 0

                if wind and counter % 50 == 0 and counter > 0:
                    wind_noise = self.get_noise(.3,.2)

                #print(curr_noise, wind_noise, curr_noise+wind_noise)
                if abs(curr_noise + wind_noise) < .44:
                    vels[1] += curr_noise + wind_noise
                else:
                    vels[1] += .44 * (-1 if curr_noise + wind_noise < 0 else 1)
            counter+=1

            #limits angular velocity to 3.6 even after noise is added
            if vels[1] > 3.6:
                vels[1] = 3.6
            elif vels[1] < -3.6:
                vels[1] = -3.6

            prev_vels = deepcopy(vels)
            r.set_velocities([0], np.transpose([vels]))

            if(waypoints_i < len(waypoints) and np.linalg.norm(x[:2]-waypoints[waypoints_i]) < .05):
                #print("WAYPOINT {} REACHED".format(waypoints[waypoints_i]))
                waypoints_i+=1
          
            r.step()
            if abs(vels[0]) < .005:
                r.call_at_scripts_end() #TODO: does this go here or at the end?
                if abs(np.linalg.norm(x[:2]-goal_points)) < .05:
                    return 1, counter
                else:
                    return 0, counter  

    def get_noise(self, max_noise, min_noise):
        angular_noise = random.random() * max_noise * 2 - max_noise
        if angular_noise < 0 and angular_noise > -min_noise:
            angular_noise = -min_noise
        elif angular_noise > 0 and angular_noise < min_noise:
            angular_noise = min_noise
        return angular_noise

    def run(self):
        results = {}
        successes = 0
        steps = []

        if self.args.scenario == 'all' or self.args.scenario == '1':
            results['1'] = self.drive(0,display_text="Scenario 1: Straight Road without Noise")
    
        if self.args.scenario == 'all' or self.args.scenario == '2':
            results['2'] = self.drive(self.get_noise(.3, .2),display_text="Scenario 2: Straight Road constant Noise")

        if self.args.scenario == 'all' or self.args.scenario == '3':
            results['3'] = self.drive(self.get_noise(.3, .2), wind=True,display_text="Scenario 3: Straight Road with Constant Noise + Wind")

        if self.args.scenario == 'all' or self.args.scenario == '4':
            results['4'] = self.drive(0, start_angle=(.81 * (1 if random.random() < .5 else -1)),display_text="Scenario 4: Straight Road at a Starting Angle")

        if self.args.scenario == 'all' or self.args.scenario == '5':
            results['5'] = self.drive(0, curve=True,display_text="Scenario 5: Curved Road")

        if self.args.scenario == 'all' or self.args.scenario == '6':
            results['6'] = self.drive(self.get_noise(.23, .18), curve=True,display_text="Scenario 6: Curved Road with Constant Noise")   

        print("Summary:")
        for scenario, result in results.items():
            successes += result[0]
            steps.append(result[1])
            result_string = "PASSED" if result[0] else "FAILED"
            print(f"Scenario {scenario} outcome: {result_string} in {steps[-1]} steps") 
        
        return successes, steps

def create_parser():
    parser = argparse.ArgumentParser(description='Run the Robatarium simulator using your PID controller :)')
    parser.add_argument('--scenario', '-s', default='all', type=str, help='The scenario to run, "all" to run all scenarios')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the scenario')
    parser.add_argument('--real_time', '-r', action='store_true', help='Run sim at same speed as the real robot would run')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()

    p5 = Project5(args)
    successes, steps = p5.run()
    print(f'{successes} scenarios completed successfully')
