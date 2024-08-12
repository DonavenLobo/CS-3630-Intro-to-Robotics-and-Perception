from grid import CozGrid
from particle import Particle
from utils import grid_distance, rotate_point, diff_heading_deg, add_odometry_noise
import setting
import math
import numpy as np

def motion_update(particles, odom, grid):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used for boundary checking
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = [] # List for the updated particles
    dX, dY, dH = odom # Odometry data

    for particle in particles: # Iterate through particles
        x, y, h = particle.xyh # Get pose of particle
        dX_world, dY_world = rotate_point(dX, dY, h) # Calculate dX and dY in world refrence frame

        # Calculate new pose of robot
        x_new = x + dX_world
        y_new = y + dY_world
        h_new = h + dH
        x_new, y_new, h_new = add_odometry_noise([x_new, y_new, h_new], setting.ODOM_HEAD_SIGMA, setting.ODOM_TRANS_SIGMA) # Factor in noise

        updated_particle = Particle(x_new, y_new, h_new) # Create an updated particle
        motion_particles.append(updated_particle) # Add updated particle to list

    return motion_particles

# ------------------------------------------------------------------------
# cd "OneDrive/Desktop/Gatech/CompSci/CS 3630/LAB3/Lab3_Release"
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    RANDOM_SAMPLE_SIZE = 300
    measured_particles = []
    
    # Update probs 
    particles_with_prob = []
    weight_of_particles = []
    normalize_factor = 0.0
    for particle in particles:
        if not grid.is_free(particle.x, particle.y) or not grid.is_in(particle.x, particle.y):
            particles_with_prob.append(particle)
            weight_of_particles.append(0.0)
        else:
            expected_marker_list = particle.read_markers(grid)
            prob = get_probability(expected_marker_list, measured_marker_list)
            particles_with_prob.append(particle)
            weight_of_particles.append(prob)
            normalize_factor += prob

    # Normalize particles
    prob_of_particles = []
    if normalize_factor == 0.0:
        prob_of_particles = [1 / len(weight_of_particles) for weight in weight_of_particles]
    else:
        prob_of_particles = [weight / normalize_factor for weight in weight_of_particles]

    # Resample
    measured_particles = np.random.choice(particles_with_prob, size=len(particles_with_prob) - RANDOM_SAMPLE_SIZE, replace=True, p=prob_of_particles)
    measured_particles = measured_particles.tolist()

    random_particles = Particle.create_random(count=RANDOM_SAMPLE_SIZE, grid=grid)

    return measured_particles + random_particles

def get_probability(expected_markers, measured_markers):
    
    marker_pairs = [] # closest robot and particle marker pairs
    for robot_marker in measured_markers: 
        min_dist = float('inf')
        if len(expected_markers) > 0:
            closest_robot_marker = robot_marker
            closest_particle_marker = expected_markers[0]
            for particle_marker in expected_markers:
                dist = grid_distance(robot_marker[0], robot_marker[1], particle_marker[0], particle_marker[1])
                if dist < min_dist:
                    closest_robot_marker = robot_marker
                    closest_particle_marker = particle_marker
                    min_dist = dist
            marker_pairs.append((closest_robot_marker, closest_particle_marker))
            expected_markers.remove(closest_particle_marker)

    prob = 1.0
    for robot_marker, particle_marker in marker_pairs:
        distance = grid_distance(robot_marker[0], robot_marker[1], particle_marker[0], particle_marker[1])
        angle = diff_heading_deg(robot_marker[2], particle_marker[2])

        expr1 = (distance ** 2) / (2 * (setting.MARKER_TRANS_SIGMA ** 2))
        expr2 = (angle ** 2) / (2 * (setting.MARKER_ROT_SIGMA ** 2))
        prob_match = math.exp(-1 * (expr1 + expr2))

        prob_no_match = setting.SPURIOUS_DETECTION_RATE * setting.DETECTION_FAILURE_RATE
        prob *= max((prob_match, prob_no_match))

    prob *= setting.SPURIOUS_DETECTION_RATE ** (len(measured_markers) - len(marker_pairs))
    prob *= setting.DETECTION_FAILURE_RATE ** len(expected_markers)

    return prob

 


