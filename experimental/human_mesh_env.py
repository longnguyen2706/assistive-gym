from assistive_gym.envs.agents.human_mesh import HumanMesh
from assistive_gym.envs.agents.sawyer import Sawyer
from assistive_gym.envs.env import AssistiveEnv
import numpy as np
import matplotlib.pyplot as plt

env = AssistiveEnv(robot=Sawyer('right'), human=HumanMesh())
# env.set_seed(100)
# Setup a camera in the environment to capture images (for rendering)
env.setup_camera(camera_eye=[0.5, -1, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
env.reset()

# Build the environment with a wheelchair
env.build_assistive_env('wheelchair')
# Change the human to a mesh model
# env.unwrapped.human = HumanMesh()

# Move the robot
env.robot.set_base_pos_orient([0, 1.5, 0.975], [0, 0, 0])

# Define the initial pose for the human mesh model
joint_angles = [(env.human.j_left_hip_x, -90), (env.human.j_right_hip_x, -90),
                (env.human.j_left_knee_x, 70), (env.human.j_right_knee_x, 70),
                (env.human.j_left_shoulder_z, -60), (env.human.j_right_shoulder_z, 60),
                (env.human.j_left_elbow_y, -90), (env.human.j_right_elbow_y, 90)]
# Add randomization to the waist and head joint poses
joint_angles += [(j, env.np_random.uniform(-10, 10)) for j in (env.human.j_waist_x, env.human.j_waist_y, env.human.j_waist_z,
                                                               env.human.j_lower_neck_x, env.human.j_lower_neck_y, env.human.j_lower_neck_z,
                                                               env.human.j_upper_neck_x, env.human.j_upper_neck_y, env.human.j_upper_neck_z)]
# Pick a random height (in meters) and body shape parameters for the person
human_height = env.np_random.uniform(1.5, 1.9)
body_shape = env.np_random.uniform(-2, 5, (1, env.human.num_body_shape))
# Create the human mesh (obj) in the environment
env.human.init(env.directory, env.id, env.np_random, gender='random', height=human_height,
               body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0])
# Place human in chair
chair_seat_position = np.array([0.05, 0.05, 0.6])
env.human.set_base_pos_orient(env.furniture.get_base_pos_orient()[0] + chair_seat_position - env.human.get_vertex_positions(env.human.bottom_index), [0, 0, 0, 1])

img, depth = env.get_camera_image_depth()
plt.imshow(img)
plt.show()