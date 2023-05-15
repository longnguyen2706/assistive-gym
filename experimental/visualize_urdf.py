# TODO: clean up this file
import pickle

import numpy as np
import pybullet as p
import pybullet_data

from pytorch3d import transforms as t3d
import torch

from assistive_gym.envs.utils.human_pip_dict import HumanPipDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict


physicsClient = p.connect(p.GUI)

# Load the URDF file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("../assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])
# robotId = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf", [0, 0, 0], [0, 0, 0,1])
robotId = p.loadURDF("test_mesh.urdf", [0, 0, 0], [0, 0, 0,1])


# print all the joints
for j in range(p.getNumJoints(robotId)):
    print (p.getJointInfo(robotId, j))
# Set the simulation parameters
p.setGravity(0,0,-9.81)
p.setTimeStep(1./240.)

# Set the camera view
cameraDistance = 3
cameraYaw = 0
cameraPitch = -30
cameraTargetPosition = [0,0,1]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
while True:
    # p.stepSimulation()'
    pass
# Disconnect from the simulation
p.disconnect()


