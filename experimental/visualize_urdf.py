import pybullet as p
import pybullet_data
def change_color(id_robot, color):
    r"""
    Change the color of a robot.

    :param id_robot: Robot id.
    :param color: Vector4 for rgba.
    """
    for j in range(p.getNumJoints(id_robot)):
        print (p.getJointInfo(id_robot, j))
        p.changeVisualShape(id_robot, j, rgbaColor=color)

# Start the simulation engine
physicsClient = p.connect(p.GUI)

# Load the URDF file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("../assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])
robotId = p.loadURDF("../assistive_gym/envs/assets/human/human_pip_simplified.urdf", [0, 0, 0])
change_color(robotId, [1,0,0,1])

# print all the joints
for j in range(p.getNumJoints(robotId)):
    print (p.getJointInfo(robotId, j))

# Set the simulation parameters
p.setGravity(0,0,-10)
p.setTimeStep(1./240.)

# move the robot
# p.setJointMotorControlArray(robotId, [0,1,2,3,4,5,6,7,8,9,10,11,12,13], p.POSITION_CONTROL, targetPositions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0])

# Set the camera view
cameraDistance = 3
cameraYaw = 0
cameraPitch = -30
cameraTargetPosition = [0,0,1]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
while True:
    p.stepSimulation()
# Disconnect from the simulation
p.disconnect()


