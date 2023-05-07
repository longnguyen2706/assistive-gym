import pybullet as p
import pybullet_data
import os
from time import sleep

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
# print (os.getcwd() + '/examples/')
p.setGravity(0,0,-10)
# load mjc
# planeId = p.loadMJCF('mjmodel.xml')
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId =  p.loadMJCF('examples/humanoid.xml')
print (boxId)
# p.stepSimulation()
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()
sleep(100)