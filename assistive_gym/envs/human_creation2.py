import os, colorsys
import pybullet as p
import numpy as np


# -- Joint Legend --

# 0-2 right_shoulder x,y,z
# 3-5 right_shoulder_socket x,y,z
# 6 right_elbow x
# 7 right_forearm_roll z
# 8-9 right_hand x,y
# 10-12 left_shoulder x,y,z
# 13-15 left_shoulder_socket x,y,z
# 16 left_elbow x
# 17 left_forearm_roll z
# 18-19 left_hand x,y
# 20 neck x
# 21-23 head x,y,z
# 25-27 waist x,y,z
# 28-30 right_hip x,y,z
# 31 right_knee x
# 32-34 right_ankle x,y,z
# 35-37 left_hip x,y,z
# 38 left_knee x
# 39-41 left_ankle x,y,z
# -- Limb (link) Legend --

# 2 right_shoulder
# 5 right_upperarm
# 7 right_forearm
# 9 right_hand
# 12 left_shoulder
# 15 left_upperarm
# 17 left_forearm
# 19 left_hand
# 20 neck
# 23 head
# 24 waist
# 27 hips
# 30 right_thigh
# 31 right_shin
# 34 right_foot
# 37 left_thigh
# 38 left_shin
# 41 left_foot

axis = {
    'x': [1, 0 , 0],
    'y': [0, 1, 0],
    'z': [0, 0, 1]
}

DEFAULT_ORIENTATION = [0, np.pi/2.0, 0]

joint_idx = {
    'right_shoulder': 2,
    'right_upperarm': 5,
    'right_forearm': 7,
    'right_hand': 9,
    'left_shoulder': 12,
    'left_upperarm': 15,
    'left_forearm': 17,
    'left_hand': 19,
    'neck': 20,
    'head': 23,
    'waist': 24,
    'hips': 27,
    'right_thigh': 30,
    'right_shin': 31,
    'right_foot': 34,
    'left_thigh': 37,
    'left_shin': 38,
    'left_foot': 41
}
class HumanCreation2:
    def __init__(self, pid=None, np_random=None, cloth=False):
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.cloth = cloth
        self.np_random = np_random
        self.hand_radius = 0.0
        self.elbow_radius = 0.0
        self.shoulder_radius = 1
        self.id = pid

    def create_body(self, shape=p.GEOM_CAPSULE,
                    radius=0, length=0, position_offset=[0, 0, 0], orientation=[0, 0, 0, 1],
                    skin_color = 'random', specular_color=[0.1, 0.1, 0.1]):
        visual_shape = p.createVisualShape(shape, radius=radius, length=length, rgbaColor=skin_color,
                                           specularColor=specular_color, visualFramePosition=position_offset,
                                           visualFrameOrientation=orientation, physicsClientId=self.id)
        collision_shape = p.createCollisionShape(shape, radius=radius, height=length,
                                                 collisionFramePosition=position_offset,
                                                 collisionFrameOrientation=orientation, physicsClientId=self.id)
        return collision_shape, visual_shape

    def create_human(self, static=True, limit_scale=1.0,
                     gender='random', config=None, mass=None, radius_scale=1.0, height_scale=1.0,  skin_color='random', specular_color=[0.1, 0.1, 0.1]):
        if gender not in ['male', 'female']:
            gender = self.np_random.choice(['male', 'female'])
        if skin_color == 'random':
            hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
            hsv[-1] = self.np_random.uniform(0.4, 0.8)
            skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
            # print(hsv, skin_color, colorsys.rgb_to_hls(0.8, 0.6, 0.4))

        joint_c, joint_v = -1, -1
        if gender == 'male':
            print('creating male human')
            if config is not None:
                c = lambda tag: config(tag, 'human_male')
                m = c('mass')  # mass of 50% male in kg
                rs = c('radius_scale')
                hs = c('height_scale')
            else:
                m = mass if mass is not None else 78.4
                rs = radius_scale
                hs = height_scale
            # TODO: see the size
            # smpl#0
            pelvis_c, pelvis_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.127 * rs, length=0.056,
                                                  orientation=p.getQuaternionFromEuler(DEFAULT_ORIENTATION, physicsClientId=self.id),
                                                skin_color=skin_color, specular_color=specular_color)
            # reuse for both left and right - smpl#1, 2
            hips_c, hips_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.1335 * rs, length=0.094,
                                         position_offset=[0, 0, -0.08125 * hs],
                                         orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
                                                                              physicsClientId=self.id),
                                              skin_color=skin_color, specular_color=specular_color
                                              )
            # smpl#3
            spine1_c, spine1_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.001 * rs, length=0.094,
                                         position_offset=[0, 0, -0.08125 * hs],
                                         orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
                                                                              physicsClientId=self.id),
                                              skin_color=skin_color, specular_color=specular_color
                                        )
            # reuse for both left and right - smpl#4,5
            chest_c, chest_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.127 * rs, length=0.056,
                                         orientation=p.getQuaternionFromEuler(DEFAULT_ORIENTATION, physicsClientId=self.id),
                                                skin_color=skin_color, specular_color=specular_color)
            right_shoulders_c, right_shoulders_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.106 * rs,
                                                               length=0.253 / 8,
                                                               position_offset=[-0.253 / 2.5 + 0.253 / 16, 0, 0],
                                                               orientation=p.getQuaternionFromEuler(DEFAULT_ORIENTATION, physicsClientId=self.id),
                                                               skin_color=skin_color, specular_color=specular_color)
            # left_shoulders_c, left_shoulders_v = create_body(shape=p.GEOM_CAPSULE, radius=0.106 * rs, length=0.253 / 8,
            #                                                  position_offset=[0.253 / 2.5 - 0.253 / 16, 0, 0],
            #                                                  orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
            #                                                                                       physicsClientId=self.id))
            # neck_c, neck_v = create_body(shape=p.GEOM_CAPSULE, radius=0.06 * rs, length=0.124 * hs,
            #                              position_offset=[0, 0, (0.2565 - 0.1415 - 0.025) * hs])
            # upperarm_c, upperarm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.043 * rs, length=0.279 * hs,
            #                                      position_offset=[0, 0, -0.279 / 2.0 * hs])
            # forearm_c, forearm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.033 * rs, length=0.257 * hs,
            #                                    position_offset=[0, 0, -0.257 / 2.0 * hs])
            # hand_c, hand_v = create_body(shape=p.GEOM_SPHERE, radius=0.043 * rs, length=0,
            #                              position_offset=[0, 0, -0.043 * rs])
            # self.hand_radius, self.elbow_radius, self.shoulder_radius = 0.043 * rs, 0.043 * rs, 0.043 * rs
            # waist_c, waist_v = create_body(shape=p.GEOM_CAPSULE, radius=0.1205 * rs, length=0.049,
            #                                orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
            #                                                                     physicsClientId=self.id))
            # hips_c, hips_v = create_body(shape=p.GEOM_CAPSULE, radius=0.1335 * rs, length=0.094,
            #                              position_offset=[0, 0, -0.08125 * hs],
            #                              orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
            #                                                                   physicsClientId=self.id))
            # thigh_c, thigh_v = create_body(shape=p.GEOM_CAPSULE, radius=0.08 * rs, length=0.424 * hs,
            #                                position_offset=[0, 0, -0.424 / 2.0 * hs])
            # shin_c, shin_v = create_body(shape=p.GEOM_CAPSULE, radius=0.05 * rs, length=0.403 * hs,
            #                              position_offset=[0, 0, -0.403 / 2.0 * hs])
            # foot_c, foot_v = create_body(shape=p.GEOM_CAPSULE, radius=0.05 * rs, length=0.215 * hs,
            #                              position_offset=[0, -0.1, -0.025 * rs],
            #                              orientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
            #                                                                   physicsClientId=self.id))
            # elbow_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=(0.043 + 0.033) / 2 * rs, length=0,
            #                               rgbaColor=skin_color, visualFramePosition=[0, 0.01, 0],
            #                               physicsClientId=self.id)
            #TODO: bring back cloth
            # if self.cloth:
            #     # Cloth penetrates the spheres at the end of each capsule, so we create physical spheres at the joints
            #     invisible_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01 * rs, length=0,
            #                                       rgbaColor=[0.8, 0.6, 0.4, 0], physicsClientId=self.id)
            #     shoulder_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.043 * rs, length=0)
            #     elbow_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.043 * rs, length=0)
            #     wrist_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.033 * rs, length=0)

            head_scale = [0.89] * 3
            head_pos = [0.09, 0.08, -0.07 + 0.01]
            head_c = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=os.path.join(self.directory, 'head_female_male',
                                                                  'BaseHeadMeshes_v5_male_cropped_reduced_compressed_vhacd.obj'),
                                            collisionFramePosition=head_pos,
                                            collisionFrameOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                                               physicsClientId=self.id),
                                            meshScale=head_scale, physicsClientId=self.id)
            head_v = p.createVisualShape(shapeType=p.GEOM_MESH,
                                         fileName=os.path.join(self.directory, 'head_female_male',
                                                               'BaseHeadMeshes_v5_male_cropped_reduced_compressed.obj'),
                                         rgbaColor=skin_color, specularColor=specular_color,
                                         visualFramePosition=head_pos,
                                         visualFrameOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                                         physicsClientId=self.id),
                                         meshScale=head_scale, physicsClientId=self.id)

            joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]
            chest_p = [0, 0, 1.2455 * hs]
            shoulders_p = [0, 0, 0.1415 / 2 * hs]
            neck_p = [0, 0, 0.1515 * hs]
            head_p = [0, 0, (0.399 - 0.1415 - 0.1205) * hs]
            right_upperarm_p = [-0.106 * rs - 0.073, 0, 0]
            left_upperarm_p = [0.106 * rs + 0.073, 0, 0]
            forearm_p = [0, 0, -0.279 * hs]
            hand_p = [0, 0, -(0.033 * rs + 0.257 * hs)]
            waist_p = [0, 0, -0.156 * hs]
            hips_p = [0, 0, -0.08125 * hs]
            right_thigh_p = [-0.08 * rs - 0.009, 0, -0.08125 * hs]
            left_thigh_p = [0.08 * rs + 0.009, 0, -0.08125 * hs]
            shin_p = [0, 0, -0.424 * hs]
            foot_p = [0, 0, -0.403 * hs - 0.025]

            joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]
            chest_p = [0, 0, 1.148 * hs]
            shoulders_p = [0, 0, 0.132 / 2 * hs]
            neck_p = [0, 0, 0.132 * hs]
            head_p = [0, 0, 0.12 * hs]
            right_upperarm_p = [-0.092 * rs - 0.067, 0, 0]
            left_upperarm_p = [0.092 * rs + 0.067, 0, 0]
            forearm_p = [0, 0, -0.264 * hs]
            hand_p = [0, 0, -(0.027 * rs + 0.234 * hs)]
            waist_p = [0, 0, -0.15 * hs]
            hips_p = [0, 0, -0.15 / 2 * hs]
            right_thigh_p = [-0.0775 * rs - 0.0145, 0, -0.15 / 2 * hs]
            left_thigh_p = [0.0775 * rs + 0.0145, 0, -0.15 / 2 * hs]
            shin_p = [0, 0, -0.391 * hs]
            foot_p = [0, 0, -0.367 * hs - 0.045 / 2]

        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []
        linkLowerLimits = []
        linkUpperLimits = []

        # NOTE: Shoulders, neck, and head
        linkMasses.extend(m * np.array([0, 0, 0.05, 0]))
        linkCollisionShapeIndices.extend(
            [joint_c, joint_c, right_shoulders_c , head_c])
        linkVisualShapeIndices.extend(
            [joint_v, joint_v, right_shoulders_v,  head_v])
        linkPositions.extend(
            [shoulders_p, shoulders_p, shoulders_p, head_p])
        linkOrientations.extend([joint_o] * 4)
        linkInertialFramePositions.extend([[0, 0, 0]] * 4)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 4)
        linkParentIndices.extend([0, 1, 2, 3])
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 4)
        linkJointAxis.extend(
            [axis['x'], axis['y'], axis['z'], axis['x']])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(-10), np.deg2rad(-10), np.deg2rad(-35), np.deg2rad(-10)]) * limit_scale)
        linkUpperLimits.extend(np.array(
            [np.deg2rad(10), np.deg2rad(30), np.deg2rad(35), np.deg2rad(10)]) * limit_scale)

        # # NOTE: Right arm
        # linkMasses.extend(m * np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        # if not self.cloth:
        #     linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
        #     linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        # else:
        #     linkCollisionShapeIndices.extend(
        #         [joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
        #     linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        # linkPositions.extend([right_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        # linkOrientations.extend([joint_o] * 7)
        # linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        # linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        # # linkParentIndices.extend([0, 1, 2, 0, 4, 5, 0])
        # # linkParentIndices.extend([6, 18, 19, 20, 21, 22, 23])
        # # linkParentIndices.extend([0, 0, 0, 0, 0, 0, 0])
        # linkParentIndices.extend([3, 11, 12, 13, 14, 15, 16])
        # linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        # linkJointAxis.extend([axis['y'], axis['x'], axis['z'], axis['x'], axis['z'], axis['x'], axis['y']])
        # linkLowerLimits.extend(np.array(
        #     [np.deg2rad(5), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-81),
        #      np.deg2rad(-27)]) * limit_scale)
        # linkUpperLimits.extend(np.array(
        #     [np.deg2rad(198), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90),
        #      np.deg2rad(47)]) * limit_scale)
        #
        # # NOTE: Left arm
        # linkMasses.extend(m * np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        # if not self.cloth:
        #     linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
        #     linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        # else:
        #     linkCollisionShapeIndices.extend(
        #         [joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
        #     linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        # linkPositions.extend([left_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        # linkOrientations.extend([joint_o] * 7)
        # linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        # linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        # linkParentIndices.extend([6, 18, 19, 20, 21, 22, 23])
        # linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        # linkJointAxis.extend([axis['y'], axis['x'], axis['z'], axis['x'], axis['z'], axis['x'], axis['y']])
        # linkLowerLimits.extend(np.array(
        #     [np.deg2rad(-198), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-81),
        #      np.deg2rad(-47)]) * limit_scale)
        # linkUpperLimits.extend(np.array(
        #     [np.deg2rad(-5), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90),
        #      np.deg2rad(27)]) * limit_scale)
        #
        # # NOTE: Waist and hips
        # linkMasses.extend(m * np.array([0, 0, 0.13, 0.14]))
        # linkCollisionShapeIndices.extend([waist_c, joint_c, joint_c, hips_c])
        # linkVisualShapeIndices.extend([waist_v, joint_v, joint_v, hips_v])
        # linkPositions.extend([waist_p, hips_p, joint_p, joint_p])
        # linkOrientations.extend([joint_o] * 4)
        # linkInertialFramePositions.extend([[0, 0, 0]] * 4)
        # linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 4)
        # linkParentIndices.extend([0, 25, 26, 27])
        # linkJointTypes.extend([p.JOINT_FIXED] + [p.JOINT_REVOLUTE] * 3)
        # linkJointAxis.extend([[0, 0, 0], axis['x'], axis['y'], axis['z']])
        # linkLowerLimits.extend(np.array([0, np.deg2rad(-75), np.deg2rad(-30), np.deg2rad(-30)]))
        # linkUpperLimits.extend(np.array([0, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)]))
        #
        # # NOTE: Right leg
        # linkMasses.extend(m * np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        # linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        # linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        # linkPositions.extend([right_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        # linkOrientations.extend([joint_o] * 7)
        # linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        # linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        # linkParentIndices.extend([28, 29, 30, 31, 32, 33, 34])
        # linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        # linkJointAxis.extend([axis['x'], axis['y'], axis['z'], axis['x'], axis['x'], axis['y'], axis['z']])
        # linkLowerLimits.extend(np.array(
        #     [np.deg2rad(-127), np.deg2rad(-40), np.deg2rad(-45), 0, np.deg2rad(-35), np.deg2rad(-23), np.deg2rad(-43)]))
        # linkUpperLimits.extend(np.array(
        #     [np.deg2rad(30), np.deg2rad(45), np.deg2rad(40), np.deg2rad(130), np.deg2rad(38), np.deg2rad(24),
        #      np.deg2rad(35)]))
        #
        # # NOTE: Left leg
        # linkMasses.extend(m * np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        # linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        # linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        # linkPositions.extend([left_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        # linkOrientations.extend([joint_o] * 7)
        # linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        # linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        # linkParentIndices.extend([28, 36, 37, 38, 39, 40, 41])
        # linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        # linkJointAxis.extend([axis['x'], axis['y'], axis['z'], axis['x'], axis['x'], axis['y'], axis['z']])
        # linkLowerLimits.extend(np.array(
        #     [np.deg2rad(-127), np.deg2rad(-45), np.deg2rad(-40), 0, np.deg2rad(-35), np.deg2rad(-24), np.deg2rad(-35)]))
        # linkUpperLimits.extend(np.array(
        #     [np.deg2rad(30), np.deg2rad(40), np.deg2rad(45), np.deg2rad(130), np.deg2rad(38), np.deg2rad(23),
        #      np.deg2rad(43)]))

        print (linkParentIndices)
        human = p.createMultiBody(baseMass=0 if static else m * 0.1, baseCollisionShapeIndex=chest_c,
                                  baseVisualShapeIndex=chest_v, basePosition=chest_p,
                                  baseOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                  linkCollisionShapeIndices=linkCollisionShapeIndices,
                                  linkVisualShapeIndices=linkVisualShapeIndices,
                                  linkPositions=linkPositions, linkOrientations=linkOrientations,
                                  linkInertialFramePositions=linkInertialFramePositions,
                                  linkInertialFrameOrientations=linkInertialFrameOrientations,
                                  linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes,
                                  linkJointAxis=linkJointAxis, linkLowerLimits=linkLowerLimits,
                                  linkUpperLimits=linkUpperLimits, useMaximalCoordinates=False,
                                  flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)

        # Self collision has been enabled for the person
        # For stability: Remove all collisions except between the arms/legs and the other body parts
        num_joints = p.getNumJoints(human, physicsClientId=self.id)
        # for i in range(-1, num_joints):
        #     for j in range(-1, num_joints):
        #         p.setCollisionFilterPair(human, human, i, j, 0, physicsClientId=self.id)
        # for i in range(3, 10):  # Right arm
        #     for j in [-1] + list(range(10, num_joints)):
        #         p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        # for i in range(13, 20):  # Left arm
        #     for j in list(range(-1, 10)) + list(range(20, num_joints)):
        #         p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        # for i in range(28, 35):  # Right leg
        #     for j in list(range(-1, 24)) + list(range(35, num_joints)):
        #         p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        # for i in range(35, num_joints):  # Left leg
        #     for j in list(range(-1, 24)) + list(range(28, 35)):
        #         p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)

        print ("Changing the dynamics of the human")
        p.changeDynamics(human, -1, spinningFriction=0.001,
                       rollingFriction=0.001,
                       linearDamping=0.0)
        # Enforce joint limits
        # TODO: bring back
        # human_joint_states = p.getJointStates(human,
        #                                       jointIndices=list(range(p.getNumJoints(human, physicsClientId=self.id))),
        #                                       physicsClientId=self.id)
        # human_joint_positions = np.array([x[0] for x in human_joint_states])
        #
        # for j in range(p.getNumJoints(human, physicsClientId=self.id)):
        #
        #     joint_info = p.getJointInfo(human, j, physicsClientId=self.id)
        #     joint_name = joint_info[1]
        #     joint_pos = human_joint_positions[j]
        #     lower_limit = joint_info[8]
        #     upper_limit = joint_info[9]
        #     # print(joint_name, joint_pos, lower_limit, upper_limit)
        #     if joint_pos < lower_limit:
        #         p.resetJointState(human, jointIndex=j, targetValue=lower_limit, targetVelocity=0,
        #                           physicsClientId=self.id)
        #     elif joint_pos > upper_limit:
        #         p.resetJointState(human, jointIndex=j, targetValue=upper_limit, targetVelocity=0,
        #                           physicsClientId=self.id)
        #     print (joint_info)

        return human

