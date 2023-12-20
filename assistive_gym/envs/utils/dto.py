import json
from typing import Optional

import numpy as np
from enum import Enum


class HandoverObject(Enum):
    PILL = "pill"
    CUP = "cup"
    CANE = "cane"

    @staticmethod
    def from_string(label):
        if label == "pill":
            return HandoverObject.PILL
        elif label == "cup":
            return HandoverObject.CUP
        elif label == "cane":
            return HandoverObject.CANE
        else:
            raise ValueError(f"Invalid handover object label: {label}")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PosTransPos:
    def __init__(self, original, transform):
        self.original = original
        self.transform = transform

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, cls=NumpyEncoder)

# TODO: Fix non serializable error
class HumanRobotResult:
    def __init__(self, pelvis: list, joint_angles: list, ee: PosTransPos, ik_target: PosTransPos,
                 robot: PosTransPos, robot_joint_angles: list):
        self.pelvis =  pelvis
        self.joint_angles = joint_angles
        self.ee = ee
        self.ik_target = ik_target
        self.robot = robot
        self.robot_joint_angles = robot_joint_angles
        self.ik_target = ik_target

    def to_json(self):
        print (self.__dict__)
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, cls=NumpyEncoder)

class HumanInfo:
    def __init__(self, original_angles: np.ndarray, original_link_positions: np.ndarray, original_self_collisions,
                 original_env_collisions):
        self.link_positions = original_link_positions  # should be array of tuples that are the link positions
        self.angles = original_angles
        self.self_collisions = original_self_collisions
        self.env_collisions = original_env_collisions


class MaximumHumanDynamics:
    def __init__(self, max_torque, max_manipulibility, max_energy):
        self.torque = max_torque
        self.manipulibility = max_manipulibility
        self.energy = max_energy


class RobotSetting:
    def __init__(self, base_pos, base_orient, robot_joint_angles, robot_side, gripper_orient):
        self.base_pos = base_pos
        self.base_orient = base_orient
        self.robot_joint_angles = robot_joint_angles if robot_joint_angles is not None else np.array([])
        self.robot_side = robot_side
        self.gripper_orient = gripper_orient


class InitRobotSetting:
    def __init__(self, base_pos, base_orient, side):
        self.base_pos = base_pos
        self.base_orient = base_orient
        self.robot_side = side

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, cls=NumpyEncoder)


class HandoverObjectConfig:
    def __init__(self, object_type: HandoverObject, weights: list, limits: list, end_effector: str, robot_side: str):
        self.object_type = object_type
        self.weights = weights
        self.limits = limits
        self.end_effector = end_effector
        self.robot_side = robot_side


class EnvConfig:
    def __init__(self, env_name, person_id, smpl_file, handover_obj, end_effector, coop):
        self.env_name = env_name
        self.person_id = person_id
        self.smpl_file = smpl_file
        self.handover_obj = handover_obj
        self.end_effector = end_effector
        self.coop = coop


class SearchConfig:
    def __init__(self, robot_ik, env_object_ids, original_info, max_dynamics, handover_obj, handover_obj_config,
                 initial_robot_setting):
        self.robot_ik = robot_ik
        self.env_object_ids = env_object_ids  # TODO: check
        self.original_info = original_info
        self.max_dynamics = max_dynamics
        self.handover_obj = handover_obj
        self.handover_obj_config = handover_obj_config
        self.initial_robot_setting = initial_robot_setting


class HandoverValidity:
    def __init__(self, new_self_penetrations, new_env_penetrations, robot_penetrations, robot_dist_to_target):
        self.new_self_penetrations = new_self_penetrations
        self.new_env_penetrations = new_env_penetrations
        self.robot_penetrations = robot_penetrations
        self.robot_dist_to_target = robot_dist_to_target

    def to_json(self):
        return {
            "new_self_penetrations": json.dumps(self.new_self_penetrations, cls=NumpyEncoder),
            "new_env_penetrations": json.dumps(self.new_env_penetrations, cls=NumpyEncoder),
            "robot_penetrations": json.dumps(self.robot_penetrations, cls=NumpyEncoder),
            "robot_dist_to_target": self.robot_dist_to_target
        }



class SearchResult:
    def __init__(self, joint_angles, cost, manipulability, dist, energy, torque, robot_setting,
                 handover_validity: HandoverValidity):
        self.joint_angles = joint_angles  # just for reference, in case multithread messed up the order
        self.cost = cost
        self.dist = dist
        self.manipulability = manipulability
        self.energy = energy
        self.torque = torque
        self.robot_setting = robot_setting
        self.handover_validity = handover_validity


class MainEnvInitResult:
    def __init__(self, original_info: HumanInfo, max_dynamics: MaximumHumanDynamics, env_object_ids,
                 human_link_robot_collision, end_effector, handover_obj_config,
                 human_joint_lower_limits, human_joint_upper_limits, robot_setting: InitRobotSetting):
        self.original_info = original_info
        self.max_dynamics = max_dynamics
        self.env_object_ids = env_object_ids
        self.human_link_robot_collision = human_link_robot_collision
        self.end_effector = end_effector
        self.handover_obj_config = handover_obj_config
        self.human_joint_lower_limits = human_joint_lower_limits
        self.human_joint_upper_limits = human_joint_upper_limits
        self.robot_setting = robot_setting


class MainEnvProcessTaskType(Enum):
    INIT = "init"
    RENDER_STEP = "render_step"
    GET_HUMAN_ROBOT_INFO = "get_human_robot_info"


class MainEnvProcessTask:
    def __init__(self, task_type: MainEnvProcessTaskType):
        self.task_type = task_type


class MainEnvProcessInitTask(MainEnvProcessTask):
    def __init__(self):
        super().__init__(MainEnvProcessTaskType.INIT)


class MainEnvProcessRenderTask(MainEnvProcessTask):
    def __init__(self, joint_angle, robot_setting: RobotSetting):
        super().__init__(MainEnvProcessTaskType.RENDER_STEP)
        self.joint_angle = joint_angle
        self.robot_setting = robot_setting


class MainEnvProcessGetHumanRobotInfoTask(MainEnvProcessTask):
    def __init__(self, joint_angle, robot_setting: RobotSetting, end_effector: str):
        super().__init__(MainEnvProcessTaskType.GET_HUMAN_ROBOT_INFO)
        self.joint_angle = joint_angle
        self.robot_setting = robot_setting
        self.end_effector = end_effector


class MeanKinematicResult:
    def __init__(self, mean_energy, mean_cost, mean_dist, mean_m, mean_torque, mean_evolution):
        self.mean_energy = mean_energy
        self.mean_cost = mean_cost
        self.mean_dist = mean_dist
        self.mean_m = mean_m
        self.mean_torque = mean_torque
        self.mean_evolution = mean_evolution


class BestKinematicResult:
    def __init__(self, best_energy, best_cost, best_dist, best_m, best_torque):
        self.energy = best_energy
        self.cost = best_cost
        self.dist = best_dist
        self.m = best_m
        self.torque = best_torque


class TrialResult:
    def __init__(self, joint_angles, best_kinematic_result: BestKinematicResult,
                 mean_kinematic_result: MeanKinematicResult, robot_setting,
                 handover_validity: HandoverValidity):
        self.joint_angles = joint_angles
        self.best_kinematic_result = best_kinematic_result
        self.mean_kinematic_result = mean_kinematic_result
        self.robot_setting = robot_setting
        self.handover_validity = handover_validity



