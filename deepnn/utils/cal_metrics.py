import json
import os

import numpy as np


class Metrics:
    def __init__(self):
        self.env_collision = 0
        self.self_collision = 0
        self.total = 0
        self.dists = []
        self.ik_dists = []
        self.torques = []

    def cal_metrics(self):
        return {
            "env_collision": self.env_collision / self.total,
            "self_collision": self.self_collision / self.total,
            "avg_dists": sum(self.dists) / len(self.dists),
            "avg_ik_dists": sum(self.ik_dists) / len(self.ik_dists),
            "std_ik_dists": np.std(self.ik_dists),
            "avg_torques": sum(self.torques) / len(self.torques),
            "invalid_ik_dists_percentage": len([d for d in self.ik_dists if d > 0.2]) / len(self.ik_dists),
            "avg_ik_dist_excluding_invalid": sum([d for d in self.ik_dists if d < 0.2]) / len([d for d in self.ik_dists if d < 0.2]),
        }


# METRICS_DIR = '../data/input/metrics2/HumanComfort-v1_rerun'

# METRICS_DIR = '/home/louis/Documents/Projects/assistive-gym/trained_models/HumanComfort-v1_rerun_dec15'
METRICS_DIR = '/home/louis/Documents/hrl/results/HumanComfort-v1_rerun_dec15_cup'
def get_data_metrics():
    person_ids = sorted([f for f in os.listdir(METRICS_DIR) if os.path.isdir(os.path.join(METRICS_DIR, f))])
    counter = {
        # 'cane': Metrics(),
        'cup':  Metrics(),
        # 'pill': Metrics()
    }
    invalid_cases = set()
    for p in person_ids:
        sub_metrics_dir = os.path.join(METRICS_DIR, p)
        pose_ids = [f for f in os.listdir(sub_metrics_dir) if os.path.isdir(os.path.join(sub_metrics_dir, f))]

        personal_counter = {
            # 'cane': Metrics(),
            'cup':  Metrics(),
            # 'pill': Metrics()
        }
        for pose_id in pose_ids:
            subsub_metrics_dir = os.path.join(sub_metrics_dir, pose_id)
            metrics_files = [f for f in os.listdir(subsub_metrics_dir) if f.endswith('.json')]
            for metrics_file in metrics_files:
                object_name = metrics_file.split('.')[0]
                metrics_file_path = os.path.join(subsub_metrics_dir, metrics_file)
                with open(metrics_file_path, 'r') as infile:
                    metrics = json.load(infile)


                if metrics['validity']['new_self_penetrations'] != '[]':
                    counter[object_name].self_collision += 1
                    personal_counter[object_name].self_collision += 1
                    invalid_cases.add((p, pose_id, object_name))

                if metrics['validity']["new_env_penetrations"]!= '[]':
                    counter[object_name].env_collision += 1
                    personal_counter[object_name].env_collision += 1
                    invalid_cases.add((p, pose_id, object_name))

                if metrics['validity']['robot_dist_to_target'] > 0.2:
                    invalid_cases.add((p, pose_id, object_name))

                    print (p, pose_id, object_name, metrics['validity']['robot_dist_to_target'])
                counter[object_name].ik_dists.append(metrics['validity']['robot_dist_to_target'])
                counter[object_name].dists.append(metrics['dist'])
                counter[object_name].torques.append(metrics['torque'])
                personal_counter[object_name].ik_dists.append(metrics['validity']['robot_dist_to_target'])
                personal_counter[object_name].dists.append(metrics['dist'])
                personal_counter[object_name].torques.append(metrics['torque'])

                counter[object_name].total += 1
                personal_counter[object_name].total += 1

        for object_name in personal_counter:
            print (p, object_name, personal_counter[object_name].cal_metrics(), personal_counter[object_name].total)
    for object_name in counter:
        print (object_name, counter[object_name].cal_metrics(), counter[object_name].total)

    # save invalid cases
    invalid_cases = sorted(list(invalid_cases))
    print ( len(invalid_cases))
    print ( len(invalid_cases), invalid_cases)
    # with open('../../invalid_ik_cases_cup.json', 'w') as outfile:
    #     outfile.write(json.dumps(invalid_cases, indent=4))


if __name__ == '__main__':
    get_data_metrics()