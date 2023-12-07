import json
import os

class Metrics:
    def __init__(self):
        self.env_collision = 0
        self.self_collision = 0
        self.total = 0

    def cal_metrics(self):
        return self.env_collision / self.total, self.self_collision / self.total

METRICS_DIR = '../data/input/metrics'


def get_data_metrics():
    person_ids = sorted([f for f in os.listdir(METRICS_DIR) if os.path.isdir(os.path.join(METRICS_DIR, f))])
    counter = {
        'cane': Metrics(),
        'cup':  Metrics(),
        'pill': Metrics()
    }
    invalid_cases = set()
    for p in person_ids:
        sub_metrics_dir = os.path.join(METRICS_DIR, p)
        pose_ids = [f for f in os.listdir(sub_metrics_dir) if os.path.isdir(os.path.join(sub_metrics_dir, f))]

        personal_counter = {
            'cane': Metrics(),
            'cup':  Metrics(),
            'pill': Metrics()
        }
        for pose_id in pose_ids:
            subsub_metrics_dir = os.path.join(sub_metrics_dir, pose_id)
            metrics_files = [f for f in os.listdir(subsub_metrics_dir) if f.endswith('.json')]
            for metrics_file in metrics_files:
                object_name = metrics_file.split('.')[0]
                metrics_file_path = os.path.join(subsub_metrics_dir, metrics_file)
                with open(metrics_file_path, 'r') as infile:
                    metrics = json.load(infile)

                if 'self_penetrations' in metrics:
                    counter[object_name].self_collision += 1
                    personal_counter[object_name].self_collision += 1
                    invalid_cases.add((p, pose_id, object_name))
                if 'env_penetrations' in metrics:
                    counter[object_name].env_collision += 1
                    personal_counter[object_name].env_collision += 1
                    invalid_cases.add((p, pose_id, object_name))
                counter[object_name].total += 1
                personal_counter[object_name].total += 1

        for object_name in personal_counter:
            print (p, object_name, personal_counter[object_name].cal_metrics(), personal_counter[object_name].total)
    for object_name in counter:
        print (object_name, counter[object_name].cal_metrics(), counter[object_name].total)

    # save invalid cases
    invalid_cases = sorted(list(invalid_cases))
    print ( len(invalid_cases), invalid_cases)
    with open('../../invalid_cases.json', 'w') as outfile:
        outfile.write(json.dumps(invalid_cases, indent=4))


if __name__ == '__main__':
    get_data_metrics()