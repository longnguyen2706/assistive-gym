import json
import torch
import os
from torch.utils.data import Dataset, DataLoader

from utils.data_parser import ModelInput, ModelOutput, ModelOutputHumanOnly

SEARCH_INPUT_DIR = 'searchinput'
SEARCH_OUTPUT_DIR = 'searchoutput'
METRIC_DIR = 'metrics'

# TODO: split such that you have unseen person for testing
class SMPLDataset(Dataset):
    def __init__(self, directory, object=None, transform=None, human_only=False):
        """
        Initialize the object with the directory where the JSON files are located.
        Each JSON file contains a single data sample.

        Parameters:
        directory (str): Directory containing the JSON files.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.inputfile_list =[]
        self.metricfile_list = []
        self.outputfile_list = []
        self.human_only = human_only

        # Expect the directory to contain two subdirectories: 'input' and 'output'
        input_dir, output_dir, metric_dir = os.path.join(directory, SEARCH_INPUT_DIR), os.path.join(directory, SEARCH_OUTPUT_DIR), os.path.join(directory, METRIC_DIR)
        # print (input_dir, output_dir)
        # list all subdirectories in 'output'
        person_ids = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])
        invalid_files = json.load(open('/invalid_ik_cases_1212_pill.json', 'r'))
        invalid_set = set()
        for f in invalid_files:
            invalid_set.add((f[0], f[1], f[2]))
        # print (len(invalid_set), invalid_set)   
        # print(person_ids)
        for p in person_ids:
            subinput_dir, sub_output_dir, sub_metric_dir = os.path.join(input_dir, p), os.path.join(output_dir, p), os.path.join(metric_dir, p)
            pose_ids = [f for f in os.listdir(sub_output_dir) if os.path.isdir(os.path.join(sub_output_dir, f))]
            sub_inputfile_list, sub_outputfile_list, sub_metricfile_list= [], [], []

            for pose_id in pose_ids:
                subsub_output_dir = os.path.join(sub_output_dir, pose_id)
                subsub_metric_dir = os.path.join(sub_metric_dir, pose_id)
                output_files = [os.path.join(subsub_output_dir, f) for f in os.listdir(subsub_output_dir) if f.endswith(object + '.json')]
                valid_output_files = []
                for f in output_files:
                    object = os.path.basename(f).split('.')[0]
                    # metric_file = os.path.join(subsub_metric_dir, object + '.json')
                    # metric = json.load(open(metric_file, 'r'))
                    # if 'env_penetrations' not in metric and 'self_penetrations' not in metric: # valid
                    # if 'env_penetrations' not in metric: # valid
                    if not (p, pose_id, object) in invalid_set:
                        valid_output_files.append(f)

                # sub_outputfile_list.extend([os.path.join(subsub_output_dir, f) for f in os.listdir(subsub_output_dir) if f.endswith(object + '.json')])
                if len(valid_output_files) >0:
                    sub_outputfile_list.extend(valid_output_files)
                    sub_inputfile_list.append(os.path.join(subinput_dir, pose_id+'.json'))

            self.inputfile_list.extend(sub_inputfile_list)
            self.outputfile_list.extend(sub_outputfile_list)

            if len(sub_inputfile_list) != len(sub_outputfile_list):
                # print (subinput_dir, sub_output_dir, len(sub_inputfile_list), len(sub_outputfile_list))
                raise AssertionError ("Number of input files and output files do not match for person_id: ", p)

        print ("Total number of samples:", len(self.inputfile_list), "for person_ids: ", person_ids)
        assert len(self.inputfile_list) == len(self.outputfile_list)


    def __len__(self):
        """Return the total number of data points (JSON files)."""
        return len(self.inputfile_list)

    def __getitem__(self, idx):
        """
        Load the JSON file corresponding to 'idx' and return the processed data.
        """
        # Build file path
        inputfile_path = os.path.join(self.directory, self.inputfile_list[idx])
        outputfile_path = os.path.join(self.directory, self.outputfile_list[idx])

        # Load JSON file
        with open(inputfile_path, 'r') as infile:
            input = json.load(infile)
        with open(outputfile_path, 'r') as outfile:
            output = json.load(outfile)

        model_input = ModelInput(input['pose'], input['betas'])
        if self.human_only:
            data = json.loads(output['wrt_pelvis'])
            # print (data['joint_angles'])

            model_output = ModelOutputHumanOnly(data['joint_angles'])
        else:
            # TODO: fix with the new format
            model_output = ModelOutput(output['joint_angles'], output['robot']['original'][0], output['robot']['original'][1], output['robot_joint_angles'])

        feature = model_input.to_tensor()
        label = model_output.to_tensor()
        end_effector = 0 if output['end_effector'] == 'left_hand' else 1
        end_effector = torch.tensor(end_effector, dtype=torch.long)

        if self.transform:
            feature = self.transform(feature)

        return {
            'feature': feature,
            'label': label,
            'end_effector': end_effector,
            'feature_path': inputfile_path,
            'label_path': outputfile_path
        }

if __name__ == '__main__':
    # Path to the directory containing your JSON files
    directory_path = os.path.join(os.getcwd(), os.path.join('data'))

    # Define transformations here (if any)
    transformations = None

    # Create an instance of your dataset
    json_dataset = SMPLDataset(directory_path, transform=transformations)
    print ("Number of samples:", len(json_dataset))

    # Create a data loader
    data_loader = DataLoader(json_dataset, batch_size=16, shuffle=True)

    # Iterate over data to see how data is loaded
    for batch_idx, (features, labels) in enumerate(data_loader):
        print("Batch:", batch_idx)
        # print("Features:", features)
        # print("Labels:", labels)