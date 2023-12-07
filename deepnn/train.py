import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune, train as ray_train
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ray.tune.schedulers import ASHAScheduler
from deepnn.model.variable_depth_net import VariableDepthNet
from deepnn.preprocess.custom_dataset import CustomDataset
from deepnn.utils.data_parser import ModelOutput, ModelOutputHumanOnly
from deepnn.utils.loss_utils import cal_loss, cal_joint_angle_loss

INPUT_PATH = os.path.join(os.getcwd(), os.path.join('data', 'input'))
OUTPUT_PATH = os.path.join(os.getcwd(), os.path.join('data', 'output'))
CHECKPOINT_PATH = os.path.join(os.getcwd(), os.path.join('checkpoints'))

# TODO: move this one to yaml
# Hyperparameters
input_size = 82  # 72 + 10
output_size = 32  # 32
output_size_human_only = 15
num_epochs = 500
OUTPUT_HUMAN_ONLY = True

def get_output_size():
    return output_size_human_only if OUTPUT_HUMAN_ONLY else output_size

def get_output_class():
    return ModelOutputHumanOnly if OUTPUT_HUMAN_ONLY else ModelOutput

def get_model(config):
    # return MyNet(input_size, config['h1_size'], config['h2_size'], config['h3_size'], get_output_size())
    return VariableDepthNet(input_size, config['layer_sizes'], get_output_size(), config['dropout'])

def get_data_split(batch_size, object):  # 60% train, 20% val, 20% test
    datasets = CustomDataset(INPUT_PATH, object, transform=None, human_only = OUTPUT_HUMAN_ONLY)

    train_size, val_size = int(len(datasets) * 0.7), int(len(datasets) * 0.1)
    test_size = len(datasets) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train(config, exp_name="ray", is_tune=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_split(config['batch_size'], config['object'])

    # init model
    model = get_model(config).to(device)
    print (model)

    criterion = nn.MSELoss()  # For regression, we use Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Initialize the SummaryWriter
    writer = SummaryWriter('runs/' + exp_name)  # Specify the directory for logging

    # Train the model
    for epoch in range(num_epochs):
        for i, train_data in enumerate(train_loader):
            # print (features.shape, labels.shape)
            # Move data to the defined device
            features, labels = train_data['feature'], train_data['label']
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels) *100

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if is_tune:
                # Report the metric to optimize
                tune.report(loss=loss.item())
            else:
                if (i + 1) % 40 == 0:
                    # Log the loss value to TensorBoard
                    train_loss, train_err, = get_dataset_loss(model, criterion, train_loader, device)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Train Loss: {train_loss:.4f}', f'Train Error: {train_err:.4f}')
                    writer.add_scalar('loss', loss, epoch * len(train_loader) + i)
                    writer.add_scalar('training loss', train_loss, epoch * len(train_loader) + i)
                    writer.add_scalar('training err', train_err, epoch * len(train_loader) + i)
                if (i + 1) % 40 == 0:
                    # Calculate the validation loss
                    val_loss, val_err = get_dataset_loss(model, criterion, val_loader, device)
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Validation Loss: {val_loss:.4f}', f'Validation Error: {val_err:.4f}')
                    # Log the validation loss to TensorBoard
                    writer.add_scalar('validation loss', val_loss, epoch * len(train_loader) + i)
                    writer.add_scalar('validation err', val_err, epoch * len(train_loader) + i)
    writer.close()
    test_model(model, test_loader, criterion)
    # Save the model checkpoint with time stamp
    torch.save(
            {'config': config,
             'model': model.state_dict()
            }, os.path.join(CHECKPOINT_PATH, f'model_{config["object"]}_epoch_{num_epochs}_{datetime.now()}.ckpt'))

def get_dataset_loss(model, criterion, loader, device):
    with torch.no_grad():
        # Calculate the loss
        loss = 0.0
        err = 0.0
        for data in loader:
            features, labels = data['feature'], data['label']
            # Move data to the correct device
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            # Update running loss value
            loss += loss.item() * features.size(0)
            # for i in range(len(outputs)):
            #     angle_err = cal_per_joint_err(labels[i], outputs[i])['human_joint_angle_loss']
            #     err += angle_err
            # print (labels.shape, outputs.shape, features.shape)
            err += cal_per_joint_err(labels, outputs) * features.size(0)
        # Calculate the average loss over the entire validation dataset
        # print ("loss: ", loss, "len: ", len(loader.dataset))
        average_val_loss = loss / len(loader.dataset)
        average_err = err / len(loader.dataset)
        print (len(loader.dataset))
    model.train()
    return average_val_loss, average_err

def cal_per_joint_err(labels, outputs):
    # label = label.cpu().numpy()
    # output = output.cpu().numpy()
    # label_obj = get_output_class().from_tensor(label)
    # output_obj = get_output_class().from_tensor(output)
    err = torch.norm ((labels - outputs),p=1, dim=1)/ labels.shape[1] * 180 / np.pi
    err = torch.mean(err)

    # err = {}
    # if OUTPUT_HUMAN_ONLY:  # only calculate human joint angle loss
    #     # human_joint_angle_loss = cal_joint_angle_loss(label_obj.human_joint_angles, output_obj.human_joint_angles)
    #     # print(f'Human joint angle err (deg): {human_joint_angle_loss}', ' file: ', test_data['feature_path'][i])
    #     err['human_joint_angle_loss'] = human_joint_angle_loss
    # else:
    #     human_joint_angle_loss, robot_joint_angle_loss, robot_base_loss, robot_base_rot_loss = cal_loss(
    #         label_obj, output_obj)
    #     # print(
    #     #     f'Human joint angle err (deg): {human_joint_angle_loss}, Robot joint angle err (deg): {robot_joint_angle_loss}, '
    #     #     f'robot base pos err (m): {robot_base_loss}, robot base orient err: {robot_base_rot_loss}')
    #     err = {
    #         'human_joint_angle_loss': human_joint_angle_loss,
    #         'robot_joint_angle_loss': robot_joint_angle_loss,
    #         'robot_base_loss': robot_base_loss,
    #         'robot_base_rot_loss': robot_base_rot_loss
    #     }
    # return err
    return err
def test_model(model, test_loader, criterion):
    """
    Evaluate the performance of a neural network model on a test dataset.

    Parameters:
    model (nn.Module): The neural network model.
    test_loader (DataLoader): DataLoader object for the test dataset.
    criterion (nn.Module): Loss function used for the model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure the model is in evaluation mode (as it affects specific layers like dropout)
    model.eval()

    # To accumulate the losses and the number of examples seen
    running_loss = 0.0
    total_examples = 0

    # No need to track gradients for evaluation, saving memory and computations
    with torch.no_grad():
        for test_data in test_loader:
            features, labels = test_data['feature'], test_data['label']
            # Move data to the correct device\
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Update running loss value
            running_loss += loss.item() * features.size(0)  # loss.item() gives the mean loss per batch
            total_examples += features.size(0)

            # calculate custom loss
            # TODO: see if we need to run it outside
            for i in range(len(outputs)):
                label, output = labels[i], outputs[i]
                label = label.cpu().numpy()
                output = output.cpu().numpy()
                label_obj = get_output_class().from_tensor(label)
                output_obj = get_output_class().from_tensor(output)

                if OUTPUT_HUMAN_ONLY: # only calculate human joint angle loss
                    human_joint_angle_loss= cal_joint_angle_loss(label_obj.human_joint_angles, output_obj.human_joint_angles)
                    print(f'Human joint angle err (deg): {human_joint_angle_loss}', ' file: ', test_data['feature_path'][i])
                else:
                    human_joint_angle_loss, robot_joint_angle_loss, robot_base_loss, robot_base_rot_loss = cal_loss(
                    label_obj, output_obj)
                    print(
                        f'Human joint angle err (deg): {human_joint_angle_loss}, Robot joint angle err (deg): {robot_joint_angle_loss}, '
                        f'robot base pos err (m): {robot_base_loss}, robot base orient err: {robot_base_rot_loss}')

    # Calculate the average loss over the entire test dataset
    average_loss = running_loss / total_examples

    # Additional metrics can be calculated such as R2 score, MAE, etc.
    print(f'Average Loss on the Test Set: {average_loss:.4f}')

    # Put the model back to training mode
    model.train()

    return average_loss  # Depending on your needs, you might want to return other metrics.

def train_with_ray(config):

    # Use the ASHA scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    # Run the experiment
    result = tune.run(
        train,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=1,
        scheduler=scheduler,
    )

    # Print the best result
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    train(best_trial.config, is_tune=False)

def eval_model(model_checkpoint):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(os.path.join(CHECKPOINT_PATH, model_checkpoint))
    config = saved_data['config']
    print ("config:", config)
    model = get_model(config).to(device)

    model.load_state_dict(saved_data['model'])
    model.eval()

    datasets = CustomDataset(INPUT_PATH, config['object'], transform=None)
    eval_loader = DataLoader(datasets, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            features, labels, input_files, output_files = data['feature'], data['label'], data['feature_path'], data[
                'label_path']
            # Move data to the correct device\
            features = features.to(device)
            # Forward pass
            outputs = model(features)
            # save to file
            for j in range(len(outputs)):
                output = outputs[j]
                output = output.cpu().numpy()
                output_obj = get_output_class().from_tensor(output)
                data = output_obj.convert_to_dict()
                output_file = output_files[j]
                output_file = output_file.replace('input/searchoutput', 'output')
                # write to output file and create folder if not exist
                print(f'Writing to {output_file}')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(json.dumps(data, indent=4))


if __name__ == '__main__':
    # Define the hyperparameter search space
    # config = {
    #     "lr": tune.loguniform(1e-5, 0.25*1e-1),
    #     "weight_decay": tune.loguniform(1e-5, 1e-1),
    #     # "layer_sizes": [tune.grid_search(list(range(256, 1024, 128))), tune.grid_search(list(range(128, 256, 32))),
    #     #                 tune.grid_search(list(range(64, 128, 32))), tune.grid_search(list(range(64, 128, 32))), tune.grid_search(list(range(32, 128, 16)))],
    #     "layer_sizes": [tune.grid_search(list(range(512, 4096, 128))), tune.grid_search(list(range(256, 1024, 64))),
    #                     tune.grid_search(list(range(64, 128, 32))), tune.grid_search(list(range(32, 128, 16)))],
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "object": "pill"
    # }

    # config = {
    #     "lr": tune.loguniform(1e-5, 0.25 * 1e-1),
    #     "weight_decay": tune.loguniform(1e-5, 1e-1),
    #     # "dropout": tune.loguniform(0.5*1e-2, 0.5*1e-1),
    #     "dropout": 0,
    #     "layer_sizes": [
    #         tune.grid_search(list(range(1024, 8192, 512))),
    #         tune.grid_search(list(range(512, 4096, 256))), tune.grid_search(list(range(256, 1024, 128))),
    #                     tune.grid_search(list(range(64, 128, 32))), tune.grid_search(list(range(32, 64, 16)))],
    #     "batch_size": tune.choice([16]),
    #     "object": "pill"
    # }
    # # hyper param tuning and train with best config
    # train_with_ray(config)

    #train with best config
    best_config = {'lr': 0.0011160391792473308, 'weight_decay': 4.984018369225781e-05, 'dropout': 0.009958794050846667,
                   'layer_sizes': [6656, 768, 384, 96, 32], 'batch_size': 16, 'object': 'pill'}
    train(best_config,exp_name='t4', is_tune=False)

    # load model and output angle to file
    # model_checkpoint= 'model_pill_epoch_200_2023-12-06 22:56:17.022110.ckpt'
    # eval_model(model_checkpoint)
