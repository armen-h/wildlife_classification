import torch
import torch.nn as nn
import Hyperparameters
import pandas as pd
import sys
from Dataloader import WildlifeDataset
from torch.utils.data import DataLoader
from resnets import resnet18, resnet34, resnet50, resnet101, resnet152
from shutil import copy
import time
import os
import numpy as np

project_root = Hyperparameters.project_root
inference_id = round(time.time())

os.mkdir(os.path.join(project_root, 'inference/' + str(inference_id)))
os.mkdir(os.path.join(project_root, 'inference/' + str(inference_id) + '/selected'))
os.mkdir(os.path.join(project_root, 'inference/' + str(inference_id) + '/non_selected'))


def infer(training_id, image_list):

    # define network to use
    raw_data = pd.read_csv(project_root + 'results/' + training_id + '/train/raw_data.csv')
    input_network = raw_data['Network'][0]

    if input_network == 'resnet18':
        network = resnet18(pretrained=False)
    elif input_network == 'resnet34':
        network = resnet34(pretrained=False)
    elif input_network == 'resnet50':
        network = resnet50(pretrained=False)
    elif input_network == 'resnet101':
        network = resnet101(pretrained=False)
    elif input_network == 'resnet152':
        network = resnet152(pretrained=False)
    else:
        print('invalid network')
        sys.exit(-1)

    # load model parameters
    if Hyperparameters.use_GPU:
        device = torch.device('cuda')
        network.load_state_dict(torch.load(project_root + 'results/' + training_id + '/model.pth'))
        network.to(device)
    else:
        device = torch.device('cpu')
        network.load_state_dict(torch.load(project_root + 'results/' + training_id + '/model.pth', map_location=device))


    # transform image_list into a dataset
    wildlife_dataset = WildlifeDataset(csv_file=image_list, root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')
    dataloader = DataLoader(wildlife_dataset, batch_size=Hyperparameters.batch_size_test, shuffle=True)

    # infer results

    # Lists of selected and non_selected images
    selected_name_array = []
    non_selected_name_array = []

    network.eval()

    with torch.no_grad():
        for batch_idx, dictionary_entry in enumerate(dataloader):
            image = dictionary_entry['image']
            image_name = dictionary_entry['image_name']

            if Hyperparameters.use_GPU:
                image = image.cuda()

            output = network(image)
            selected_names, non_selected_names = update_measures(output, image_name)

            if selected_names:
                for item in selected_names:
                    selected_name_array.append(item)  # flatten and append if list is not empty (empty = false)

            if non_selected_names:
                for item in non_selected_names:
                    non_selected_name_array.append(item)

    # save images
    save_image(selected_name_array, 'selected')
    save_image(non_selected_name_array, 'non_selected')

    # save image names
    selected_name_array = np.asarray(selected_name_array).reshape(len(selected_name_array), 1)
    non_selected_name_array = np.asarray(non_selected_name_array).reshape(len(non_selected_name_array), 1)

    selected_headers = ['Selected images']
    non_selected_headers = ['Non-selected images']

    pd.DataFrame(selected_name_array).to_csv(
        project_root + 'inference/' + str(inference_id) + '/selected/selected.csv',
        mode='w', header=selected_headers, index=None)

    pd.DataFrame(non_selected_name_array).to_csv(
        project_root + 'inference/' + str(inference_id) + '/non_selected/non_selected.csv',
        mode='w', header=non_selected_headers, index=None)


def save_image(image_array, folder):
    for image_name in image_array:
        copy('/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images/' + str(image_name), project_root + 'inference/' + str(inference_id) + '/' + folder)


def update_measures(output, image_name):
    # Lists to keep names of images based on: selected, not selected
    selected_names = []
    non_selected_names = []

    output_len = len(output)
    output_proba = nn.Softmax(dim=1)(output)  # apply Softmax to convert output to probabilities

    # Check decision probability for each batch element
    for i in range(0, output_len):
        decision_output = output_proba[i][0].item()  # extract the probability of the first class (negative class)

        if decision_output < 0.5:
            selected_names.append(image_name[i])

        if decision_output > 0.5:
            non_selected_names.append(image_name[i])

    return selected_names, non_selected_names


infer('1588870344', project_root + 'inference/inference.csv')
