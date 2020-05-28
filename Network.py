import torch
import torch.nn as nn
import torch.optim as optim

import Hyperparameters
import Dataloader
import os
import sys
from resnets import resnet18, resnet34, resnet50, resnet101, resnet152
import numpy as np
import datetime
from datetime import date
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from shutil import copy
from skimage import io

# indicate index of GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_root = Hyperparameters.project_root

# Network initialization
assert Hyperparameters.network in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

if Hyperparameters.network == 'resnet18':
    network = resnet18(pretrained=False)
elif Hyperparameters.network == 'resnet34':
    network = resnet34(pretrained=False)
elif Hyperparameters.network == 'resnet50':
    network = resnet50(pretrained=False)
elif Hyperparameters.network == 'resnet101':
    network = resnet101(pretrained=False)
elif Hyperparameters.network == 'resnet152':
    network = resnet152(pretrained=False)
else:
    print('invalid network')
    sys.exit(-1)

if Hyperparameters.use_GPU:
    network = network.cuda()

if Hyperparameters.optimizer == 'SGD':
    optimizer = optim.SGD(network.parameters(), lr=Hyperparameters.learning_rate,
                          momentum=Hyperparameters.momentum)
elif Hyperparameters.optimizer == 'ADAM':
    optimizer = optim.Adam(network.parameters(), lr=Hyperparameters.learning_rate)
else:
    print('invalid optimizer')
    sys.exit(-1)

if Hyperparameters.loss == 'cross_entropy':
    loss_function = torch.nn.CrossEntropyLoss()
    sum_loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
else:
    print('invalid loss function')
    sys.exit(-1)

# variable to keep track of trainings
training_id = round(time.time())

# create results directories
os.mkdir(os.path.join(project_root, 'results/' + str(training_id)))
os.mkdir(os.path.join(project_root, 'results/' + str(training_id) + '/train'))
os.mkdir(os.path.join(project_root, 'results/' + str(training_id) + '/validation'))
os.mkdir(os.path.join(project_root, 'results/' + str(training_id) + '/test'))
os.mkdir(os.path.join(project_root, 'results/' + str(training_id) + '/test/false_positives'))
os.mkdir(os.path.join(project_root, 'results/' + str(training_id) + '/test/false_negatives'))

# Arrays to keep track of progress
train_losses = []
train_counter = []
train_precision = []
train_recall = []
train_fmeasure = []

validation_losses = []
validation_counter = []
validation_precision = []
validation_recall = []
validation_fmeasure = []
validation_graph_counter = [i * len(Dataloader.train_dataloader_reduced.dataset)
                            for i in
                            range(1, Hyperparameters.n_epochs + 1)]  # one validation round after each training epoch
post_train_validation_loss = []
post_train_validation_precision = []
post_train_validation_recall = []
post_train_validation_fmeasure = []

test_losses = []
test_counter = []
test_precision = []
test_recall = []
test_fmeasure = []

average_train_loss = []
average_validation_loss = []
average_test_loss = []

train_current_epoch = []
validation_current_epoch = []
test_current_epoch = []

current_epoch = 0


# Network training
def train(epoch):
    start = time.time()

    # Variables to calculate precision, recall and f-measure
    precision = 0.0
    recall = 0.0
    f_measure = 0.0

    # Array to keep track of cumulated measures in order: selected, relevant, true_pos, false_pos, true_neg, false_neg
    measures = np.zeros(6)

    network.train()
    train_length = len(Dataloader.train_dataloader_reduced.dataset)
    train_loss = 0.0

    for batch_idx, dictionary_entry in enumerate(Dataloader.train_dataloader_reduced):
        image = dictionary_entry['image']
        image_class = dictionary_entry['class']
        image_name = dictionary_entry['image_name']

        if Hyperparameters.use_GPU:
            image = image.cuda()
            image_class = image_class.cuda()

        optimizer.zero_grad()
        output = network(image)

        # Check decision probability vs ground truth for each batch element
        batch_measures, selected_names, not_selected_names, false_pos_names, false_neg_names = update_measures(output,
                                                                                                               image_class,
                                                                                                               image_name)
        measures = np.add(measures, batch_measures)

        # update cumulative precision, recall and F-measure at each batch
        selected = measures[0]
        relevant = measures[1]
        true_pos = measures[2]
        if selected != 0.0:
            precision = true_pos / selected
        if relevant != 0.0:
            recall = true_pos / relevant
        if precision != 0 or recall != 0:
            f_measure = 2 * (precision * recall) / (precision + recall)

        loss = loss_function(output, image_class)  # mean loss among batch elements
        train_loss += sum_loss_function(output, image_class).item()  # cumulative loss
        loss.backward()
        optimizer.step()

        # log results for graphs at each log interval
        if batch_idx % Hyperparameters.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF-measure: {:.2f}'.format(
                    epoch, batch_idx * len(image), len(Dataloader.train_dataloader_reduced.dataset),
                           100. * batch_idx / len(Dataloader.train_dataloader_reduced), loss.item(), precision, recall,
                    f_measure))

            train_current_epoch.append(current_epoch)
            train_losses.append(loss.item())
            train_precision.append(precision)
            train_recall.append(recall)
            train_fmeasure.append(f_measure)
            train_counter.append(
                (batch_idx * Hyperparameters.batch_size_train) + (
                        (epoch - 1) * len(Dataloader.train_dataloader_reduced.dataset)))

    # Calculate average loss for current training
    train_loss /= train_length
    # fill average loss arrays for results csv file, with corresponding amount of lines
    for i in range(0, math.ceil(float(len(Dataloader.train_dataloader_reduced)) / float(Hyperparameters.log_interval))):
        average_train_loss.append(train_loss)

    end = time.time()
    elapsed_time = str(datetime.timedelta(seconds=(end - start)))

    # Log training results in CSV file (at the end of epoch)
    results = np.asarray(
        [training_id, date.today().strftime("%d/%m/%Y"), Hyperparameters.use_GPU, Hyperparameters.network,
         Hyperparameters.batch_size_train, Hyperparameters.learning_rate, Hyperparameters.optimizer,
         Hyperparameters.loss, epoch, loss.item(), precision, recall, f_measure, elapsed_time])
    results = pd.DataFrame(results).swapaxes(0, 1)
    results.to_csv('/home/homberge/Projet/results/Training_results.csv', mode='a', header=None, index=None)

    torch.save(network.state_dict(), './results/' + str(training_id) + '/model.pth')
    torch.save(optimizer.state_dict(), './results/' + str(training_id) + '/optimizer.pth')


# Network validation
def validate():
    # Variables to calculate precision, recall and f-measure
    precision = 0.0
    recall = 0.0
    f_measure = 0.0

    # Array to keep track of cumulated measures in order: selected, relevant, true_pos, false_pos, true_neg, false_neg
    measures = np.zeros(6)

    network.eval()
    validation_length = len(Dataloader.validation_dataloader_reduced.dataset)
    validation_loss = 0.0

    with torch.no_grad():
        for batch_idx, dictionary_entry in enumerate(Dataloader.validation_dataloader_reduced):
            image = dictionary_entry['image']
            image_class = dictionary_entry['class']
            image_name = dictionary_entry['image_name']

            if Hyperparameters.use_GPU:
                image = image.cuda()
                image_class = image_class.cuda()

            output = network(image)
            loss = loss_function(output, image_class)  # mean loss among batch elements
            validation_loss += sum_loss_function(output, image_class).item()  # cumulative loss

            batch_measures, selected_names, not_selected_names, false_pos_names, false_neg_names = update_measures(
                output,
                image_class,
                image_name)
            measures = np.add(measures, batch_measures)

            # update cumulative precision, recall and F-measure at each batch
            selected = measures[0]
            relevant = measures[1]
            true_pos = measures[2]
            if selected != 0.0:
                precision = true_pos / selected
            if relevant != 0.0:
                recall = true_pos / relevant
            if precision != 0 or recall != 0:
                f_measure = 2 * (precision * recall) / (precision + recall)

            # log results for graphs at each log interval
            if batch_idx % Hyperparameters.log_interval == 0:
                validation_current_epoch.append(current_epoch)
                validation_losses.append(loss.item())
                validation_precision.append(precision)
                validation_recall.append(recall)
                validation_fmeasure.append(f_measure)
                validation_counter.append(
                    (batch_idx * Hyperparameters.batch_size_validation) + (
                            (current_epoch - 1) * validation_length))

    # Calculate average loss for current validation
    validation_loss /= validation_length
    for i in range(0, math.ceil(float(len(Dataloader.validation_dataloader_reduced)) / float(Hyperparameters.log_interval))):
        average_validation_loss.append(validation_loss)

    # Keep track of measures over all validations for graphs
    post_train_validation_loss.append(validation_loss)
    post_train_validation_precision.append(precision)
    post_train_validation_recall.append(recall)
    post_train_validation_fmeasure.append(f_measure)

    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, (measures[2] + measures[4]), validation_length,
        100. * (measures[2] + measures[4]) / validation_length))


# Network testing
def test():
    # Lists of false pos and false neg images
    false_pos_name_array = []
    false_neg_name_array = []

    # Variables to calculate precision, recall and f-measure
    precision = 0.0
    recall = 0.0
    f_measure = 0.0

    # Array to keep track of cumulated measures in order: selected, relevant, true_pos, false_pos, true_neg, false_neg
    measures = np.zeros(6)

    network.eval()
    test_length = len(Dataloader.test_dataloader_reduced.dataset)
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, dictionary_entry in enumerate(Dataloader.test_dataloader_reduced):
            image = dictionary_entry['image']
            image_class = dictionary_entry['class']
            image_name = dictionary_entry['image_name']

            if Hyperparameters.use_GPU:
                image = image.cuda()
                image_class = image_class.cuda()

            output = network(image)
            loss = loss_function(output, image_class)  # mean loss among batch elements
            test_loss += sum_loss_function(output, image_class).item()  # cumulative loss

            batch_measures, selected_names, not_selected_names, false_pos_names, false_neg_names = update_measures(
                output,
                image_class,
                image_name)
            measures = np.add(measures, batch_measures)

            if false_pos_names:
                for item in false_pos_names:
                    false_pos_name_array.append(item)  # flatten and append if list is not empty (empty = false)

            if false_neg_names:
                for item in false_neg_names:
                    false_neg_name_array.append(item)

            # update cumulative precision, recall and F-measure at each batch
            selected = measures[0]
            relevant = measures[1]
            true_pos = measures[2]
            if selected != 0.0:
                precision = true_pos / selected
            if relevant != 0.0:
                recall = true_pos / relevant
            if precision != 0 or recall != 0:
                f_measure = 2 * (precision * recall) / (precision + recall)

            # log results for graphs at each log interval
            if batch_idx % Hyperparameters.log_interval == 0:
                test_current_epoch.append(current_epoch)
                test_losses.append(loss.item())
                test_precision.append(precision)
                test_recall.append(recall)
                test_fmeasure.append(f_measure)
                test_counter.append(
                    (
                            batch_idx * Hyperparameters.batch_size_test))  # epoch independent as testing is done once at the end

    test_loss /= test_length  # average test loss
    for i in range(0, math.ceil(float(len(Dataloader.test_dataloader_reduced)) / float(Hyperparameters.log_interval))):
        average_test_loss.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, (measures[2] + measures[4]), test_length,
        100. * (measures[2] + measures[4]) / test_length))

    # Save images mistaken by model
    save_image(false_pos_name_array, 'false_positives')
    save_image(false_neg_name_array, 'false_negatives')

    # Save names of images mistaken by model
    false_pos_name_array = np.asarray(false_pos_name_array).reshape(len(false_pos_name_array), 1)
    false_neg_name_array = np.asarray(false_neg_name_array).reshape(len(false_neg_name_array), 1)

    false_pos_headers = ['False positives']
    false_neg_headers = ['False negatives']

    pd.DataFrame(false_pos_name_array).to_csv(
        project_root + 'results/' + str(training_id) + '/test/false_positives/false_positives.csv',
        mode='w', header=false_pos_headers, index=None)

    pd.DataFrame(false_neg_name_array).to_csv(
        project_root + 'results/' + str(training_id) + '/test/false_negatives/false_negatives.csv',
        mode='w', header=false_neg_headers, index=None)


def save_image(image_array, folder):
    for image_name in image_array:
        if 'flipped_' in image_name:
            image_name = image_name.replace('flipped_', '')
            img_path = '/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images/' + image_name
            image = io.imread(img_path)  # reads original (non-flipped) image
            horizontal_flip = image[:, ::-1]  # flips image along horizontal axis
            io.imsave(project_root + 'results/' + str(training_id) + '/test/' + folder + '/flipped_' + image_name, horizontal_flip)
        else:
            copy('/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images/' + str(image_name), project_root + 'results/' + str(training_id) + '/test/' + folder)


def update_measures(output, image_class, image_name):
    # Variables to calculate precision, recall and f-measure
    selected = 0.0
    relevant = 0.0
    true_pos = 0.0
    false_pos = 0.0
    true_neg = 0.0
    false_neg = 0.0

    # Lists to keep names of images based on: selected, not selected, false pos, false neg
    selected_names = []
    not_selected_names = []
    false_pos_names = []
    false_neg_names = []

    output_len = len(output)
    output_proba = nn.Softmax(dim=1)(output)  # apply Softmax to convert output to probabilities

    # Check decision probability vs ground truth for each batch element
    for i in range(0, output_len):
        decision_output = output_proba[i][0].item()  # extract the probability of the first class (negative class)
        ground_truth = image_class[i].item()

        if ground_truth == 1:
            relevant += 1.0

        if decision_output < 0.5 and ground_truth == 0:
            false_pos += 1.0
            selected += 1.0
            false_pos_names.append(image_name[i])
            selected_names.append(image_name[i])

        if decision_output < 0.5 and ground_truth == 1:
            true_pos += 1.0
            selected += 1.0
            selected_names.append(image_name[i])

        if decision_output > 0.5 and ground_truth == 0:
            true_neg += 1.0
            not_selected_names.append(image_name[i])

        if decision_output > 0.5 and ground_truth == 1:
            false_neg += 1.0
            false_neg_names.append(image_name[i])
            not_selected_names.append(image_name[i])

    return np.asarray([selected, relevant, true_pos, false_pos, true_neg, false_neg]), \
            selected_names, not_selected_names, false_pos_names, false_neg_names


def plot(x_data, y_data, color, legend, location, x_label, y_label, training_id, folder, filename):
    plt.clf()
    plt.plot(x_data, y_data, color=color)
    plt.legend([legend], loc=location)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(project_root, 'results/' + str(training_id) + '/' + folder + '/' + filename))


def scatter_plot(x_data, y_data, scatter_x, scatter_y, color, scatter_color, legend, location, x_label, y_label,
                 training_id, folder, filename):
    plt.clf()
    plt.plot(x_data, y_data, color=color)
    plt.scatter(scatter_x, scatter_y, color=scatter_color)
    plt.legend(legend, loc=location)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(project_root,
                             'results/' + str(training_id) + '/' + folder + '/' + filename))


def log_rawdata(folder, epoch, counter, losses, average_loss, precision, recall, fmeasure):
    x_length = len(counter)
    network_array = np.full((x_length, 1), Hyperparameters.network)
    epoch_array = np.asarray(epoch).reshape(x_length, 1)
    counter_array = np.asarray(counter).reshape(x_length, 1)
    losses_array = np.asarray(losses).reshape(x_length, 1)
    average_loss_array = np.asarray(average_loss).reshape(x_length, 1)
    precision_array = np.asarray(precision).reshape(x_length, 1)
    recall_array = np.asarray(recall).reshape(x_length, 1)
    fmeasure_array = np.asarray(fmeasure).reshape(x_length, 1)
    seed_array = np.full((x_length, 1), torch.random.initial_seed())
    raw_data = np.concatenate(
        (network_array, epoch_array, counter_array, losses_array, average_loss_array, precision_array,
         recall_array, fmeasure_array, seed_array), axis=1)

    raw_data_headers = ['Network', 'Epoch', 'Examples seen', 'CrossEntropy loss',
                        'Average CrossEntropy loss over epoch',
                        'Precision', 'Recall', 'F-measure', 'Seed']

    pd.DataFrame(raw_data).to_csv('/home/homberge/Projet/results/' + str(training_id) + '/' + folder + '/' +
                                  'raw_data.csv', mode='w', header=raw_data_headers, index=None)


def create_graphs():
    # Training

    # Loss function visualization
    scatter_plot(train_counter, train_losses, validation_graph_counter, post_train_validation_loss, 'blue', 'red',
                 ['Train Loss', 'Validation Loss'], 'upper right', 'number of training examples seen',
                 'CrossEntropy loss', training_id, 'train', 'LossFunction.png')

    # Precision visualization
    scatter_plot(train_counter, train_precision, validation_graph_counter, post_train_validation_precision, 'green',
                 'red',
                 ['Train Precision', 'Validation Precision'], 'lower right', 'number of training examples seen',
                 'Precision', training_id, 'train', 'Precision.png')

    # Recall visualization
    scatter_plot(train_counter, train_recall, validation_graph_counter, post_train_validation_recall, 'orange', 'red',
                 ['Train Recall', 'Validation Recall'], 'lower right', 'number of training examples seen',
                 'Recall', training_id, 'train', 'Recall.png')

    # F-measure visualization
    scatter_plot(train_counter, train_fmeasure, validation_graph_counter, post_train_validation_fmeasure, 'red', 'blue',
                 ['Train F-measure', 'Validation F-measure'], 'lower right', 'number of training examples seen',
                 'F-measure', training_id, 'train', 'F-measure.png')

    # Log training raw data in CSV file
    log_rawdata('train', train_current_epoch, train_counter, train_losses, average_train_loss, train_precision,
                train_recall, train_fmeasure)

    # Validation

    # Loss function visualization
    plot(validation_counter, validation_losses, 'blue', 'Validation Loss', 'upper right',
         'number of validation examples seen', 'CrossEntropy loss', training_id, 'validation', 'LossFunction.png')

    # Precision visualization
    plot(validation_counter, validation_precision, 'green', 'Validation Precision', 'lower right',
         'number of validation examples seen', 'Precision', training_id, 'validation', 'Precision.png')

    # Recall visualization
    plot(validation_counter, validation_recall, 'orange', 'Validation Recall', 'lower right',
         'number of validation examples seen', 'Recall', training_id, 'validation', 'Recall.png')

    # F-measure visualization
    plot(validation_counter, validation_fmeasure, 'red', 'Validation F-measure', 'lower right',
         'number of validation examples seen', 'F-measure', training_id, 'validation', 'F-measure.png')

    log_rawdata('validation', validation_current_epoch, validation_counter, validation_losses, average_validation_loss,
                validation_precision,
                validation_recall, validation_fmeasure)

    # Testing

    # Loss function visualization
    plot(test_counter, test_losses, 'blue', 'Test Loss', 'upper right',
         'number of testing examples seen', 'CrossEntropy loss', training_id, 'test', 'LossFunction.png')

    # Precision visualization
    plot(test_counter, test_precision, 'green', 'Test Precision', 'lower right',
         'number of testing examples seen', 'Precision', training_id, 'test', 'Precision.png')

    # Recall visualization
    plot(test_counter, test_recall, 'orange', 'Test Recall', 'lower right',
         'number of testing examples seen', 'Recall', training_id, 'test', 'Recall.png')

    # F-measure visualization
    plot(test_counter, test_fmeasure, 'red', 'Test F-measure', 'lower right',
         'number of testing examples seen', 'F-measure', training_id, 'test', 'F-measure.png')

    log_rawdata('test', test_current_epoch, test_counter, test_losses, average_test_loss, test_precision, test_recall,
                test_fmeasure)


for epoch in range(1, Hyperparameters.n_epochs + 1):
    current_epoch = epoch
    train(epoch)
    validate()
test()

create_graphs()


