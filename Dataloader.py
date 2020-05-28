import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from skimage import io
from skimage.transform import rescale
import Transforms
import Hyperparameters
import time

torch.manual_seed(0)


class WildlifeDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
                Args:
                    csv_file (string): Path to the csv file with annotations.
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
        """

        self.image_data = pd.read_csv(csv_file)
        self.image_data = np.asarray(self.image_data)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # idx = 0    # when activated, will give always the same image to the network, which should rapidly achieve overfiting
        img_path = os.path.join(self.root_dir, self.image_data[idx, 0])
        image = io.imread(img_path)
        image = rescale(image, 0.25, multichannel=True)
        image = torch.tensor(image)
        image = image.permute(2, 1, 0)  # puts channel dimension first
        image = image.float()           # converts from integers to float

        image_class = self.image_data[idx, 1]
        image_class = torch.tensor(image_class)

        image_name = self.image_data[idx, 0]

        sample = {'image': image/255.0, 'class': image_class, 'image_name': image_name}   # division by 255: tensor will have values between 0 and 1

        if self.transform:
            sample = self.transform(sample)

        return sample


wildlife_dataset = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean.csv',       # Use the clean CSV file
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')

'''
fig = plt.figure()
result_root = '/home/homberge/Projet/results'

for i in range(len(wildlife_dataset)):
    sample = wildlife_dataset[i]
    print(i, sample['image'].shape, sample['class'].shape)

    if i == 3:
        break
# end of instantiation example
'''

dataloader = DataLoader(wildlife_dataset, batch_size=Hyperparameters.batch_size_train, shuffle=True)


positive_dataset = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_positive.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')

negative_dataset = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_negative.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')


# first 5'701 negative images were added to a new CSV so that they are horizontally flipped
flipped_negative_dataset = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_negative_to_augment.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images', transform=Transforms.HorizontalFlip)

# augmented negative dataset will have same size as positive dataset (15'133 images)
augmented_negative_dataset = ConcatDataset([negative_dataset, flipped_negative_dataset])


'''
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        # length of the smallest dataset
        return min(len(d) for d in self.datasets)
'''


# preparing training, validation and testing datasets
train_size_pos = int(len(positive_dataset) * Hyperparameters.train_proportion)
train_size_neg = int(len(negative_dataset) * Hyperparameters.train_proportion)
validation_size_pos = int(len(positive_dataset) * Hyperparameters.validation_proportion)
validation_size_neg = int(len(negative_dataset) * Hyperparameters.validation_proportion)
test_size_pos = len(positive_dataset) - train_size_pos - validation_size_pos
test_size_neg = len(negative_dataset) - train_size_neg - validation_size_neg

positive_train_dataset, positive_validation_dataset, positive_test_dataset = \
    torch.utils.data.random_split(positive_dataset, [train_size_pos, validation_size_pos, test_size_pos])

negative_train_dataset, negative_validation_dataset, negative_test_dataset = \
    torch.utils.data.random_split(negative_dataset, [train_size_neg, validation_size_neg, test_size_neg])

train_dataset = ConcatDataset([positive_train_dataset, negative_train_dataset])
validation_dataset = ConcatDataset([positive_validation_dataset, negative_validation_dataset])
test_dataset = ConcatDataset([positive_test_dataset, negative_test_dataset])


train_dataloader = DataLoader(train_dataset, batch_size=Hyperparameters.batch_size_train, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=Hyperparameters.batch_size_validation, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=Hyperparameters.batch_size_test, shuffle=True)


# Reduced datasets for testing purposes
train_dataset_reduced = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_reduced_train.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')

validation_dataset_reduced = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_reduced_validation.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')

test_dataset_reduced = WildlifeDataset(csv_file='/home/homberge/Projet/datasets/image_list_clean_reduced_test.csv',
                                    root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')

train_dataloader_reduced = DataLoader(train_dataset_reduced, batch_size=Hyperparameters.batch_size_train, shuffle=True)
validation_dataloader_reduced = DataLoader(validation_dataset_reduced, batch_size=Hyperparameters.batch_size_validation, shuffle=True)
test_dataloader_reduced = DataLoader(test_dataset_reduced, batch_size=Hyperparameters.batch_size_test, shuffle=True)


# Save dataloaders for comparison
def save_dataloaders():
    torch.save(train_dataloader, './results/dataloaders/' + str(round(time.time())) + '_train_dataloader.pth')
    torch.save(validation_dataloader, './results/dataloaders/' + str(round(time.time())) + '_validation_dataloader.pth')
    torch.save(test_dataloader, './results/dataloaders/' + str(round(time.time())) + '_test_dataloader.pth')


def compare_datasets(dataloader_1, dataloader_2):
    image_list_1 = []
    image_list_2 = []

    for dictionary_entry in dataloader_1:
        image = dictionary_entry['image_name']
        entry_len = len(image)
        for i in range(0, entry_len):
            image_list_1.append(image[i])

    for dictionary_entry in dataloader_2:
        image = dictionary_entry['image_name']
        entry_len = len(image)
        for i in range(0, entry_len):
            image_list_2.append(image[i])

    if len(image_list_1) != len(image_list_2):
        print('datasets are not of equal length')
        return

    result = all(elem in image_list_1 for elem in image_list_2)

    if result:
        print('datasets are equal')
    else:
        print('datasets are not equal')

    image_list_1 = np.asarray(image_list_1).reshape(len(image_list_1), 1)
    image_list_2 = np.asarray(image_list_2).reshape(len(image_list_2), 1)

    data_header_1 = ['dataloader_1']
    data_header_2 = ['dataloader_2']

    pd.DataFrame(image_list_1).to_csv('/home/homberge/Projet/results/dataloaders/dataloader_1.csv', mode='w',
                                      header=data_header_1, index=None)
    pd.DataFrame(image_list_2).to_csv('/home/homberge/Projet/results/dataloaders/dataloader_2.csv', mode='w',
                                      header=data_header_2, index=None)



# save_dataloaders()
# dataloader_1 = torch.load('./results/dataloaders/1589183211_test_dataloader.pth')
# dataloader_2 = torch.load('./results/dataloaders/1589268812_test_dataloader.pth')
# compare_datasets(dataloader_1, dataloader_2)