import pandas as pd
import os
from skimage import io, transform
import numpy as np

def dataClean(csv_file, root_dir):

    image_data_raw = pd.read_csv(csv_file)
    # extract columns "image_name" and "class"
    image_data_raw = np.asarray(image_data_raw)[:, [0, 8]]
    image_data = np.array([['image_name', 'class']])  # array initialization

    for i in range(0, len(image_data_raw)):

        try:
            img_name = os.path.join(root_dir, image_data_raw[i, 0])
            print('reading image: ', img_name)
            image = io.imread(img_name)
            image_data = np.append(image_data, [[image_data_raw[i, 0], image_data_raw[i, 1]]], axis=0)
            print(image_data.shape)
        except(IOError, ValueError):
            print('image error')
            continue

    pd.DataFrame(image_data).to_csv('/home/homberge/Projet/datasets/image_list_clean.csv', header=None, index=None)

dataClean(csv_file='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/image_list.csv',
          root_dir='/cvlabsrc1/cvlab/datasets_hugonot/images_gibier/images')