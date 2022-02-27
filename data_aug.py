
# IMAGE DATA AUGMENTATION
###########################################

import os
from collections import deque
import numpy as np
import imageio
import imgaug.augmenters as iaa


ORIGIN_FOLDER_PATH = './images_not_augmented/'
WRITE_FOLDER_PATH = './images_augmented/'
NUM_TO_AUGMENT = 20

def augment_dataset():
    """
    1.Reads through images in the folder_path
    2.Randomly selects files to augment
    3.Stores augmented images in a folder
    -------------------------
    Parameters: None
    -------------------------
    Return: Augmented images in specified write folder
    """
    # Path of folder to images that should be augmented
    folder_path = ORIGIN_FOLDER_PATH
    # Number of file to generate
    num_files_desired = NUM_TO_AUGMENT

    # Loop on all files of the folder and build a list of files paths
    images = deque([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
            os.path.isfile(os.path.join(folder_path, f))])

    # Randomly select images to augment based on num_files_desired
    random_images = np.random.choice(images, size = num_files_desired)

    for each in random_images:

        # Read each image
        each_image = imageio.imread(each)

        # Object that implements
        # augmentation in a random order
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),
            iaa.AdditiveGaussianNoise(scale=(0, 30)), # (30, 90)
            iaa.AddToHueAndSaturation((-10, 10)),# (-20, 20)
            iaa.Crop(percent=(0, 0.3)) # .2
        ], random_order=True)
        
        # Separate image name
        an_image = each.split('/')[-1]
        # Retrieve image path
        write_folder =  WRITE_FOLDER_PATH
        # Sequence to augment the random image
        images_aug = seq(image=each_image)

        # Write images to write_folder
        imageio.imwrite(write_folder + 'augmented.' + an_image,images_aug)
        print(f"Augmented: {an_image}")

    return print("\nCompleted Augmentation")

if __name__ == '__main__':
    augment_dataset()

