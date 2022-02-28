
# IMAGE DATA AUGMENTATION
###########################################

import os
from collections import deque
import numpy as np
import imageio
import imgaug.augmenters as iaa


ORIGIN_FOLDER_PATH = './tests/Test_images/images_to_augment'
WRITE_FOLDER_PATH = './tests/Test_images/augmented_images/'
NUM_TO_AUGMENT = 20

def augment_dataset(folder_path:str,write_folder:str,num_files_desired:int) -> str:
    """
    1.Reads through images in the folder_path
    2.Randomly selects files to augment
    3.Performs random augmentation techniques
    4.Stores augmented images in a folder
    -------------------------
    Parameters
    -------------------------
    folder_path: str
        - The path of the folder that contains the images that should be augmented
    write_folder: str
        - The path of the folder where the augmented images should be written to
    num_images_desired: int
        - The number of images that should be augmented
    -------------------------
    Return: Augmented images in specified write folder
    """
    # Loop on all files of the folder and build a list of files paths
    images = deque([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
            os.path.isfile(os.path.join(folder_path, f))])

    # Randomly select images to augment based on num_files_desired
    random_images = np.random.choice(images, size = num_files_desired)

    for each in random_images:

        # Read each image
        each_image = imageio.imread(each)
        
        # Ensure image is the correct shape
        if not each_image.shape[2] == 3:
            each_image = each_image[:,:,:3]

        # Object that implements
        # augmentation in a random order
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),
            iaa.AdditiveGaussianNoise(scale=(0, 30)),
            iaa.AddToHueAndSaturation((-10, 10)),
            iaa.Crop(percent=(0, 0.3))
        ], random_order=True)

        # Separate image name
        an_image = each.split('/')[-1]

        # Sequence to augment the random image
        images_aug = seq(image=each_image)

        # Write images to write_folder
        imageio.imwrite(write_folder + '_augmented_' + an_image,images_aug)
        print(f"Augmented: {an_image}")

    return f'Completed: {len(os.listdir(write_folder))}'

if __name__ == '__main__':
    augment_dataset(ORIGIN_FOLDER_PATH,WRITE_FOLDER_PATH,NUM_TO_AUGMENT)
