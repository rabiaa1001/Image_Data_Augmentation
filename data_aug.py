
# IMAGE DATASET AUGMENTATION
###########################################

import imageio
import imgaug.augmenters as iaa
import os
import random


def augment_dataset():
    """
    Read through images in the folder_path
    Randomly select files by to augment
    Store augmented images in a folder
    """
    # Path of folder to images that should be augmented
    folder_path = './images_not_augmented/'
    # Number of file to generate
    num_files_desired = 20


    # Loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
            os.path.isfile(os.path.join(folder_path, f))]

    # Image folder should have atleast 20 images, although this can be changed
    if len(images) <= 20:
        return print("Need at least 20 images")


    # Randomly select images to augment based on num_files_desired
    random_images = random.choices(images, k = num_files_desired)

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

        an_image = each.split('/')[-1]
        write_folder =  './images_augmented/'
        images_aug = seq(image=each_image)

        # Write images to write_folder
        imageio.imwrite(write_folder + 'augmented.' + an_image,images_aug)
        print(f"Augmented: {an_image}")

    return print("\nCompleted Augmentation")

if __name__ == '__main__':
    augment_dataset()

