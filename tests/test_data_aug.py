import os
from data_aug import augment_dataset
import pytest

PATH_TO_AUGMENT = './Test_images/images_to_augment'
WRITE_FOLDER = './Test_images/augmented_images/'
NUM_TO_AUGMENT = 5

def test_augment_dataset():
    """
    Test to verify images are being augmented as expected
    and written to write folder
    """
    augmented_image = augment_dataset(PATH_TO_AUGMENT,WRITE_FOLDER,NUM_TO_AUGMENT)
    assert len(os.listdir(WRITE_FOLDER)) == NUM_TO_AUGMENT
