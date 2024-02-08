# Split dataset into train and test groups

import sys
import os
import json
import shutil
import numpy as np

BASE_PATH = 'datasets'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'annotations'])
RAW_ANNOT_PATH = os.path.sep.join([ANNOTS_PATH, 'distalradius_1_200.json'])

TRAIN_PATH = os.path.sep.join([IMAGES_PATH, 'train'])
TEST_PATH = os.path.sep.join([IMAGES_PATH, 'test'])

PROCESSED_ANNOT_PATH = os.path.sep.join([ANNOTS_PATH, 'distalradius_processed_1_200.json'])

TRAIN_TEST_SPLIT = 0.85


"""
Make directories for train and test data
"""


def clear_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        print("Error: %s : %s" % (folder_path, e.strerror))


def reset_folder(folder_path):
    if os.path.isdir(folder_path):
        clear_folder(folder_path)
    os.mkdir(folder_path)


def reset_train_test_folders():
    reset_folder(TRAIN_PATH)
    reset_folder(TEST_PATH)


"""
Randomize data based on train_test_split
"""


def split_data(train_test_split=TRAIN_TEST_SPLIT):
    with open(PROCESSED_ANNOT_PATH, 'r') as openfile:
        annots = json.load(openfile)
    image_ids = [a['image_id'] for a in annots]  # extract image ids from list of dicts
    print(image_ids)

    np.random.shuffle(image_ids)
    train_keys, test_keys = (
        image_ids[:int(len(image_ids) * TRAIN_TEST_SPLIT)],
        image_ids[int(len(image_ids) * TRAIN_TEST_SPLIT):]
    )

    print(train_keys)
    print(test_keys)

    reset_train_test_folders()

    for id in train_keys:
        annot = next((item for item in annots if item["image_id"] == id), None)  # get dict entry where image_id == id
        print(annot)
        source_img_path = os.path.sep.join([IMAGES_PATH, annot['file_name']])
        dest_img_path = os.path.sep.join([TRAIN_PATH, annot['file_name']])
        shutil.copy2(source_img_path, dest_img_path)

    for id in test_keys:
        annot = next((item for item in annots if item["image_id"] == id), None)  # get dict entry where image_id == id
        print(annot)
        source_img_path = os.path.sep.join([IMAGES_PATH, annot['file_name']])
        dest_img_path = os.path.sep.join([TEST_PATH, annot['file_name']])
        shutil.copy2(source_img_path, dest_img_path)


"""
Main function
"""


def main():
    os.chdir(sys.path[0])
    split_data()


main()
