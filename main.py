# Main code file for distal radius project implementing PyTorch

import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A

import transforms
import utils
import engine
import train

from utils import collate_fn
from engine import train_one_epoch, evaluate

import annotations_parser

BASE_PATH = 'datasets'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
keypoints_classes_ids2names = {0: 'dorsal_distal_radius', 1: 'volar_distal_radius', 2: 'dorsal_distal_shaft', 3: 'dorsal_middle_shaft', 4: 'dorsal_proximal_shaft', 5: 'volar_distal_shaft', 6: 'volar_middle_shaft', 7: 'volar_proximal_shaft'}


def train_transform():
    return A.Compose([
        A.Sequential([
            A.Rotate(limit=45, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,
                                       brightness_by_max=True, always_apply=False, p=1),
            # Random change of brightness & contrast
        ], p=1)],
        keypoint_params=A.KeypointParams(format='xy'),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
        # Bboxes should have labels
    )


class DistalRadiusDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo
        # Use demo=True if you need transformed and original images
        self.imgs_files = sorted(glob.glob(os.path.join(root, "images") + '/*.png'))  # .png files only
        self.annotations_files = sorted(glob.glob(os.path.join(root, "annotations") + '/*.json'))
        self.data = annotations_parser.convert_coco_annots()

    def __getitem__(self, idx):
        img_path = self.data[idx]['path']
        data = self.data[idx]
        img_original = cv2.imread(img_path[1:])
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        # with open(annotations_path) as f:
        #     data = json.load(f)
        # data = annotations_parser.convert_coco_annots()
        bboxes_original = data['bboxes']
        keypoints_original = data['keypoints']

        # All objects are glue tubes
        bboxes_labels_original = ['Glue tube' for _ in bboxes_original]

        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-
            #    format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three
            #    objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]
            #    ], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            # Apply augmentations
            transformed = self.transform(image=img_original,
                                         bboxes=bboxes_original,
                                         bboxes_labels=bboxes_labels_original,
                                         keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for
            #   three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2],
            #   where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2],
            #   [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array
                                                           (transformed['keypoints']),
                                                           (-1, 2, 2)).tolist()
            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            print(keypoints_transformed_unflattened)

            for o_idx, obj in enumerate(keypoints_transformed_unflattened):  # Iterating over objects
                obj_keypoints = []

                for k_idx, kp in enumerate(obj):  # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    kp.append(1)
                    obj_keypoints.append(kp)
                    # print([keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
                print(obj_keypoints)

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)  # all objects are glue tubes

        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 0, (255, 0, 0), 5)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 0, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.show()
    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255, 0, 0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)


def main():
    os.chdir(sys.path[0])
    dataset = DistalRadiusDataset(root=BASE_PATH, transform=train_transform(), demo=True)
    myiter = iter(dataset)
    img, target, img_original, target_original = next(myiter)
    print()
    # plt.imshow(img_original.permute(1, 2, 0))
    # plt.show()

    image = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bboxes = target['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    keypoints = []
    for kps in target['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kps])
    visualize(image, bboxes, keypoints)


main()
