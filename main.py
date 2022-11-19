# Main code file for distal radius project implementing PyTorch

import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A

from utils import collate_fn
from engine import train_one_epoch, evaluate

BASE_PATH = 'datasets'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
# IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images', 'smallscale']) uncomment to use n = 5
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'annotations'])
TRAIN_PATH = os.path.sep.join([IMAGES_PATH, 'train'])
TEST_PATH = os.path.sep.join([IMAGES_PATH, 'test'])
PROCESSED_ANNOT_PATH = os.path.sep.join([ANNOTS_PATH, 'distalradius_processed.json'])

MODEL_PATH = 'models'
FIGURE_PATH = 'figures'
LOG_PATH = 'logs'

keypoints_classes_ids2names = {0: 'dorsal_distal_radius', 1: 'volar_distal_radius', 2: 'dorsal_distal_shaft', 3: 'dorsal_middle_shaft', 4: 'dorsal_proximal_shaft', 5: 'volar_distal_shaft', 6: 'volar_middle_shaft', 7: 'volar_proximal_shaft'}

NUM_KEYPOINTS = 8


def train_transform():
    return A.Compose([
        A.Sequential([
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1),
            #  A.Blur(blur_limit=3, p=0.5),
        ], p=1)],
        keypoint_params=A.KeypointParams(format='xy'),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
        # Bboxes should have labels
    )


def get_processed_annots(processed_annot_path=PROCESSED_ANNOT_PATH):
    with open(processed_annot_path, 'r') as openfile:
        annots = json.load(openfile)
    return annots


"""
Implementation of Dataset adapted for custom distal radius data
"""


class DistalRadiusDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo
        # Use demo=True if you need transformed and original images
        self.imgs_files = sorted(glob.glob(root + '/*.png'))  # .png files only
        self.annots = get_processed_annots()

    def __getitem__(self, idx):
        img_path = self.imgs_files[idx]
        data = next((item for item in self.annots if item["file_name"] == os.path.basename(img_path)), None)
        img_original = cv2.imread(img_path)
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
            # print(keypoints_original)
            # print(np.array(transformed['keypoints']))
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']),
                                                           (-1, NUM_KEYPOINTS, 2)).tolist()
            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []

            for o_idx, obj in enumerate(keypoints_transformed_unflattened):  # Iterating over objects
                obj_keypoints = []

                for k_idx, kp in enumerate(obj):  # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    kp.append(1)
                    obj_keypoints.append(kp)
                keypoints.append(obj_keypoints)

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


"""
Visualize method adapted from https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da
"""


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
                image_original = cv2.circle(image_original, tuple(kp), 0, (255, 0, 0), 5)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 0, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(image_original)
        ax[0].set_title('Ground Truth', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Prediction', fontsize=fontsize)
        plt.show()


"""
Permute image to displayable format and extrace bboxes and keypoints from output
"""


def get_images_bboxes_and_keypoints(img, target):
    image = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bboxes = target['boxes'].detach().cpu().numpy().astype(np.int32).tolist()
    keypoints = []
    for kps in target['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kps])
    return image, bboxes, keypoints


"""
Create model from torchvision model zoo
"""


def get_model(num_keypoints=NUM_KEYPOINTS, weights_path=None):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None,
                                                                   weights_backbone=ResNet50_Weights.DEFAULT,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes=2,  # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model


"""
Train model
"""


def train_model(data_loader_train, data_loader_test, model=None, num_epochs=5, save=None, log=""):
    if model is None:
        model = get_model()
        # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    train_losses = []
    train_kp_losses = []
    val_aps = []

    for epoch in range(num_epochs):
        date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        log += f'{date_time}\tEpoch: {epoch}\n'  # log time of each epoch start

        metrics = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)

        date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        log += f'{date_time}\t{str(metrics)}\n'  # log training metrics

        train_losses.append(metrics.loss.global_avg)
        train_kp_losses.append(metrics.loss_keypoint.global_avg)
        lr_scheduler.step

        temp_log = io.StringIO()  # metrics not returnable; capture print statemets to obtain formatted results
        with redirect_stdout(temp_log):
            eval_results = evaluate(model, data_loader_test, device)
        temp_log = temp_log.getvalue()
        print(temp_log)
        val_aps.append(eval_results.coco_eval['keypoints'].stats[0])  # AP @ IoU=0.50:0.95

        date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        try:
            temp_log = temp_log[temp_log.index('IoU'):]
            log += f'{date_time}\tCOCO evaluator results:\n{temp_log}\n'
        except ValueError:
            log += 'COCO evaluator results missing \'IoU\''
        log += '\n'

    # print(f'Training losses:{train_losses}')
    # print(f'Keypoint losses:{train_kp_losses}')
    # print(f'Validation Average Precision @ IoU = 0.50:0.95:{val_aps}')

    # Save model weights after training
    date_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if save:
        torch.save(model.state_dict(), os.path.sep.join([MODEL_PATH, f'{save}_{date_time}.pth']))

    train_metrics = {'train_losses': train_losses, 'train_kp_losses': train_kp_losses, 'val_aps': val_aps}
    return train_metrics, log


"""
Display graph of training metrics
"""


def plot_metrics(train_metrics, save_fig_name=None):
    train_losses = train_metrics['train_losses']
    train_kp_losses = train_metrics['train_kp_losses']
    val_aps = train_metrics['val_aps']
    num_epochs = len(train_losses)
    params = {"ytick.color": "black",
              "xtick.color": "black",
              "axes.labelcolor": "black",
              "axes.edgecolor": "black"}
    plt.rcParams.update(params)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # t = f.suptitle('Performance', fontsize=12, color='black')
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(num_epochs))
    ax1.plot(epoch_list, train_losses, label='Global Loss')
    ax1.plot(epoch_list, train_kp_losses, label='Keypoint Loss')
    ax1.set_xticks(np.arange(0, num_epochs, 2))
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Loss', color='w')
    # l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, val_aps, label='Average Precision @ IoU = 0.50:0.95')
    ax2.set_xticks(np.arange(0, num_epochs, 5))
    ax2.set_ylabel('')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.set_title('Average Precision', color='w')
    # l2 = ax2.legend(loc="best")

    plt.show()

    if save_fig_name is not None:
        date_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        plt.savefig(os.path.sep.join([FIGURE_PATH, f'{save_fig_name}_{date_time}.png']), bbox_inches='tight', pad_inches=0)
    return f


"""
Run predict and display results
"""


def predict(data_loader, model, device=None, display_preds=True):
    if device is None:
        # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        device = torch.device('cpu')
    model.to(device)

    iterator = iter(data_loader)
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)

    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)

    # print("Predictions: \n", output[0])

    image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    print(f'Prediction scores: {scores}')

    high_scores_idxs = np.where(scores > 0.3)[0].tolist()  # Indexes of boxes with scores > 0.7
    if len(high_scores_idxs) == 0:
        high_scores_idxs = [0]
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

    # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
    # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
    # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    if display_preds:
        img, original_bboxes, original_keypoints = get_images_bboxes_and_keypoints(images[0], targets[0])
        visualize(image, bboxes, keypoints, img, original_bboxes, original_keypoints)


"""
Save log as text tile in log
"""


def save_log(log, save_name=""):
    date_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    file_path = os.path.sep.join([LOG_PATH, f'{save_name}_{date_time}.txt'])
    with open(file_path, "w") as text_file:
        text_file.write(log)


"""
Main function
"""


def main():
    os.chdir(sys.path[0])
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    train_dataset = DistalRadiusDataset(root=TRAIN_PATH, transform=train_transform(), demo=False)
    test_dataset = DistalRadiusDataset(root=TEST_PATH, transform=None, demo=False)
    data_loader_train = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    model = get_model()
    model = get_model(weights_path=os.path.sep.join([MODEL_PATH, 'keypointsrcnn_e10plus10_2022_11_19_182057.pth']))
    savename = 'keypointsrcnn_e10plus10'
    model.to(device)
    # metrics, log = train_model(data_loader_train, data_loader_test, model, num_epochs=10, save=savename)
    # plot_metrics(metrics, save_fig_name=savename)
    # save_log(log, save_name=savename)

    # evaluate(model, data_loader_test, device)

    predict(data_loader_test, model, device)

    # train_iter = iter(train_dataset)
    # img, target, img_original, target_original = next(train_iter)

    # image, bboxes, keypoints = get_images_bboxes_and_keypoints(img, target)
    # image_original, bboxes_original, keypoints_original = get_images_bboxes_and_keypoints(img_original, target_original)
    # visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)
    # batch = next(myiter)
    # print(len(batch))
    # print(batch)


main()
