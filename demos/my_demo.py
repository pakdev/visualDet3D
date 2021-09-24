import json
import os
import sys

import cv2
import torch
import jsbeautifier
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, "/ssd/git/visualDet3D")

from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.networks.utils.registry import (
    DETECTOR_DICT,
    DATASET_DICT,
    PIPELINE_DICT,
)
from visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.utils.utils import draw_3D_box

print(f"CUDA available: {torch.cuda.is_available()}")
cfg = cfg_from_file("/ssd/git/visualDet3D/config/pete.py")

checkpoint_name = "GroundAware_pretrained.pth"
dataset_name = cfg.data.val_dataset
dataset = DATASET_DICT[dataset_name](cfg, "validation")

detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
detector = detector.cuda()

weight_path = os.path.join(cfg.path.checkpoint_path, checkpoint_name)
state_dict = torch.load(weight_path, map_location="cuda:{}".format(cfg.trainer.gpu))
new_dict = state_dict.copy()
for key in state_dict:
    if "focalLoss" in key:
        new_dict.pop(key)
detector.load_state_dict(new_dict, strict=False)
detector.eval().cuda()

projector = BBox3dProjector().cuda()
backprojector = BackProjection().cuda()


def denorm(image):
    new_image = np.array(
        (image * cfg.data.augmentation.rgb_std + cfg.data.augmentation.rgb_mean) * 255,
        dtype=np.uint8,
    )
    return new_image


def draw_bbox2d_to_image(image, bboxes2d, color=(255, 0, 255)):
    drawed_image = image.copy()
    for box2d in bboxes2d:
        cv2.rectangle(
            drawed_image,
            (int(box2d[0]), int(box2d[1])),
            (int(box2d[2]), int(box2d[3])),
            color,
            3,
        )
    return drawed_image


def compute_once(index):
    name = "%06d" % index
    data = dataset[index]
    if isinstance(data["calib"], list):
        P2 = data["calib"][0]
    else:
        P2 = data["calib"]
    original_height = data["original_shape"][0]
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]
    image = collated_data[0]

    with torch.no_grad():
        image = collated_data[0]
        P2 = collated_data[1]
        scores, bbox, _ = detector([image.cuda().contiguous(), P2.cuda().contiguous()])

        # Uncomment to write images to files
        # img = image.squeeze().permute(1, 2, 0).numpy()
        # rgb_image = denorm(img)

        # P2 = P2[0]
        # bbox_2d = bbox[:, 0:4]
        # bbox_3d_state = bbox[:, 4:]  # [cx,cy,z,w,h,l,alpha]

        # bbox_3d_state_3d = backprojector(
        #     bbox_3d_state, P2.cuda()
        # )  # [x, y, z, w, h ,l, alpha]
        # abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, P2.cuda())

        # if len(scores) > 0:
        #     rgb_image = draw_bbox2d_to_image(rgb_image, bbox_2d.cpu().numpy())
        # for box in bbox_3d_corner_homo:
        #     box = box.cpu().numpy().T
        #     rgb_image = draw_3D_box(rgb_image, box)

        # cv2.imwrite(f"{index}.png", rgb_image)

        bbox_2d = bbox[:, 0:4]
        for score, bbox in zip(scores, bbox_2d):
            score_num = float(score.cpu().numpy())
            yield score_num, bbox.cpu().numpy()


coco = {"annotations": []}
for i in range(669):
    for score, bbox in compute_once(i):
        if bbox is not None:
            coco["annotations"].append(
                {
                    "image_id": i,
                    "bbox": [
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2] - bbox[0]),
                        int(bbox[3] - bbox[1]),
                    ],
                    "category_id": 0,
                    "score": score,
                }
            )

with open("annotations.json", "w") as f:
    json_annotations = jsbeautifier.beautify(json.dumps(coco))
    f.write(json_annotations)
