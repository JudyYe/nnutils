"""Wrapper for PointRend Segmentation algorithm.
"""
# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from typing import List

import numpy as np
import torch
from detectron2.modeling import build_model
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes
from detectron2.projects import point_rend
from detectron2.config import get_cfg


class Box2Mask:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, xyxy_boxes, is_object=None, pad=0.1):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            xyxy_boxes: (B, 4?)
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            _, H, W = image.size()

            if isinstance(xyxy_boxes, List):
                xyxy_boxes = Boxes(xyxy_boxes)
            elif torch.is_tensor(xyxy_boxes):
                xyxy_boxes = Boxes(xyxy_boxes)
            xyxy_boxes = pad_box(xyxy_boxes, pad)
            
            inputs = {"image": image, "height": height, "width": width}

            batched_inputs = [inputs]
            model = self.model
            images = model.preprocess_image(batched_inputs)
            features = model.backbone(images.tensor)
            device = images.tensor.device
            if is_object is None:
                is_object = torch.LongTensor([1, ] * len(xyxy_boxes)).to(device)
            if isinstance(is_object, List):
                is_object = torch.LongTensor(is_object).to(device)
            
            # use GT boxes as proposal
            proposals = Instances(
                (height, width),
                proposal_boxes=xyxy_boxes.to(device),
                objectness_logits=torch.ones([len(xyxy_boxes)]).long().to(device),
            )
            # rescale it to processed size
            proposals = detector_postprocess(proposals, H, W)
            # box_head:forward_box, 
            # this is just to obtain class score, the predicted boxes for each class will be discarded 
            box_features = [features[f] for f in model.roi_heads.box_in_features]
            box_features = model.roi_heads.box_pooler(box_features, [x.proposal_boxes for x in [proposals]]) # 256, 7, 7
            box_features = model.roi_heads.box_head(box_features)  # (1024)
            predictions = model.roi_heads.box_predictor(box_features)  # (3, 81), (3, 320=80*4)

            # pred_classes: if is object class number, else person cat (0)
            # !!! the bounding box is an object/non-human class: find the max in score[1:80]
            # !!! the bouding box is a person class: directly use class 0???
            # 
            pred_classes = torch.argmax(predictions[0][:, 1:-1]) * is_object.to(device) + 1 # (3, ) exclude person class [0]
            pred_classes = pred_classes * is_object # not object: pred_class = 0
            proposals.pred_boxes = proposals.proposal_boxes
            proposals.pred_classes = pred_classes

            # pred_instances: top(proposal * 80): pred_boxes, pred_classes, scores
            instances = model.roi_heads.forward_with_given_boxes(features, [proposals])
            predictions = GeneralizedRCNN._postprocess(instances, batched_inputs, images.image_sizes)[0]

            mask = predictions['instances'].pred_masks[0].cpu().detach().numpy()
            mask = mask.astype(np.uint8) * 255
            return predictions,   mask # [H, W]



def pad_box(boxes: Boxes, ratio):
    box = boxes.tensor
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    center = boxes.get_centers()

    widths *= (1 + 2 * ratio)
    heights *= (1 + 2 * ratio)

    boxes = torch.stack([
        center[:, 0] - widths / 2, 
        center[:, 1] - heights / 2, 
        center[:, 0] + widths / 2, 
        center[:, 1] + heights / 2
        ], 1    )
    boxes = Boxes(boxes)
    return boxes


def setup_model(detbase='/home/yufeiy2/scratch/pretrain/detectron2/detectron2', mode='ins_x101_3x') -> Box2Mask:
    """

    :param detbase: 
    model_final_edd263.pkl  
    PointRend/
        configs/
    :param mode: _description_, defaults to 'instance'
    :return: _description_
    """
    model_card = {
        'ins_x101_3x': 
            ('InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml', 
             'model_final_ba17b9.pkl'),
        'ins_50_3x': 
            ('InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml', 
             'model_final_edd263.pkl'),
    }

    cfg = get_cfg()

    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(f'{detbase}/projects/PointRend/configs/{model_card[mode][0]}')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.
    cfg.MODEL.WEIGHTS = f'{detbase}/../weights/{model_card[mode][1]}'
    predictor = Box2Mask(cfg)
    return predictor


def test(image_file):
    print('test')
    image = cv2.imread(image_file + '.png')
    # im_transformed = box2mask.aug.get_transform(image).apply_image(image)
    v = Visualizer(image, coco_metadata)
    # im_transformed = box2mask.aug(image).apply_image(image)
    H, W, _ = image.shape
    box = [[165, 176, 355, 344], [245, 267, 313, 328]]
    prediction, mask = box2mask(image, box, [0, 0])
    point_rend_result = v.draw_instance_predictions(prediction["instances"].to("cpu")).get_image()
    cv2.imwrite(osp.join(save_dir, osp.basename(image_file) + '_vis_hand.png'), point_rend_result)

    prediction, mask = box2mask(image, box, )
    point_rend_result = v.draw_instance_predictions(prediction["instances"].to("cpu")).get_image()
    cv2.imwrite(osp.join(save_dir, osp.basename(image_file) + '_vis.png'), point_rend_result)



if __name__ == '__main__':
    import os.path as osp
    import cv2
    from detectron2.utils.visualizer import Visualizer, ColorMode
    coco_metadata = MetadataCatalog.get("coco_2017_val")
    save_dir = '/home/yufeiy2/scratch/result/vis/'
    box2mask = setup_model()
    image_file = '/home/yufeiy2/scratch/result/HOI4D/ZY20210800003_H3_C3_N31_S282_s03_T2_00163_00200/image/00030'
    test(image_file)

