from IoU import intersection_over_union
import torch

def nms(
    bboxes, # [box_id, box_iou, x1, y1, x2, y2]
    iou_threshold,
    threshold,
    box_format="corners"
):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes
        bboxes: bounding boxes predictions (N, 5): [[box_id, box_iou, x1, y1, x2, y2]]
        iou_threshold: intersection over union threshold to eliminate overlapping boxes 
        threshold: confidence threshold to keep boxes of different class 
        box_format: coordinate format  

        While BoundingBoxes:
            Take out the largest confidence box (box1)
            For each class:
                Remove all other boxes with IoU(box1, box_i) > threshold
    """

    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > iou_threshold]        

