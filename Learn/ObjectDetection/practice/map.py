import torch 
from iou import intersection_over_union
from data import true_boxes, pred_boxes

#* note: nếu chưa có lọc ground truth detection thì sẽ có lặp lại khi debug, ko đáp ứng đk mỗi gt đi vs 1 pred.
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    for c in num_classes:
        print(c)
    
    # precision = tp / tp + fp
    # recall tp = tp + fn
    
    


if __name__ == "__main__":
    #? Predicted and Ground truth bounding boxes
    pred_boxes = pred_boxes
    true_boxes = true_boxes

    mean_average_precision(pred_boxes, true_boxes)