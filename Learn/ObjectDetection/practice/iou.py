import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates the intersection over union of two set of boxes.
    note: tensor (N, 4): where N is the number of bouding boxes

    Parameters:

        boxes_pred (tensor): Predictions of Bounding Boxes, predict bbox (BATCH_SIZE, 4): [[x,y,w,h], ...]
        boxes_labels (tensor): Correct labels of Bouding Boxes, true bbox (BATCH_SIZE, 4): [[x,y,w,h], ...]
        box_format (str): midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)

    Returns:
        tensor: Intersection over union for all examples
    """
        
    if box_format not in {"midpoint", "corners"}:
        print("corners or midpoint box_format only !!!")
    
    #! Get both pred and label to choose max and min    
    #? corners mean given 2 corners points of 2 rectangles find the top-right and bottom-left of the intersection areas
    if box_format == "corners": # [x1, y1, x2, y2]
        #? top right point 
        pred_x1 = boxes_preds[..., 0:1]
        pred_y1 = boxes_preds[..., 1:2]
        #? bottom left point
        pred_x2 = boxes_preds[..., 2:3]
        pred_y2 = boxes_preds[..., 3:4]
        
        #? top right point
        label_x1 = boxes_labels[..., 0:1]
        label_y1 = boxes_labels[..., 1:2]
        #? bottom left point
        label_x2 = boxes_labels[..., 2:3]
        label_y2 = boxes_labels[..., 3:4]


    #? midpoint mean given 2 rectangle [x,y,w,h] find the top-rights and bottom-lefts for both retangles 
    if box_format == "midpoint": # [x, y, w, h]
        # bottom
        pred_x1 = (boxes_preds[..., 0:1] - boxes_preds[..., 2:3]) / 2
        pred_y1 = (boxes_preds[..., 1:2] - boxes_preds[..., 3:4]) / 2
        
        # top
        pred_x2 = (boxes_preds[..., 0:1] + boxes_preds[..., 2:3]) / 2
        pred_y2 = (boxes_preds[..., 1:2] + boxes_preds[..., 3:4]) / 2
        
        # bottom
        label_x1 = (boxes_labels[..., 0:1] - boxes_labels[..., 2:3]) / 2
        label_y1 = (boxes_labels[..., 1:2] - boxes_labels[..., 3:4]) / 2
        # top
        label_x2 = (boxes_labels[..., 0:1] + boxes_labels[..., 2:3]) / 2
        label_y2 = (boxes_labels[..., 1:2] + boxes_labels[..., 3:4]) / 2
        
        
        
    #? intersection = min (top point) and max (bottom points)
    inter_x1 = torch.max(pred_x1, label_x1) # top 
    inter_x2 = torch.min(pred_x2, label_x2) # bottom
    inter_y1 = torch.max(pred_y1, label_y1) # top
    inter_y2 = torch.min(pred_y2, label_y2) # bottom
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) 
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)
    
    #? Calculate Intersection and Union areas
    epsilon = 1e-6 # e^(-6)
    intersection = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0) # width - height
    union = pred_area + label_area - intersection + epsilon
    
    
    return intersection / union
    
    
if __name__ == '__main__':
    # boxes_preds contains the predicted bounding boxes
    # Each box is represented by [x1, y1, x2, y2] where
    # (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    boxes_preds = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.7, 0.8, 1.0, 2.0]]) 
    boxes_labels = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.3, 0.9, 1.0, 2.0]])
    print(boxes_preds[..., 0:1]) # ... mean all rows, 0:1 mean 1st column
    
    # Midpoint format
    iou_midpoint = intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint")
    print(f"IoU (midpoint): {iou_midpoint}")

    # Corners format
    boxes_preds_corners = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    boxes_labels_corners = torch.tensor([[0.5, 0.5, 1.0, 1.0]])

    iou_corners = intersection_over_union(boxes_preds_corners, boxes_labels_corners, box_format="corners")
    print(f"IoU (corners): {iou_corners}")

