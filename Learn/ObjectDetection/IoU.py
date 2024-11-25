import torch

# parameters (prediction_boounding_box, label_bounding_box, box_format)
def intersection_over_union(boxes_preds, boxes_labels, box_format=""):
    """
    Calculates the intersection over union of two set of boxes.
    note: tensor (N, 4): where N is the number of bouding boxes

    Parameters:

        boxes_pred (tensor): Predictions of Bounding Boxes, predict bbox (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bouding Boxes, true bbox (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    
    #? Localize Box1 and Box2    
    if box_format == "midpoint":
        # Subtract half width/height for the top-left corner
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2 
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        # Add half width/height for the bottom-right corner
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2 
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_preds[..., 2:3] / 2 
        box2_y1 = boxes_labels[..., 1:2] - boxes_preds[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] - boxes_preds[..., 2:3] / 2 
        box2_y2 = boxes_labels[..., 1:2] - boxes_preds[..., 3:4] / 2
        
        
    if box_format == "corners":    
        # Extract x1, y1, x2, y2 from the boxes
        box1_x1 = boxes_preds[..., 0:1] 
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] # (N, 1)
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    
    #? Convert to torch value
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    
    #? Calculate the Intersection Area
    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Area of Union (check math formula later)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = (box1_area + box2_area - intersection + 1e-6)
    
    return intersection / union

 