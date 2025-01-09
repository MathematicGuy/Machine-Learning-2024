import torch
import matplotlib.pyplot as plt
 
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", debug = False):
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
    
    #? Ensure boxes_preds and boxes_labels are tensors
    if not isinstance(boxes_preds, torch.Tensor):
        boxes_preds = torch.tensor(boxes_preds)
    if not isinstance(boxes_labels, torch.Tensor):
        boxes_labels = torch.tensor(boxes_labels)

    
    #! Get both pred and label to choose max and min    
    #? corners mean given 2 corners points of 2 rectangles find the top-right and bottom-left of the intersection areas
    if box_format == "corners": # [x1, y1, x2, y2] - (x1, y1): top-left and (x2, y2): bottom-right
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
    
    if debug:
        fig, ax = plt.subplots(1)
        # Plotting the rectangles and intersection area
        # Plot predicted boxes
        for i in range(boxes_preds.shape[0]):
            pred_rect = plt.Rectangle((pred_x1[i], pred_y1[i]), pred_x2[i] - pred_x1[i], pred_y2[i] - pred_y1[i], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(pred_rect)

        # Plot label boxes
        for i in range(boxes_labels.shape[0]):
            label_rect = plt.Rectangle((label_x1[i], label_y1[i]), label_x2[i] - label_x1[i], label_y2[i] - label_y1[i], linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(label_rect)

        # Plot intersection area
        for i in range(inter_x1.shape[0]):
            inter_rect = plt.Rectangle((inter_x1[i], inter_y1[i]), inter_x2[i] - inter_x1[i], inter_y2[i] - inter_y1[i], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(inter_rect)

            # Add IoU score to the center of the intersection area
            iou_score = (intersection / union)[i].item()
            center_x = (inter_x1[i] + inter_x2[i]) / 2
            center_y = (inter_y1[i] + inter_y2[i]) / 2
            plt.text(center_x, center_y, f'IoU: {iou_score:.2f}', fontsize=9, color='blue', ha='center', va='center')

        # Set fixed YOLO range (0 to 1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.title('Bounding Boxes IoU Visualization\nRed: Predicted, Green: Ground Truth, Blue: Intersection')
        plt.show()



    return intersection / union
    
    
if __name__ == '__main__':
    # boxes_preds contains the predicted bounding boxes
    # Each box is represented by [x1, y1, x2, y2] where
    # (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    boxes_preds_midpoint = torch.tensor([[0.5, 0.5, 0.4, 0.3]])  # [x, y, w, h]
    boxes_labels_midpoint = torch.tensor([[0.45, 0.48, 0.35, 0.28]])
    
    # Midpoint format
    iou_midpoint = intersection_over_union(boxes_preds_midpoint, boxes_labels_midpoint, box_format="midpoint", debug = True)
    print(f"IoU (midpoint): {iou_midpoint}")

    # Corners format
    boxes_preds_corners = torch.tensor([[0.2, 0.1, 0.8, 0.7], [0.5, 0.5, 0.7, 0.8]]) 
    boxes_labels_corners = torch.tensor([[0.2, 0.1, 0.8, 0.7], [0.55, 0.55, 0.75, 0.85]])

    iou_corners = intersection_over_union(boxes_preds_corners, boxes_labels_corners, box_format="corners", debug = True)
    
    print(f"IoU (corners): {iou_corners}")
