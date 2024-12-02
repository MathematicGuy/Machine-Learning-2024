from iou import intersection_over_union
import torch

#! Take input of each class. Meaning all bboxes is in the same class
def nms(bboxes, # box in bboxes: [box_id, confidence, x1, y1, x2, y2]
    iou_threshold, # IoU threshold
    threshold,
    box_format="corners"
):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes
        bboxes: bounding boxes predictions (N, 5): [[box_id, confidence, x1, y1, x2, y2]]
        iou_threshold: intersection over union threshold to eliminate overlapping boxes 
        threshold: confidence threshold to keep boxes of different class 
        box_format: coordinate format  

        While BoundingBoxes:
            Take out the largest confidence box (box1)
            For each class:
                Remove all other boxes with IoU(box1, box_i) < threshold
    
        note:
        + confidence say how sure the model prediction is
        + IoU say how close the prediction bounding box to the label bounding box is
    """
    
    #? filter all bboxes by confidence threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda box:box[1], reverse=True) # sort from max to min by confidence score
    
    #! Main Motive: Extract the best bounding boxes of each class  
    #! Method: Remove all bouding boxes that are too close to the best bounding box using best_bboxes as a anchor value (i.e. bboxes work around best_bbox)
    #* In other word, remove all that is too close, retain all is different in class
    
    #? Create a list to store best_bboxes called: after_nms_list = [] 
    after_nms_list = []
    
    #! For each class in bboxes, extract (remove) best_bboxes and remove all bboxes close to best_bbox by a threshold -> Method: Update bboxes by Condition 
    #? Update bboxes: 
    while bboxes:
        #? Extract best_bboxes from bboxes (since we have sort max to min by confidence score)
        best_bbox = bboxes.pop(0)

        #? Update bboxes and only retain bboxes that are not "in the same class" or "close" to best_bbox
        bboxes = [ 
            #? if (box and best_bboxes not in the same class) or (IoU between box and best_bbox < iou_threshold) 
            box for box in bboxes 
            if box[0] != best_bbox[0] or            
            intersection_over_union(
                box[2:],
                best_bbox[2:],
                box_format=box_format
            ) < iou_threshold
        ]

        #? save best_bboxes to after_nms_list 
        after_nms_list.append(best_bbox)
    
    return after_nms_list
    
# Example bounding boxes: [box_id, confidence, x1, y1, x2, y2]
bboxes = torch.tensor([
    [0, 0.95, 100, 100, 200, 200],  # Class 0, high confidence
    [0, 0.90, 110, 110, 210, 210],  # Class 0, slightly lower confidence, overlaps with previous box
    [1, 0.85, 50, 50, 150, 150],    # Class 1, different region
    [0, 0.80, 120, 120, 220, 220],  # Class 0, overlaps with first two boxes
    [1, 0.75, 55, 55, 155, 155],    # Class 1, overlaps slightly with another box in the same class
    [2, 0.60, 300, 300, 400, 400],  # Class 2, no overlap with others
    [2, 0.50, 310, 310, 410, 410],  # Class 2, overlaps with another box in the same class
])


if __name__ == '__main__':
    # Test the nms function
    nms_list = nms(bboxes, iou_threshold=0.5, threshold=0.6)
    print("Remaining boxes after NMS:")
    for box in nms_list:
        print(box)    
    