from IoU import intersection_over_union
import torch


def nms(
    bboxes, # box in bboxes: [box_id, box_iou/confidence, x1, y1, x2, y2]
    confidence_threshold, # IoU threshold
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
    if isinstance(bboxes, torch.Tensor) == False:
        bboxes = torch.tensor(bboxes)
    
    # assert isinstance(bboxes, torch.Tensor) # warning if bboxes is not tensor        
    
    # Discard all bboxes with IoU < iou_threshold by only taking boxes with iou > threshold                               ````
    bboxes = [box for box in bboxes if box[1] > threshold]        
    
    #? key: sort key, allow list/dict to sort base on "given parameter" e.g. key=len, key= 
    #? lambda: input as input condition, retrieve iterable element and modified its   
    #? Sorted allow us to iterate throught the given iterable (i.e. bbxoes),
    #? We can extract bboxes element and set sort key to the element value (i.e. bounding box)  
    bboxes = sorted(bboxes, key=lambda box:box[1], reverse=True) # iterate through each elements inside bboxes and sort (min-max) by its key
    nms_list = []
    
    while bboxes: # loop through bboxes until it empty
        chosen_box = bboxes.pop(0) # get and remove the 1st bboxes's box 

        #? removing all boxes in the same class that are close to `chosen_box` and retain all other class boxes 
        bboxes = [
            box for box in bboxes 
            if box[0] != chosen_box[0] # if box in bboxes not the same as chosen box
            or intersection_over_union(
                box[2:],
                chosen_box[2:],
                box_format) < confidence_threshold
        ]
        
        nms_list.append(chosen_box)
        
    return nms_list

# Example bounding boxes: [box_id, box_iou/confidence, x1, y1, x2, y2]

bboxes = [
    [0, 0.95, 100, 100, 200, 200],  # Class 0, high confidence
    [0, 0.90, 110, 110, 210, 210],  # Class 0, slightly lower confidence, overlaps with previous box
    [1, 0.85, 50, 50, 150, 150],    # Class 1, different region
    [0, 0.80, 120, 120, 220, 220],  # Class 0, overlaps with first two boxes
    [1, 0.75, 55, 55, 155, 155],    # Class 1, overlaps slightly with another box in the same class
    [2, 0.60, 300, 300, 400, 400],  # Class 2, no overlap with others
    [2, 0.50, 310, 310, 410, 410],  # Class 2, overlaps with another box in the same class
]

if __name__ == '__main__':
    # Test the nms function
    nms_list = nms(bboxes, confidence_threshold=0.5, threshold=0.6)
    print("Remaining boxes after NMS:")
    for box in nms_list:
        print(box)    
    