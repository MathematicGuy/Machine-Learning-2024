from iou import intersection_over_union
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    
    # assert type(bboxes) == list 
    
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
                torch.tensor(box[2:]).unsqueeze(0),
                torch.tensor(best_bbox[2:]).unsqueeze(0),
                box_format=box_format
            ) < iou_threshold
        ]

        #? save best_bboxes to after_nms_list 
        after_nms_list.append(best_bbox)
    
    return after_nms_list
    

if __name__ == '__main__':
    # Example bounding boxes: [box_id, confidence, x1, y1, x2, y2]
    pred_boxes = torch.tensor([
        [2, 0.720081090927124, 33.40931, 242.45248, 70.53368, 282.84384],
        [3, 0.6698232889175415, 439.35947, 230.78842, 477.95145, 283.24826],
        [1, 0.5960968136787415, 443.94025, 484.69257, 489.34003, 536.5535],
        [0, 0.44452500343322754, 21.572742, 492.26105, 61.92349, 544.39557]
    ])

    def visualize_boxes(pred_boxes, nms_boxes):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        titles = ['Before NMS', 'After NMS']
        boxes_list = [pred_boxes, nms_boxes]

        for ax, title, boxes in zip(axes, titles, boxes_list):
            ax.set_title(title)
            
            for box in boxes:
                rect = patches.Rectangle((box[2], box[3]), box[4] - box[2], box[5] - box[3], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[2], box[3], f'Class: {box[0]}, Conf: {box[1]:.2f}', color='r', fontsize=12, verticalalignment='top')

            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            ax.invert_yaxis()

        plt.show()
    
    nms_boxes = nms(pred_boxes.tolist(), iou_threshold=0.5, threshold=0.6)
    visualize_boxes(pred_boxes, nms_boxes)    
    
    # Test the nms function
    nms_list = nms(pred_boxes.tolist(), iou_threshold=0.5, threshold=0.6)
    print("Remaining boxes after NMS:")
    for box in nms_list:
        print(box)    
    