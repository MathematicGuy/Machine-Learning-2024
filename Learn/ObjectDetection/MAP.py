import torch 
from collections import Counter
from IoU import intersection_over_union
from data import true_boxes, pred_boxes

#! notice: chưa có lọc gt-detection nên sẽ có lặp lại 
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

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    #? each class ~ each image (e.g. dog is a class, cat is another class)
    for c in range(num_classes): # c = [0, 5]
        detections = []
        ground_truths = []

        print(f"Class {c}\n")
        
        #? get all bounding boxes of the same class 'c'
        for box in pred_boxes:
            if box[1] == c:  # If class_id == c
                detections.append(box)
                        
        for box in true_boxes:
            if box[1] == c:
                ground_truths.append(box) 
        
        
        # print('dt:', detections)
        print(f'class {c} gt:', ground_truths)

        #! Counting total bbox index of the same class by box's index
        #? e.g. true_boxes = [[1,0,x1,y1,w1,h1], [1,0,x2,y2,w2,h2]] as Counter({1:2}) where 1 is box's index
        amount_bboxes = Counter([boxes[0] for boxes in ground_truths]) # Counter({0: 2, 1: 1, 2: 1})        
        print('lllll:',amount_bboxes)
        
        #? Create key:value dictionary to take count of matched gt_bboxes where key = ground truth box's index
        amount_bboxes_list = []
        #? convert bboxes numbers into zeros list (to take count of ground_truth - detection pair matches)
        for gt_idx, total_index in amount_bboxes.items(): # example input: [2, 1] -> take value in from dict key-value pair 
            amount_bboxes[gt_idx] = torch.zeros(total_index)  # [{ 0: tensor([0., 0.]), 1: tensor([0.]) }]
            amount_bboxes_list.append({gt_idx: amount_bboxes[gt_idx]})
                           
        print(amount_bboxes_list)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_truth_bboxes = len(ground_truths)
        print('True Positive:',TP)
        
        # skip if there no ground truth bboxes for this class
        if total_truth_bboxes == 0:
            continue
        
        #? get largest detection confidence bbox -> sort max-min (then select by class_id)
        #? sort detections from large to small base on each detection probability/confidence 
        detections.sort(key= lambda x:x[1], reverse=True)
        
        
        for detection_idx, detection in enumerate(detections):
            
            #! Solve Multiple Prediction and Classes Problems 
            #? only get same bbox index (train_idx == truth_idx)              
            class_ground_truth = [
                gt for gt in ground_truths if gt[0] == detection[0]
            ]
            print(f'gt-detection {detection_idx}:', class_ground_truth)
            print('detection_index', detection)
            
            best_gt_idx = 0
            best_iou = 0
            
            #? find the best detection to ground truth IoU
            for gt_idx, gt in enumerate(class_ground_truth):
                iou = intersection_over_union(
                    torch.tensor(detection[2:]),
                    torch.tensor(gt[2:]),
                    box_format='midpoint'
                )

                print(f'iou: {detection}\n {gt} -> {iou}')
                if iou > best_iou:
                    best_iou = iou        
                    best_gt_idx = gt_idx
            
            print('train_index:', detection[0])
            
            if detection[0] > len(amount_bboxes):
                print("!!!!!!!!!!!!!!!!")
                
        print('train index:', detection[0])
        print(f'best_gt_idx: {best_gt_idx}')
        print('best_iou:', best_iou)
        # print(f'detection {detection_idx} amount_bboxes: {amount_bboxes[detection[0]]}')
        
        #? annotate gt_box index in amount_bboxes acoording to detection index
        #? TP & FP classified by threshold 
        #! How to track detection_index along with gt_index <-> detection[0] = gt[0]  
        if iou > iou_threshold:
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
                print(f'amount_bboxes: {amount_bboxes[detection[0]][best_gt_idx]}')
            else:
                FP[detection_idx] = 1                                         
        else:
            #? if ground truth already matched
            FP[detection_idx] = 1
            #? Prediction bboxes > Ground Truth bbox: detection_idx > best_gt_idx (i.e. amount_bboxes index)
        
        print(f"----- End Detection {detection_idx} ----\n")
        

        amount_bboxes
        for key, value in amount_bboxes.items():
            print(f'amount_bboxes[{key}]: {value}')
        
        print(f'True Postive: {TP}')
        print(f'True Postive: {FP}')
        
        
        print(f'---- End Class {c} ---- \n\n')
    
    
    
    

if __name__ == "__main__":
    #? Predicted and Ground truth bounding boxes
    pred_boxes = pred_boxes
    true_boxes = true_boxes

    map = mean_average_precision(pred_boxes, true_boxes)
    print('map:',map)