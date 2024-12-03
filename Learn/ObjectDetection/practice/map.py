import torch 
from iou import intersection_over_union
from data import true_boxes, pred_boxes
from collections import Counter


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
    debug = False
    average_precisions = []  
    #! Goal: classify TP and FP for each pred_boxes to calc precision and recall 
    for c in range(num_classes):
        # print('class:', c)
        # get all boxes of the same class (e.g. dog, cat, table, etc..)
        detections = []
        ground_truths = []
        
        for box in pred_boxes:
            if box[1] == c:
                detections.append(box)
                
        for box in true_boxes:
            if box[1] == c:
                ground_truths.append(box)
        
        
        # Count total true_box of each class with train_idx/box[0] as key
        gt_amount = Counter([box[0] for box in ground_truths]) # Counter({0: 2, 1: 1, 2: 1})
        detections.sort(key=lambda box:box[2], reverse=True)
        
        
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_truth_box = len(ground_truths)
        
        if total_truth_box == 0:
            average_precisions.append(0) 
            continue
        
        
        gt_amount = {key: torch.zeros(val) for key, val in gt_amount.items()}
        
        
        # #? choose the best detection for each gt by IoU
        # # get detection pred_ifx and gt index to assign 1 for match gt in gt_amount[detection[0]][gt_idx]
        for detection_idx, detection in enumerate(detections):
            # get gt_boxes in the same image (i.e. pred_idx)
            gt_boxes = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            # print('gt_boxes:\n',gt_boxes)
            
            best_iou = 0 
            best_iou_idx = -1 # -1 indicate "invalid index" or "not found" 
            
            
            for gt_idx, gt in enumerate(gt_boxes):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)   
                
                # print(iou)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = gt_idx
               
                  
            if debug: 
                print('best_iou:',best_iou)
                print('pred_idx:', detection[0])
                print('best_iou_idx:',best_iou_idx)
                print('check match:',gt_amount)   
            
            #? Case 1: if iou(pred_box, gt_box) > threshold: TP else FP
            if best_iou > iou_threshold and best_iou_idx != -1: 
                if gt_amount[detection[0]][best_iou_idx] == 0:
                    TP[detection_idx] = 1 # since TP = len(detections)
                    gt_amount[detection[0]][best_iou_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            #? Case 2: if detection have no match
            else:
                FP[detection_idx] = 1
      
            print()
            
        print('final gt_amount:', gt_amount)
    
    
        epsilon = 1e-8
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        print("Class precision and Recall:")
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_truth_box + epsilon)
       
        precisions = torch.cat((torch.tensor([1], device=TP.device), precisions))
        recalls = torch.cat((torch.tensor([0], device=TP.device), recalls))

        # print(f'precision: {TP_cumsum}/ {TP_cumsum} + {FP_cumsum} = {precisions}')    
        # print(f'recall: {TP_cumsum} / {TP_cumsum} + {FP_cumsum} = {recalls}')
        
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        print()
    
    # return 0 if there are no average precision    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0
if __name__ == "__main__":
    #? Predicted and Ground truth bounding boxes
    pred_boxes = pred_boxes
    true_boxes = true_boxes
    map = mean_average_precision(pred_boxes, true_boxes)
    print('MAP:',map)