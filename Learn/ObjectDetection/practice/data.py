# pred_boxes = [[train_idx, class_prediction, prob_score, x1, y1, x2, y2], ...]
pred_boxes = [
    # Class 0 (e.g., "cat")
    [0, 0, 0.9, 50, 50, 150, 150],
    [0, 0, 0.7, 55, 55, 145, 145],
    [1, 0, 0.6, 60, 60, 160, 160],
    [2, 0, 0.85, 45, 45, 155, 155],

    # Class 1 (e.g., "dog")
    [0, 1, 0.8, 30, 30, 120, 120],
    [1, 1, 0.5, 20, 20, 80, 80],

    # Class 2 (e.g., "car")
    [0, 2, 0.95, 200, 200, 300, 300],
    [1, 2, 0.8, 190, 190, 290, 290],

    # Class 3 (e.g., "bicycle")
    [0, 3, 0.6, 400, 400, 500, 500],
    [1, 3, 0.7, 390, 390, 510, 510],

    # Class 4 (e.g., "bird")
    [2, 4, 0.9, 100, 100, 200, 200],

    # Class 5 (e.g., "person")
    [1, 5, 0.85, 300, 300, 400, 400],
    [2, 5, 0.7, 310, 310, 420, 420],

    # Class 6 (e.g., "horse")
    [0, 6, 0.8, 50, 60, 150, 170],
    [1, 6, 0.6, 55, 65, 155, 175],

    # Class 7 (e.g., "cow")
    [2, 7, 0.9, 300, 320, 400, 420],
    [0, 7, 0.85, 310, 330, 410, 430],

    # Class 8 (e.g., "sheep")
    [1, 8, 0.75, 200, 210, 300, 310],
    [2, 8, 0.65, 220, 230, 320, 330],

    # Class 9 (e.g., "airplane")
    [0, 9, 0.9, 400, 450, 500, 550],

    # Class 10 (e.g., "boat")
    [1, 10, 0.92, 50, 50, 100, 100],
    [2, 10, 0.87, 55, 55, 105, 105],

    # Class 11 (e.g., "train")
    [0, 11, 0.88, 20, 20, 60, 60],
    [1, 11, 0.6, 25, 25, 65, 65],

    # Class 12 (e.g., "bus")
    [2, 12, 0.95, 150, 150, 200, 200],

    # Class 13 (e.g., "motorbike")
    [0, 13, 0.8, 250, 250, 300, 300],
    [1, 13, 0.85, 260, 260, 310, 310],

    # Class 14 (e.g., "train station")
    [2, 14, 0.92, 400, 400, 500, 500],

    # Class 15 (e.g., "traffic light")
    [0, 15, 0.9, 300, 300, 350, 350],

    # Class 16 (e.g., "tree")
    [1, 16, 0.87, 100, 120, 180, 200],

    # Class 17 (e.g., "house")
    [2, 17, 0.89, 50, 100, 120, 170],

    # Class 18 (e.g., "road")
    [0, 18, 0.88, 220, 270, 300, 350],

    # Class 19 (e.g., "bridge")
    [1, 19, 0.92, 450, 450, 550, 550],
]
# print(pred_boxes[:4])

true_boxes = [
    # Class 0 (e.g., "cat")
    [0, 0, 1.0, 44, 44, 150, 149],
    [0, 0, 1.0, 48, 48, 152, 152],
    [1, 0, 1.0, 58, 58, 158, 158],
    [2, 0, 1.0, 46, 46, 154, 154],

    # Class 1 (e.g., "dog")
    [0, 1, 1.0, 28, 28, 122, 122],
    [1, 1, 1.0, 18, 18, 82, 82],

    # Class 2 (e.g., "car")
    [0, 2, 1.0, 198, 198, 302, 302],
    [1, 2, 1.0, 188, 188, 292, 292],

    # Class 3 (e.g., "bicycle")
    [0, 3, 1.0, 398, 398, 502, 502],
    [1, 3, 1.0, 388, 388, 508, 508],

    # Class 4 (e.g., "bird")
    [2, 4, 1.0, 98, 98, 202, 202],

    # Class 5 (e.g., "person")
    [1, 5, 1.0, 298, 298, 402, 402],
    [2, 5, 1.0, 308, 308, 422, 422],

    # Class 6 (e.g., "horse")
    [0, 6, 1.0, 48, 58, 152, 168],
    [1, 6, 1.0, 53, 63, 158, 173],

    # Class 7 (e.g., "cow")
    [2, 7, 1.0, 298, 318, 402, 422],
    [0, 7, 1.0, 308, 328, 412, 432],

    # Class 8 (e.g., "sheep")
    [1, 8, 1.0, 198, 208, 302, 312],
    [2, 8, 1.0, 218, 228, 322, 332],

    # Class 9 (e.g., "airplane")
    [0, 9, 1.0, 398, 448, 502, 552],

    # Class 10 (e.g., "boat")
    [1, 10, 1.0, 48, 48, 102, 102],
    [2, 10, 1.0, 53, 53, 107, 107],

    # Class 11 (e.g., "train")
    [0, 11, 1.0, 18, 18, 62, 62],
    [1, 11, 1.0, 23, 23, 67, 67],

    # Class 12 (e.g., "bus")
    [2, 12, 1.0, 148, 148, 202, 202],

    # Class 13 (e.g., "motorbike")
    [0, 13, 1.0, 248, 248, 302, 302],
    [1, 13, 1.0, 258, 258, 312, 312],

    # Class 14 (e.g., "train station")
    [2, 14, 1.0, 398, 398, 502, 502],

    # Class 15 (e.g., "traffic light")
    [0, 15, 1.0, 298, 298, 352, 352],

    # Class 16 (e.g., "tree")
    [1, 16, 1.0, 98, 118, 182, 198],

    # Class 17 (e.g., "house")
    [2, 17, 1.0, 48, 98, 122, 172],

    # Class 18 (e.g., "road")
    [0, 18, 1.0, 218, 268, 302, 352],

    # Class 19 (e.g., "bridge")
    [1, 19, 1.0, 448, 448, 552, 552],
    
    [10, 20, 1.0, 448, 448, 552, 552],
    [10, 22, 1.0, 448, 448, 552, 552],
    [10, 23, 1.0, 448, 448, 552, 552],
]

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_boxes(pred_boxes, true_boxes):
    pred_indices = set(box[0] for box in pred_boxes)
    num_subplots = len(pred_indices)
    fig, axes = plt.subplots(1, num_subplots, figsize=(12 * num_subplots, 12))

    if num_subplots == 1:
        axes = [axes]

    for ax, pred_idx in zip(axes, pred_indices):
        ax.set_title(f'Prediction Index: {pred_idx}')
        
        # Plot predicted boxes
        for box in pred_boxes:
            if box[0] == pred_idx:
                rect = patches.Rectangle((box[3], box[4]), box[5] - box[3], box[6] - box[4], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[3], box[4], f'Pred: {box[1]} ({box[2]:.2f})', color='r', fontsize=12, verticalalignment='top')

        # Plot true boxes
        for box in true_boxes:
            if box[0] == pred_idx:
                rect = patches.Rectangle((box[3], box[4]), box[5] - box[3], box[6] - box[4], linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[3], box[4], f'True: {box[1]}', color='g', fontsize=12, verticalalignment='bottom')

        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.invert_yaxis()

    plt.show()

if __name__ == '__main__':
    visualize_boxes(pred_boxes, true_boxes)