import torch

amount_boxes = [{0: torch.tensor([0., 0.])}, {1: torch.tensor([0.])}, {2: torch.tensor([0.])}]

print("Original value:", amount_boxes[0])

# Update the value at index 0, 0
amount_boxes[0][0] = torch.tensor([1])
print("Updated value:", amount_boxes[0][0])
