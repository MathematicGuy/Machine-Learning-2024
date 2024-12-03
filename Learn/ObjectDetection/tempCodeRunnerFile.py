import torch

# Create a 3D tensor
tensor = torch.rand(2, 3, 4)
print(tensor)

# Using [..., 0:1]
result_slice = tensor[..., 0:1]
print("Shape with slicing:", result_slice.shape)

# Using [..., 0]
result_index = tensor[..., 0]
print("Shape with indexing:", result_index.shape)