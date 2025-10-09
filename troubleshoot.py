import torch
torch.cuda.is_available()
torch_staus = torch.cuda.is_available()
print(f'torch available: {torch_staus}')