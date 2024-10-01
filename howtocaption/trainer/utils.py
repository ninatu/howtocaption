import torch

def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    else:
        return data