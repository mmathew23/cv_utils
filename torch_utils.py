import torch
import cv2


def cv2torch(img):
    """Converts an image from OpenCV to PyTorch"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255)


def torch2cv(tensor):
    """Converts an image from PyTorch to OpenCV"""
    img = tensor.mul(255).byte().numpy()
    img = img.transpose(1, 2, 0)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)