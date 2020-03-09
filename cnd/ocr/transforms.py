import torch
import numpy as np
import cv2
from typing import Tuple, List
from torchvision.transforms import Compose
from albumentations import Blur, GaussNoise, Resize, RandomBrightness, RandomContrast, \
                            Rotate, InvertImg, RandomCrop


class ImageNormalization(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image / 255.

class FromNumpyToTensor(object) :
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        image = image[np.newaxis,:,:]

        return torch.from_numpy(image)

class ScaleTransform(object):
    def __init__(self, img_size: List):
        self.height = img_size[0]
        self.width = img_size[1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return Resize(self.height, self.width)(image=image)['image']

class ToGrayScale(object):
    def __call__(self, image: np.ndarray) ->  np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

class BrightnessTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return RandomBrightness()(image=image)['image']

class ContrastTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return RandomContrast()(image=image)['image']

class RotateTransform(object):
    def __init__(self, angle: int=30):
        self.angle = angle

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return Rotate(limit=self.angle, border_mode=1)(image=image)['image']

class BlurTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return Blur(blur_limit=3)(image=image)['image']

class GaussNoiseTransfrorm(object):
    def __init__(self, var_limit: Tuple=(75.0, 250.0)):
        self.var_limit = var_limit

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return GaussNoise(var_limit=self.var_limit)(image=image)['image']

class InvertTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return InvertImg(p=0.2)(image=image)['image']

class RandomCropTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        height = np.random.randint(image.shape[0] // 2, image.shape[0])
        width = np.random.randint(image.shape[1] // 2, image.shape[1])
        return RandomCrop(height, width)(image=image)['image']

def get_transforms(image_size):
    transform = Compose([
        ToGrayScale(),
        BrightnessTransform(),
        ContrastTransform(),
        RotateTransform(),
        BlurTransform(),
        GaussNoiseTransfrorm(),
        InvertTransform(),
        RandomCropTransform(),
        ImageNormalization(),
        ScaleTransform(image_size),
        FromNumpyToTensor()])
    return transform