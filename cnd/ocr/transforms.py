import torch
import numpy as np
import PIL
import random
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as F
from albumentations import Blur, GaussNoise


class ImageNormalization(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image / 255.

class ToType(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype('float32')

class FromNumpyToTensor(object) :
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        return torch.from_numpy(image)

class FromNumpyToPIL(object):
    def __call__(self, image: np.ndarray) -> PIL.Image.Image:
        return Image.fromarray(image)

class FromPILToTensor(object):
    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        return F.to_tensor(image)

class FromPILToNumpy(object):
    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        return np.asarray(image)

class ToGrayScale(object):
    def __call__(self, image: PIL.Image.Image) ->  PIL.Image.Image:
        return F.to_grayscale(image)

class ScaleTransform(object):
    def __call__(self, image: PIL.Image.Image, out_size: Tuple) -> PIL.Image.Image:
        return F.resize(image, out_size)

class RandomCropTransform(object):
    def __call__(self, image: PIL.Image.Image, out_size: Tuple) ->  PIL.Image.Image:
        img_width, img_height = image.size
        out_height, out_width = out_size
        i = random.randint(0, img_height - out_height)
        j = random.randint(1, img_width - out_width)
        return F.crop(image, i, j, out_height, out_width)

class RandomFlipTransform(object):
    def __call__(self, image: PIL.Image.Image, horizontal: bool, vertical: bool) -> PIL.Image.Image:
        if horizontal:
            image = F.hflip(image)
        if vertical:
            image = F.vflip(image)
        return image

class BrightnessTransform(object):
    def __call__(self, image: PIL.Image.Image, factor: float) -> PIL.Image.Image:
        return F.adjust_brightness(image, factor)

class ContrastTransform(object):
    def __call__(self, image: PIL.Image.Image, factor: float) -> PIL.Image.Image:
        return F.adjust_contrast(image, factor)

class RotateTransform(object):
    def __call__(self, image: PIL.Image.Image, angle: float) -> PIL.Image.Image:
        return image.rotate(angle)

class BlurTransform(object):
    def __call__(self, image: np.ndarray, blur_limit: int=10) -> np.ndarray:
        return Blur(blur_limit=blur_limit, p=1)(image=image)['image']

class GaussNoiseTransfrorm(object):
    def __call__(self, image: np.ndarray, var_limit: Tuple=(0.1,0.5)) -> np.ndarray:
        return GaussNoise(var_limit=var_limit, p=1)(image=image)['image']

class InvertTransform(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return 1 - image


#TODO: Your transforms here
# Basic transforms:
# - Scale transform, to change size of input image
# - ToType transform, change type of image (usually image has type uint8)
# - ToTensor transform, move image to PyTorch tensor (to GPU?)
# - ImageNormalization, change scale of image from [0., ..., 255.] to [0., ..., 1.] (all float)
# Also you can add augmentations:
# - RandomCrop
# - RandomFlip
# - Brightness and Contrast augmentation
# Or any other, you can use https://github.com/albumentations-team/albumentations or any other lib