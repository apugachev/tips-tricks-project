# HERE YOUR PREDICTOR
import torch
from cnd.ocr.model import CRNN


class Predictor:
    def __init__(self, model_path, model_params, image_size, converter, transforms):

        self.checkpoint = torch.load(model_path)
        self.model = CRNN(**model_params)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms
        self.converter = converter

    def predict(self, images):

        if len(images.shape) == 3:
            images_new = self.transform(images)[None, :, :, :]
        else:
            self.image_size = [len(images), 1] + self.image_size
            images_new = torch.zeros(self.image_size)

            for i, img in enumerate(images):
                images_new[i] = self.transform(img)

        pred = self.model(images_new)
        batch_len = torch.IntTensor([pred.shape[1]])

        text = self.converter.preds_converter(pred, batch_len)[0]
        return text
