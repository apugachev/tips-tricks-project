# HERE YOUR PREDICTOR
import torch
from cnd.ocr.model import CRNN
from cnd.ocr.transforms import *
from cnd.ocr.metrics import WrapCTCLoss
from torchvision.transforms import Compose
from cnd.ocr.converter import strLabelConverter


class Predictor:
    def __init__(self, model_path, model_params, image_size):

        self.checkpoint = torch.load(model_path)
        self.model = CRNN(**model_params)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        self.image_size = image_size
        self.transform = Compose([
            ToGrayScale(),
            ScaleTransform(self.image_size),
            ImageNormalization(),
            FromNumpyToTensor()])
        self.alphabet = " ABEKMHOPCTYX" + "".join([str(i) for i in range(10)])
        self.converter = strLabelConverter(self.alphabet)

    def predict(self, images):

        if len(images.shape) == 3:
            images_new = self.transform(images)[None, :, :, :]
        else:
            self.image_size = [len(images), 1] + self.image_size
            images_new = torch.zeros(self.image_size)

            for i, img in enumerate(images):
                images_new[i] = self.transform(img)

        logits = self.model(images_new)
        len_images = torch.IntTensor([logits.size(0)] * logits.size(1))

        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text = self.converter.decode(preds, len_images)
        return text

if __name__ == "__main__":

    alphabet = " ABEKMHOPCTYX"
    alphabet += "".join([str(i) for i in range(10)])

    path = '/Users/alex/PycharmProjects/tips-tricks-project/cnd/ocr/logs/ocr/checkpoints/best_full.pth'
    model_params = {
        "image_height": 32,
        "number_input_channels": 1,
        "number_class_symbols": len(alphabet),
        "rnn_size": 64,
    }

    pic = cv2.imread('/Users/alex/PycharmProjects/tips-tricks-project/CropNumbers/NumBase/Y446YK 19726.bmp')

    p = Predictor(path, model_params, [32,96])
    print(p.predict(pic))