# HERE YOUR PREDICTOR
from argus.model import load_model
from cnd.ocr.transforms import *
from torchvision.transforms import Compose
from cnd.ocr.converter import strLabelConverter


class Predictor:
    def __init__(self, model_path, image_size, device="cpu"):
        self.device = device
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size
        self.image_size = image_size
        self.transform = Compose([
            ToGrayScale(),
            ScaleTransform(self.image_size),
            ImageNormalization(),
            FromNumpyToTensor()])
        # alphabet = " "
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

        logits = self.model.predict(images_new)
        len_images = torch.IntTensor([logits.size(0)] * logits.size(1))

        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text = self.converter.decode(preds, len_images)
        return text

if __name__ == "__main__":

    alphabet = " ABEKMHOPCTYX"
    alphabet += "".join([str(i) for i in range(10)])

    path = '/Users/alex/PycharmProjects/tips-tricks-main-repo/Tips-Tricks/project/experiments/ex1/model-000-6.984760.pth'


    pic = cv2.imread('/Users/alex/PycharmProjects/tips-tricks-project/CropNumbers/NumBase/Y446YK 19726.bmp')
    pic2 = cv2.imread('/Users/alex/PycharmProjects/tips-tricks-project/CropNumbers/NumBase/P494KE 19793.bmp')
    pica = np.stack((pic, pic))
    print(pica.shape)
    p = Predictor(path, [32, 96])
    print(p.predict(pica))