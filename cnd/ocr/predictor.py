from argus.model import load_model
from cnd.ocr.transforms import *
from cnd.ocr.converter import strLabelConverter
import argparse
import pathlib
from cnd.config import CONFIG_PATH, Config
from cnd.ocr.transforms import get_transforms
from tabulate import tabulate


class Predictor:
    def __init__(self, model_path, image_size, device="cpu"):
        self.device = device
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size
        self.image_size = image_size
        self.alphabet = " ABEKMHOPCTYX" + "".join([str(i) for i in range(10)])
        self.converter = strLabelConverter(self.alphabet)

    def predict(self, images):

        logits = self.model.predict(images)
        len_images = torch.IntTensor([logits.size(0)] * logits.size(1))

        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text = self.converter.decode(preds, len_images)
        return text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", help="Path to folder with images", required=True)
    parser.add_argument("--model", help="Path to model", required=True)
    parser.add_argument("--cuda", action="store_true", help="Device (gpu or cpu)")
    args = parser.parse_args()

    test_paths = pathlib.Path(args.img_folder)
    device = "cuda" if args.cuda else "cpu"

    filepaths = list(test_paths.glob('**/*'))
    targets = [file.name.split('.')[0] for file in filepaths]
    filepaths = [str(file) for file in filepaths if file.is_file()]

    CV_CONFIG = Config(CONFIG_PATH)
    IMG_SIZE = CV_CONFIG.get('ocr_image_size')
    transforms = get_transforms(IMG_SIZE)

    images = torch.zeros(tuple([len(filepaths), 1] + IMG_SIZE))

    for i in range(len(filepaths)):
        images[i] = transforms(cv2.imread(filepaths[i]))

    p = Predictor(args.model, IMG_SIZE, device)
    preds = p.predict(images)

    result = [(t, p) for t,p in zip(targets, preds)]

    print(tabulate(result, headers=['Target', 'Predict']))