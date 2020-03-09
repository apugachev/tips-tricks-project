# HERE YOUR PREDICTOR
import torch


class Predictor:
    def __init__(self, model_path, image_size, converter, transforms):

        self.model = torch.load(model_path)
        self.ocr_image_size = image_size
        self.transform = transforms
        self.converter = converter

    def predict(self, images):
        #TODO: check for correct input type, you can receive one image [x,y,3] or batch [b,x,y,3]
        images = self.transform(images)
        self.model = self.model.model_state_dict
        pred = self.model.predict({"image": images})
        text = self.converter(pred)
        return text
