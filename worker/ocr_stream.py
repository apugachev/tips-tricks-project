import logging
from cnd.ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader


class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader, model_path):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.end = False
        self.video_reader = video_reader
        self.ocr_thread = None

        self.predictor = Predictor(model_path, [32,96])

        self.logger.info("Create OcrStream")

    def __call__(self, frames):
        pred_text = self.predictor.predict(frames)
        return pred_text
