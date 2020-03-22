import logging
import numpy as np
import torch
from threading import Thread

import cv2
from worker.state import State
from worker.video_reader import VideoReader
from worker.video_writer import VideoWriter
from worker.ocr_stream import OcrStream
from cnd.ocr.transforms import get_transforms
from cnd.config import CONFIG_PATH, Config


class Visualizer:
    def __init__(self, state: State, coord, color=(255, 255, 255), thick=3, font_scale=3, font=cv2.FONT_HERSHEY_SIMPLEX):
        self.state = state
        self.coord_x, self.coord_y = coord
        self.color = color
        self.thickness = thick
        self.font_scale = font_scale
        self.font = font

    def _draw_ocr_text(self, frames):
        text = self.state.text

        for i in range(len(frames)):
            if text[i]:
                frames[i] = cv2.putText(frames[i], text[i // 20 * 20], (self.coord_x, self.coord_y), self.font, self.font_scale,
                                        self.color, self.thickness)
        return frames

    def __call__(self, frames):
        frames = self._draw_ocr_text(frames)
        return frames


class VisualizeStream:
    def __init__(self, name,
                 in_video: VideoReader,
                 state: State, video_path, fps, frame_size, coord):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.coord = coord
        self.fps = fps
        self.frame_size = tuple(frame_size)
        self.ocr_size = Config(CONFIG_PATH).get('ocr_image_size')

        self.out_video = VideoWriter("VideoWriter", video_path, self.fps, self.frame_size)
        self.in_video = in_video
        self.stopped = True
        self.visualize_thread = None
        self.visualizer = Visualizer(self.state, self.coord)
        self.ocr_stream = self.ocr_stream = OcrStream("OcrStream", self.state, self.in_video)

        self.logger.info("Create VisualizeStream")

    def _visualize(self):
        try:
            while True:
                if self.stopped:
                    return

                frames_main = np.zeros((self.fps, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                frames_resized = torch.zeros((self.fps, 1, self.ocr_size[0], self.ocr_size[1]))

                for i, pos in enumerate(range(self.fps)):
                    frame = self.in_video.read()
                    frame = cv2.resize(frame, self.frame_size)
                    frames_main[i] = frame
                    frame = get_transforms(self.ocr_size)(frame)
                    frames_resized[i] = frame

                self.state.text = self.ocr_stream(frames_resized)
                result_frames = self.visualizer(frames_main)
                
                for frame in result_frames:
                    self.out_video.write(frame)

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def start(self):
        self.logger.info("Start VisualizeStream")
        self.stopped = False
        self.visualize_thread = Thread(target=self._visualize, args=())
        self.visualize_thread.start()
        self.in_video.start()

    def stop(self):
        self.logger.info("Stop VisualizeStream")
        self.stopped = True
        self.out_video.stop()
        if self.visualize_thread is not None:
            self.visualize_thread.join()