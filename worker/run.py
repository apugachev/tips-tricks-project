import cv2

cv2.setNumThreads(0)
import logging
import argparse
from datetime import datetime
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
from worker.visualize_stream import VisualizeStream

parser = argparse.ArgumentParser()
parser.add_argument("-vp", "--video_path", help = "Path to video file", required=True)
parser.add_argument("-sp", "--save_path", help = "Save path for video", default='experiments/result.mp4')
parser.add_argument("-m", "--model_path", help = "Path to model", default='models/best_model_colab_700.pth')
parser.add_argument("-fp", "--pred_path", help = "Path to frames prediction file", default='experiments/pred_frames.txt')
parser.add_argument("-log", "--log_path", help="Logging file", default='experiments/logs/video_logs.txt')
parser.add_argument("-lvl", "--log_level", help="Level for logging", default='INFO')
args = parser.parse_args()


def setup_logging(path, level='INFO'):
    handlers = [logging.StreamHandler()]
    file_path = path
    if file_path:
        file_handler = RotatingFileHandler(filename=file_path,
                                           maxBytes=10 * 10 * 1024 * 1024,
                                           backupCount=5)
        handlers.append(file_handler)
    logging.basicConfig(
        format='[{asctime}][{levelname}] - {name}: {message}',
        style='{',
        level=logging.getLevelName(level),
        handlers=handlers,
    )


class CNDProject:
    def __init__(self, name, video_path, save_path, model_path, pred_path, fps=30, frame_size=(1280, 720), coord=(100, 100)):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, model_path, pred_path, fps, frame_size, coord)
        self.logger.info("Start Project")

    def start(self):
        self.logger.info("Start project act start")
        try:
            self.video_reader.start()
            self.visualize_stream.start()
            self.state.exit_event.wait()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stop Project")
        self.video_reader.stop()
        self.visualize_stream.stop()


if __name__ == '__main__':
    setup_logging(args.log_path, args.log_level)
    logger = logging.getLogger(__name__)
    project = None
    start = datetime.now()
    open(args.pred_path, 'w').close()

    try:
        project = CNDProject("CNDProject", args.video_path, args.save_path, args.model_path, args.pred_path)
        project.start()
    except Exception as e:
        logger.exception(e)
    finally:
        if project is not None:
            project.stop()
        time = (datetime.now() - start).total_seconds()
        logger.info('TOTAL TIME (seconds): ' + str(time))
