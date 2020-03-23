import cv2

cv2.setNumThreads(0)
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
from worker.visualize_stream import VisualizeStream

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
    def __init__(self, name, video_path, save_path, fps=30, frame_size=(1280, 720), coord=(100, 100)):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, fps, frame_size, coord)
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
    #setup_logging(sys.argv[1], sys.argv[2])
    logger = logging.getLogger(__name__)
    project = None
    start = datetime.now()
    try:
        project = CNDProject("CNDProject", '/Users/alex/Downloads/3.mp4',
                             '/Users/alex/PycharmProjects/tips-tricks-project/experiments/res.mp4')
        project.start()
    except Exception as e:
        logger.exception(e)
    finally:
        time = (datetime.now() - start).total_seconds()
        print('TIME (seconds):', time)
        logger.info('TIME (seconds):', time)
        if project is not None:
            project.stop()
