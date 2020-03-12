import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.model import CRNN
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from cnd.ocr.transforms import get_transforms
from cnd.ocr.metrics import WrapCTCLoss
from catalyst.dl import SupervisedRunner, CheckpointCallback
from cnd.ocr.metrics import WrapAccuracyScore, WrapLevenshteinScore
from pathlib import Path
import torch

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]
# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 128
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 100

alphabet = " "
alphabet += "ABEKMHOPCTYX"
alphabet += "".join([str(i) for i in range(10)])

MODEL_PARAMS = {
    "image_height": 32,
    "number_input_channels": 1,
    "number_class_symbols": len(alphabet),
    "rnn_size": 128,
}

if __name__ == "__main__":

    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(CV_CONFIG.get('ocr_image_size'))

    path = CV_CONFIG.get('data_path')
    dataset_paths = Path(path)

    filepaths = list(dataset_paths.glob('**/*'))[1:]
    filepaths = [file for file in filepaths if file.is_file()]
    filepaths = [file for file in filepaths if not file.name.startswith('.')]

    train_paths, val_paths = train_test_split(filepaths, random_state=6)

    train_dataset = ConcatDataset([
        OcrDataset(train_paths, transforms)
    ])

    val_dataset = ConcatDataset([
        OcrDataset(val_paths, transforms)
    ])


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CRNN(**MODEL_PARAMS)
    # YOU CAN ADD CALLBACK IF IT NEEDED, FIND MORE IN
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # define callbacks if any
    callbacks = [CheckpointCallback(save_n_best=10), WrapAccuracyScore(), WrapLevenshteinScore()]
    # input_keys - which key from dataloader we need to pass to the model
    runner = SupervisedRunner(input_key="image", input_target_key="targets")

    runner.train(
        model=model,
        criterion=WrapCTCLoss(alphabet),
        optimizer=optimizer,
        scheduler=scheduler,
        loaders={'train': train_loader},
        logdir="./logs/ocr",
        num_epochs=NUM_EPOCHS,
        verbose=True,
        callbacks=callbacks
    )
