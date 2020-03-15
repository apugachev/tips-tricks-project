import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from argus.callbacks import Checkpoint
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.argus_model import CRNNModel
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from cnd.ocr.transforms import get_transforms
from cnd.ocr.metrics import StringAccuracy, LevDistance

from pathlib import Path
import torch

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
args = parser.parse_args()


EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]

BATCH_SIZE = 128

NUM_EPOCHS = 400

alphabet = " ABEKMHOPCTYX" + "".join([str(i) for i in range(10)])

CRNN_PARAMS = {"image_height": 32,
                        "number_input_channels": 1,
                        "number_class_symbols": len(alphabet),
                        "rnn_size": 16,
                    }

MODEL_PARAMS = {"nn_module":
                    ("CRNN", CRNN_PARAMS),
                "alphabet": alphabet,
                "loss": {"reduction":"mean"},
                "optimizer": ("Adam", {"lr": 0.00001}),
                "device": "cpu",
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

    model = CRNNModel(MODEL_PARAMS)

    callbacks = [
        Checkpoint(EXPERIMENT_DIR)
    ]

    metrics = [
        StringAccuracy(),
        LevDistance()
    ]

    model.fit(
        train_loader,
        val_loader=val_loader,
        max_epochs=NUM_EPOCHS,
        metrics=metrics,
        callbacks=callbacks,
        metrics_on_train=True,
    )
