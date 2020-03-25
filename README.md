# Car Number Recognition

#### Dependencies

```bash
pip install -r requirements.txt
```

#### Training

```bash
python3 -m cnd.ocr.train_script -en ex1
```

#### Inference on images

```bash
python3 -m cnd.ocr.predictor --model models/best_model.pth --img-folder test_folder/
```

#### Inference on video

```bash
 python3 -m worker.run -vp test_folder/video/sample.mp4
```
