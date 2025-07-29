# Face Detection and Alignment Pipeline

This repository provides two modular, PyTorch-based pipelines for detecting and aligning human faces in images using **MTCNN**. One supports roll angle refinement using **HopeNet**, while the other offers a lightweight version focused solely on eye landmark-based alignment and rotation.

---

## Features

### Main (with HopeNet)

- Detect multiple faces in a single image using `facenet-pytorch` MTCNN
- Align faces by calculating roll angle from eye landmarks
- Optional: Refine roll angle using HopeNet model
- Handles both full-size images and pre-cropped face images
- Saves aligned faces and logs results in CSV format

### Lightweight Version (No HopeNet)

- Calculates roll angle based only on eye landmarks
- Automatically rotates and saves aligned faces
- Efficient inference with fallbacks and logging

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### A. HopeNet-based Pipeline

```bash
python face-alig.py --input img-1/ --output results/ --weights hopenet_weights_fixed.pth
```

#### Arguments:

| Argument    | Description                                            |
| ----------- | ------------------------------------------------------ |
| `--input`   | Path to input image or directory                       |
| `--output`  | Directory to save outputs (default: `output/`)         |
| `--weights` | (Optional) Path to HopeNet weights for roll refinement |
|             |                                                        |

### B. Alignment

```bash
python alignment.py --input path/to/images --output path/to/save --csv results.csv
```

#### Arguments:

| Argument   | Description                                 |
| ---------- | ------------------------------------------- |
| `--input`  | Directory containing input images           |
| `--output` | Directory to save aligned results           |
| `--csv`    | Path to CSV file where alignment results go |

---

## Output

- `output/aligned_faces/`: Aligned face crops (HopeNet version)
- `output/detections/`: Face detection visualization
- `output/alignment_results.csv`: Metadata with bounding boxes, angles, etc.
- `output/aligned/`: Aligned full-face images (lightweight version)
- `results.csv`: Angle, inference time, and status per image (lightweight version)

---

## Model Details

### HopeNet

- ResNet50-based model with 3 classification heads for yaw, pitch, and roll.
- Expected to have weights trained on a head pose dataset
- To use HopeNet, download pretrained weights and pass the file path via `--weights`

---

## Acknowledgments

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [HopeNet: Fine-Grained Head Pose Estimation](https://arxiv.org/abs/1809.04159)
- OpenCV, PIL, and PyTorch for image processing

