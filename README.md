# TB-Net

[TB-Net: A Three-Stream Boundary-Aware Network for Fine-Grained Pavement Disease Segmentation](https://arxiv.org/pdf/2011.03703.pdf)

## Introduction

Fine-grained pavement disease segmentation aims to not only automatically segment cracks, but also segment other complex pavement diseases as well as typical landmarks (markings, runway lights, etc.) and commonly seen water/oil stains. TB-Net consists of three streams fusing the low-level spatial and the high-level contextual representations as well as the detailed boundary information.

## Requirements

* Python 2
* Tensorflow 1.8.0
* Numpy
* Scipy
* Opencv

## Instructions

### Dataset Preparation

Set up the dataset folder in the following structure:
```
|—— "dataset_name"
|   |—— train: gray-scale images for training
|   |—— train_labels: annotations for training images
|   |—— train_edge_labels: binary boundary annotations for training
|   |—— val: gray-scale images for validation
|   |—— val_labels: annotations for validation images
|   |—— val_edge_labels: binary boundary annotations for validation
|   |—— test: gray-scale images for testing
|   |—— test_labels: annotations for testing images
|   |—— test_edge_labels: binary boundary annotations for testing
```

The class dictionary file "class_dict.csv" contains the list of classes along with the R, G, B color labels:
```
name,r,g,b
background,0,0,0
crack,128,0,0
cornerfracture,0,128,0
seambroken,128,128,0
patch,0,0,128
repair,128,0,128
slab,0,128,128
track,128,128,128
light,64,0,0
```

### Training

Run `python train_tbnet.py` with the following args:
* `--num_epochs`: Number of epochs to train for.
* `--dataset`: The dataset to use.
* `--resize_height`: Height of resized input image to network.
* `--resize_width`: Width of resized input image to network.
* `--batch_size`: Number of images in each batch.
* `--num_val_images`: The number of images to use for validation.

### Testing

Run `python test_tbnet.py` with the following args:
* `--checkpoint_path`: The path to the checkpoint weights.
* `--resize_height`: Height of resized input image to network.
* `--resize_width`: Width of resized input image to network.
* `--dataset`: The dataset to use.

### Reference
[Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/tree/master)
