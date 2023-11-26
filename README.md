# BeyondAGrayscalePerception
![](./example.jpeg)
## Installation

### Requirements
- Python >= 3.7
- PyTorch >= 1.7

### Install with conda (Recommend)

```
conda create -n bgpcolor python=3.8
conda activate bgpcolor

pip install -r requirements.txt
```
## Train

1. Dataset preparation: Download [COCOStuff](https://github.com/nightrome/cocostuff) dataset, or prepare any custom dataset of your own. For this follow thw following steps:
    1. Download the images from [COCO](https://github.com/nightrome/cocostuff#versions-of-coco-stuff) with the following commands:
    ```
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    ```
    2. Unzip the files:
    ```
    unzip train2017.zip
    unzip val2017.zip
    ```

## Baseline 
In order to run the file, you can add an argument --mode with 3 different configurations (Train, to train the model, Test, to evalaute de model and Demo to save the prediction of an image)

## Beta and Gamma Models

In order to run the file, you can add an argument --mode with 3 different configurations (Train, to train the model, Test, to evalaute de model and Demo to save the prediction of an image)
Or you can find the individual scripts in the BETAGAMMA directory

## ECCV16 and siggraph17
Yu can find the individual scripts in the CIC directory

## Saved Models

| Model | Route |
|----------|----------|
| Baseline   |  /media/disk2/lvchaparro/AML/PROYECTO/NEWCOLOR/CODIGO FINAL/models/Baseline/modeloBaseline.pt   | 
| Gamma    | /media/disk2/lvchaparro/AML/PROYECTO/NEWCOLOR/CODIGO FINAL/models/gamma/betagammag.pt   | 
| Gamma (PL)    | /media/disk2/lgomez/COLORIZATION/models/modelgammap.pt   | 
| Beta    |/media/disk2/lvchaparro/AML/PROYECTO/NEWCOLOR/CODIGO FINAL/models/beta/betagammab.pt   | 
| Beta (PL)    | /media/disk2/lgomez/COLORIZATION/models/modelbetap.pt   | 
