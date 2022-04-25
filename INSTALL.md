# Installation
We provide installation instructions for Cityscapes segmentation experiments here.
## Dependency Setup
Create a new conda virtual environment
```
conda create -n anytime python=3.8 -y
conda activate anytime
```
Install `PyTorch=1.1.0`
```
pip install torch==1.1.0
```
Clone this repo and install required packages
```
pip install -r requirements.txt
```

## Data preparation
Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and place a symbolic link under the `data` folder.

```
mkdir data
ln -s $DATA_ROOT data
```

Structure the data as follows
````
$ROOT/data
└── cityscapes
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
        ├── test
        ├── train
        └── val

````

## Pretrained model preparation
Create a folder named `pretrained_models` under the root directory.
```
mkdir pretrained_models
```
Download the [HRNet-W18-C-Small-v2](https://1drv.ms/u/s!Aus8VCZ_C_33gRmfdPR79WBS61Qn?e=HVZUi8) and [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification.git)
and structure the directory as follows
```
pretrained_models
├── hrnet_w18_small_model_v2.pth
└── hrnetv2_w48_imagenet_pretrained.pth
```
