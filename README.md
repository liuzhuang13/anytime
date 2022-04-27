# [Anytime Dense Prediction with Confidence Adaptivity](https://openreview.net/forum?id=kNKFOXleuC)

Official PyTorch implementation for the following paper:

[Anytime Dense Prediction with Confidence Adaptivity](https://openreview.net/forum?id=kNKFOXleuC). ICLR 2022.\
[Zhuang Liu](https://liuzhuang13.github.io), [Zhiqiu Xu](https://www.linkedin.com/in/oscar-xu-1250821a1/), [Hung-ju Wang](https://www.linkedin.com/in/hungju-wang-5a5124172/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Evan Shelhamer](http://imaginarynumber.net/)\
UC Berkeley, Adobe Research

Our implementation is based upon [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

---
<p align="center">
<img src="https://user-images.githubusercontent.com/29576696/161406403-15c6da87-cd09-4203-adaa-09cc2badf1c1.jpeg" width=100% height=100% 
class="center">
</p>

Our full method, named **Anytime Dense Prediction with Confidence (ADP-C)**, achieves the same level of final accuracy with HRNet-w48, and meanwhile significantly reduces total computation.

### Main Results


|     Setting (HRNet-W48)     | model | exit1 | exit2 | exit3 |  exit4   | mean mIoU | exit1 | exit2 | exit3 |   exit4   | mean GFLOPs |
| ------------------------- | :---: | :---: | :---: | :---: | :------: | :---------: | :---: | :---: | :---: | :-------: | :---------: |
|          HRNet-W48          |   -   |   -   |   -   |   80.7   |      -      |   -   |   -   |   -   |   696.2   |      -      |
|         EE           | [model](https://drive.google.com/file/d/11AnwHiNmWZqtXbulJOGWTUCjlYfFweFB/view?usp=sharing) | 34.3  | 59.0  | 76.9  |   80.4   |    62.7     | 521.6 | 717.9 | 914.2 |  1110.5   |    816.0    |
|       EE + RH        | [model](https://drive.google.com/file/d/1zkgTRm8HyBqKA7dolM1i3AW6vs2V9AkE/view?usp=sharing) | 44.6  | 60.2  | 76.6  |   79.9   |    65.3     | 41.9  | 105.6 | 368.0 |   701.3   |    304.2    |
| ADP-C: EE + RH + CA  | [model](https://drive.google.com/file/d/1Un4XDqPOubGnKmm2vUis5CHM-EFOZsm0/view?usp=sharing) | 44.3  | 60.1  | 76.8  | **81.3** |  **65.7**   | 41.9  | 93.9  | 259.3 | **387.1** |  **195.6**  |



## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation on pretrained models

Download our pretrained model from the table above and specify its location by `TEST.MODEL_FILE`

**Early Exits (EE)**
```bash
python tools/test_ee.py --cfg experiments/cityscapes/w48.yaml \
TEST.MODEL_FILE <PRETRAINED MODEL>.pth
```
This should give
```
34.33	59.01	76.90	80.43	62.67
```

**Redesigned Heads (RH)**
```bash
python tools/test_ee.py --cfg experiments/cityscapes/w48.yaml \
EXIT.TYPE 'flex' EXIT.INTER_CHANNEL 128 \
TEST.MODEL_FILE <PRETRAINED MODEL>.pth
```

This should give
```
44.61	60.19	76.64	79.89	65.33
```

**ADP-C (EE + RH + CA)**
```bash
python tools/test_ee.py \
--cfg experiments/cityscapes/w48.yaml MODEL.NAME model_anytime  \
EXIT.TYPE 'flex' EXIT.INTER_CHANNEL 128 \
MASK.USE True MASK.CONF_THRE 0.998 \
TEST.MODEL_FILE <PRETRAINED MODEL>.pth
```

This should give
```
44.34	60.13	76.82	81.31	65.65
```


## Train

There are two configurations for the backbone HRnet model, `w48.yaml` and `w18.yaml` under `experimens/cityscapes`. Note that the following commands are for using `HRNet-w48` as backbone. Please change `EXIT.INTER_CHANNEL` to `64` when using `w18` as backbone.

**Early Exits (EE)**

````bash
python -m torch.distributed.launch tools/train_ee.py \
--cfg experiments/cityscapes/w48.yaml
````


**Redesigned Heads (RH)**

````bash
python -m torch.distributed.launch tools/train_ee.py \
--cfg experiments/cityscapes/w48.yaml \
EXIT.TYPE 'flex' EXIT.INTER_CHANNEL 128
````


**Confidence Adatative (CA)**

````bash
python -m torch.distributed.launch tools/train_ee.py \
--cfg experiments/cityscapes/w48.yaml \
MASK.USE True MASK.CONF_THRE 0.998
````


**ADP-C (EE + RH + CA)**

````bash
python -m torch.distributed.launch tools/train_ee.py \
--cfg experiments/cityscapes/w48.yaml \
EXIT.TYPE 'flex' EXIT.INTER_CHANNEL 128 \
MASK.USE True MASK.CONF_THRE 0.998
````

Evaulation results will be generated at the end of training.

- `result.txt`: contains mIOU for each exit and the average mIOU of the four exits. 

- `test_stats.json`: contains FLOPs and number of parameters.

- `final_state.pth`: the trained model file.

- `config.yaml`: the configuration file. 

## Test

**Evaluation** 

```
python tools/test_ee.py --cfg <Your output directoy>/config.yaml
```

**Customized Evaluation**

```
python tools/test.py --cfg experiments/cityscapes/<Your config file>.yaml \
TEST.MODEL_FILE <Your model>.pth 
```

## Acknowledgement
This repository is built upon [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022anytime,
  author  = {Zhuang Liu and Zhiqiu Xu and Hung-Ju Wang and Trevor Darrell and Evan Shelhamer},
  title   = {Anytime Dense Prediction with Confidence Adaptivity},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2022},
}
```
