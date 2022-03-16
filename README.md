# Adversarial Training + Rotational Invariance of Transformers


<img align="center" src="assets/targeted_attack.gif" width="750">



## Introduction

We provide evidence against the unexpected trend of Vision Transformers (ViT) being not perceptually aligned with human visual representations by showing how a dual-stream Transformer ([CrossViT](https://github.com/IBM/CrossViT)) under a joint rotationally-invariant and adversarial optimization procedure yields 2nd place in the aggregate [Brain-Score](http://www.brain-score.org/) 2022 competition averaged across all visual categories, and currently (March 1st, 2022) holds the 1st place for the highest explainable variance of area V4. Against our initial expectations, these results provide tentative support for an *''All roads lead to Rome''* argument enforced via a joint optimization rule even for non biologically-motivated models of vision such as Vision Transformers

For more details please see our [BSW 2022 paper](https://openreview.net/forum?id=SOulrWP-Xb5&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DBrain-Score.org%2F2022%2FWorkshop%2FBSW%2FAuthors%23your-submissions)).



## Setup

1.  Install Python (>=3.7), PyTorch and other required python libraries with:
    ```
    pip install -r requirements.txt
    ```
2.  Download Imagenet dataset and [valprep.sh]() for preparing validation set:
    ```
    mkdir -p ./Dataset
    # Unzip data inside "Dataset"
    cd ./Dataset/val
    bash valprep.sh
    ``` 


## Usage
+ Generate or choose a config file from "Configs" folder and run the experiments:
```
python -u train_adv.py --data path/to/dataset --config path/to/config.yaml
``` 


## Pretrained weights

| ID | Description | Val. Acc(%)  |  Avg  | V1  | V2 | V4 | IT | Behavior |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|1057 | [CrossViT-18&dagger;]()              |83.05 | 0.442 | 0.473 | 0.274 | 0.478 | 0.484 | 0.500 |
|1095 | [CrossViT-18&dagger;+Rotation]()     | 79.22 | 0.458 | 0.458 | 0.288 | 0.495 | 0.503 | 0.547 |
|1084 | [CrossViT-18&dagger;+Adv]()          | 64.60 | 0.462 | 0.497 | 0.343 | 0.508 | 0.519 | 0.441 |
|991  | [CrossViT-18&dagger;+Rotation+Adv]() | 73.53 | 0.488 | 0.493 | 0.342 | 0.514 | 0.531 | 0.562 |




## Citation

If you find this useful for your work, please consider citing

```
@inproceedings{
anonymous2022joint,
title={Joint rotational invariance and adversarial training of a dual-stream Transformer yields state of the art Brain-Score for Area V4},
author={Anonymous},
booktitle={Submitted to Brain-Score Workshop},
year={2022},
url={https://openreview.net/forum?id=SOulrWP-Xb5},
note={under review}
}
```

