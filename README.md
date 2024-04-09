# EASY - Multi-granularity self-attention mechanisms for few-shot image classification.
This repository is the implementation of [EASY - Multi-granularity self-attention mechanisms for few-shot image classification).

EASY proposes a simple methodology, that reaches or even beats state of the art performance on multiple standardized benchmarks of the field, while adding almost no hyperparameters or parameters to those used for training the initial deep learning models on the generic dataset.

## Downloads 
Please click the [Google Drive link](https://drive.google.com/drive/) for downloading the datasets.

Each of the files (backbones and features) have the following prefixes depending on the backbone: 

|  Backbone  | prefix | Number of parameters |  
|:--------:|:------------:|:------------:|
| ResNet12 | | 12M|
| ResNet12(1/sqrt(2)) | small | 6M|
| Res2Net12 | 20M|
| Vit | 80M|


Suffixes <backbone_suffix>: 
- .pt11 : For 1-shot classification, the best backbone selected during training is based on the 1-shot performance of the validation dataset.
- .pt55 : For 5-shot classification, the best backbone selected during training is based on the 5-shot performance of the validation dataset.

## Testing scripts for EASY
Run scripts to evaluate the features on FSL tasks for Y and ASY. For EY and EASY use the corresponding features in 1-shot setting. For 5-shot setting, change --n-shots to 5

### Inductive setup using NCM
Test features on miniimagenet using Y (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11' --preprocessing ME --n-shots 1

Test features on miniimagenet using ASY (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeaturesAS1.pt11' --preprocessing ME --n-shots 1

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "['<path>/minifeatures1.pt11', '<path>/minifeatures2.pt11', '<path>/minifeatures3.pt11']" --preprocessing ME --n-shots 1
    
Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "['<path>/minifeaturesAS1.pt11', '<path>/minifeaturesAS2.pt11', '<path>/minifeaturesAS3.pt11']" --preprocessing ME --n-shots 1


### Transductive setup using Soft k-means
Test features on miniimagenet using Y (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11'--postprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using ASY (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeaturesAS1.pt11' --postprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "['<path>/minifeatures1.pt11', '<path>/minifeatures2.pt11', '<path>/minifeatures3.pt11']" --postrocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "['<path>/minifeaturesAS1.pt11', '<path>/minifeaturesAS2.pt11', '<path>/minifeaturesAS3.pt11']" --postrocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

## Training scripts for Y
Train a model on miniimagenet using manifold mixup, self-supervision and cosine scheduler. The best backbone is based on the 1-shot performance in the validation set. In order to get the best 5-shot performing model during validation, change --n-shots to 5 :

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME --n-shots 1 --skip-epochs 450 --save-model "<path>/mini<backbone_number>.pt1"

## Important Arguments
Some important arguments for our code.

**Training arguments**
- `dataset`: choices=['miniimagenet','tieredimagenet', 'fc100', 'cifarfs']
- `model`: choices=['vit','resnet12', 'resnet18', 'res2net', 'wideresnet', ]
- `dataset-path`: path of the datasets folder which contains folders of all the datasets.
- `rotations` : if mentionned, self-supervision will be used during training.
- `cosine` : if mentionned, cosine scheduler will be used during training.
- `save-model`: path where to save the best model based on validation data.
- `manifold-mixup`: number of epochs where to use manifold-mixup.
- `skip-epochs`: number of epochs to skip before evaluating few-shot performance. Used to speed-up training.
- `n-shots` : how many shots per few-shot run, can be int or list of ints. 

**Few-shot Classification**
- `preprocessing`: preprocessing sequence for few shot given as a string, can contain R:relu P:sqrt E:sphering and M:centering using the base data.
- `postprocessing`: postprocessing sequence for few shot given as a string, can contain R:relu P:sqrt E:sphering and M:centering on the few-shot data, used for transductive setting.

## Few-shot classification Results

Experimental results on few-shot learning datasets with ResNet-12 backbone. We report our average results with 10000 randomly sampled episodes for both 1-shot and 5-shot evaluations.






|LR+DC [17] |78.19 ± 0.25 |89.90 ± 0.41|
|EPNet [31] |78.50 ± 0.91 |88.36 ± 0.57|
|ODC [43] |85.22 ± 0.34| 91.35 ± 0.42|
|iLPC [45] |88.50 ± 0.75| 92.46 ± 0.42|
|PEMnE-BMS∗ [32]| 86.07 ± 0.25 |91.09 ± 0.14|
|EASY 3×ResNet12 (ours)| 84.29 ± 0.24| 89.76 ± 0.14|
