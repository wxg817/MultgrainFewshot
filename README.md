#  Multi-granularity self-attention mechanisms for few-shot image classification.
This repository is the implementation of [EASY - Multi-granularity self-attention mechanisms for few-shot image classification).

EASY proposes a simple methodology, that reaches or even beats state of the art performance on multiple standardized benchmarks of the field, while adding almost no hyperparameters or parameters to those used for training the initial deep learning models on the generic dataset.

## Downloads 
The following datasets are used to evaluate universal meta-learning performance.
CIFAR-FS
Mini ImageNet
Tiered Mini ImageNet


Each of the files (backbones and features) have the following prefixes depending on the backbone: 

|  Backbone  | prefix | Number of parameters |  
|:--------:|:------------:|:------------:|
| ResNet12 | | 6M|
| ResNet18 | |11.2M|
| ResNet34 | |22M |
| ResNet50||26M |
| Res2Net12 | 20M|
| Multi-Vit | 80M|


Suffixes <backbone_suffix>: 
- .pt11 : For 1-shot classification, the best backbone selected during training is based on the 1-shot performance of the validation dataset.
- .pt55 : For 5-shot classification, the best backbone selected during training is based on the 5-shot performance of the validation dataset.

## Testing scripts for EASY
Run scripts to evaluate the features on FSL tasks for Y and ASY. For EY and EASY use the corresponding features in 1-shot setting. For 5-shot setting, change --n-shots to 5

### 
Test features on miniimagenet using Y (Multi-vit)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model vit --test-features '<path>/minifeatures1.pt11' --preprocessing ME --n-shots 1

Test features on miniimagenet using ASY (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeaturesAS1.pt11' --preprocessing ME --n-shots 1

T

**Training arguments**
- `dataset`: choices=['miniimagenet','tieredimagenet', 'cifarfs']
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

Experimental results on few-shot learning datasets with ResNet-12,18,34,50 backbone. We report our average results with 500 randomly sampled episodes for both 1-shot and 5-shot evaluations.

