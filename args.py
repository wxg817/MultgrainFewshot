import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description="""Optimized code for training usual datasets/model
""", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--batch-fs", type=int, default=20, help="batch size for few shot runs")
parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
parser.add_argument("--lr", type=float, default="0.1", help="initial learning rate (negative is for Adam, e.g. -0.001)")
parser.add_argument("--epochs", type=int, default=350, help="total number of epochs")
parser.add_argument("--milestones", type=str, default="100", help="milestones for lr scheduler, can be int (then milestones every X epochs) or list. 0 means no milestones")
parser.add_argument("--gamma", type=float, default=-1., help="multiplier for lr at milestones")
parser.add_argument("--cosine", action="store_true", help="use cosine annealing scheduler with args.milestones as T_max")
parser.add_argument("--mixup", action="store_true", help="use of mixup since beginning")
parser.add_argument("--mm", action="store_true", help="to be used in combination with mixup only: use manifold_mixup instead of classical mixup")
parser.add_argument("--label-smoothing", type=float, default=0, help="use label smoothing with this value")
parser.add_argument("--dropout", type=float, default=0, help="use dropout")
parser.add_argument("--rotations", action="store_true", help="use of rotations self-supervision during training")
parser.add_argument("--model", type=str, default="vit", help="model to train")
parser.add_argument("--preprocessing", type=str, default="", help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
parser.add_argument("--postprocessing", type=str, default="", help="postprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")

parser.add_argument("--manifold-mixup", type=int, default="0", help="deploy manifold mixup as fine-tuning as in S2M2R for the given number of epochs")
parser.add_argument("--temperature", type=float, default=1., help="multiplication factor before softmax when using episodic")
parser.add_argument("--ema", type=float, default=0, help="use exponential moving average with specified decay (default, 0 which means do not use)")

### pytorch options
parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
# parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASETS"), help="dataset path")
parser.add_argument("--dataset-path", type=str, default="./datasets", help="dataset path")
parser.add_argument("--dataset-device", type=str, default="", help="use a different device for storing the datasets (use 'cpu' if you are lacking VRAM)")
parser.add_argument("--deterministic", action="store_true", help="use desterministic randomness for reproducibility")

### run options
parser.add_argument("--skip-epochs", type=int, default="0", help="number of epochs to skip before evaluating few-shot performance")
parser.add_argument("--runs", type=int, default=1, help="number of runs")
parser.add_argument("--quiet", action="store_true", help="prevent too much display of info")
parser.add_argument("--dataset", type=str, default="miniImageNet", help="dataset to use")
parser.add_argument("--dataset-size", type=int, default=-1, help="number of training samples (using a subset for classical classification, and reducing size of epochs for few-shot)")
parser.add_argument("--output", type=str, default="", help="output file to write")
parser.add_argument("--save-features", type=str, default="", help="save features to file")
parser.add_argument("--save-model", type=str, default="", help="save model to file")
parser.add_argument("--test-features", type=str, default="", help="test features and exit")
parser.add_argument("--load-model", type=str, default="", help="load model from file")
parser.add_argument("--seed", type=int, default=-1, help="set random seed manually, and also use deterministic approach")
parser.add_argument("--wandb", type=str, default='', help="Report to wandb, input is the entity name")

### few-shot parameters
parser.add_argument("--n-shots", type=str, default="[1,5]", help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
parser.add_argument("--n-runs", type=int, default=10000, help="number of few-shot runs")
parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
parser.add_argument("--n-queries", type=int, default=15, help="number of few-shot queries")
parser.add_argument("--sample-aug", type=int, default=1, help="number of versions of support/query samples (using random crop) 1 means no augmentation")
parser.add_argument("--ncm-loss", action="store_true", help="use ncm output instead of linear")
parser.add_argument("--episodic", action="store_true", help="use episodic training")
parser.add_argument("--episodes-per-epoch", type=int, default=100, help="number of episodes per epoch")
# only for transductive, used with "test-features"
parser.add_argument("--transductive", action="store_true", help ="test features in transductive setting")
parser.add_argument("--transductive-softkmeans", action="store_true", help="use softkmeans for few-shot transductive")
parser.add_argument("--transductive-n-iter", type=int, default=50, help="number of iterations for few-shot transductive")
parser.add_argument("--transductive-n-iter-sinkhorn", type=int, default=200, help="number of iterations of sinkhorn for few-shot transductive")
parser.add_argument("--transductive-temperature", type=float, default=14, help="temperature for few-shot transductive")
parser.add_argument("--transductive-temperature-softkmeans", type=float, default=20, help="temperature for few-shot transductive is using softkmeans")
parser.add_argument("--transductive-alpha", type=float, default=0.84, help="momentum for few-shot transductive")
parser.add_argument("--transductive-cosine", action="store_true", help="use cosine similarity for few-shot evaluation")

try :
    # get_ipython()
    args = parser.parse_args(args=[])
except :
    args = parser.parse_args()

### process arguments
if args.dataset_device == "":
    args.dataset_device = args.device
    
if args.dataset_path[-1] != '/':
    args.dataset_path += "/"

if args.device[:5] == "cuda:" and len(args.device) > 5:
    args.devices = []
    for i in range(len(args.device) - 5):
        args.devices.append(int(args.device[i+5]))
    args.device = args.device[:6]
else:
    args.devices = [args.device]

if args.seed == -1:
    args.seed = random.randint(0, 1000000000)

try:
    n_shots = int(args.n_shots)
    args.n_shots = [n_shots]
except:
    args.n_shots = eval(args.n_shots)

try:
    milestone = int(args.milestones)
    args.milestones = list(np.arange(milestone, args.epochs + args.manifold_mixup, milestone))
except:
    args.milestones = eval(args.milestones)
if args.milestones == [] and args.cosine:
    args.milestones = [args.epochs + args.manifold_mixup]

if args.gamma == -1:
    if args.cosine:
        args.gamma = 1.
    else:
        args.gamma = 0.1

if args.mm:
    args.mixup = True
    
print("args, ", end='')
