# Pyramid Stereo Matching Network

This repository contains the code (in PyTorch) for "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) by [Jia-Ren Chang](https://jiarenchang.github.io/) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).


### original repo
https://github.com/JiaRenChang/PSMNet


### difference 
this repo estimate Left and Right Disparity(original is left only). so you can train without Ground Truth. It is more practical to use


### Dependencies
tested environment

- [Python 3.69](https://www.python.org/downloads/)
- [PyTorch(1.5.0)](http://pytorch.org)
- torchvision 0.6.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [CUDA 10.2] [https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork]
- [Ubuntu 18.04]



## installation
sudo apt install python3-pip

pip3 install torch torchvision

pip3 install tensorboard



### Train
sh train.sh

### Inference
sh inference.sh

## visualize training log by tensorboard
tensorboard --logdir {your_workingdir}/logs

access from your browser
http://localhost:6006/