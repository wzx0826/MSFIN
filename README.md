# MSFIN：Lightweight Image Super-Resolution with Multi-Scale Feature Interaction Network
This repository is an official PyTorch implementation of the paper Lightweight Image Super-Resolution with Multi-Scale Feature Interaction Network.

## Prerequisites:
1. Python 3.6
2. PyTorch 0.4.0
3. numpy
4. skimage
5. imageio
6. matplotlib
7. tqdm

For more informaiton, please refer to <a href="https://github.com/thstkdgus35/EDSR-PyTorch">EDSR</a> and <a href="https://github.com/yulunzhang/RCAN">RCAN</a>.

## Document
Train/             : all train files

Test/              : all test files

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

```
  cd Train/
  
# MSFIN x4  LR: 48 * 48  HR: 192 * 192
python main.py --template MSFIN --save MSFIN --scale 4 --reset --save_results --patch_size 192 --ext sep_reset

```

## Test
### Quick start
1. Using pre-trained model for training, all test datasets must be pretreatment by  ''Test/Prepare_TestData_HR_LR.m" and all pre-trained model should be put into "Test/model/".

2. Cd to '/Test/code', run the following scripts.

```
#MSFIN x4
python main.py --data_test MyImage --scale 4 --model MSFIN --pre_train ../model/MSFIN/MSFIN_X4.pt --test_only --save_results --chop --save "MSFIN" --testpath ../LR/LRBI --testset Set5

#MSFIN+ x4
python main.py --data_test MyImage --scale 4 --model MSFIN --pre_train ../model/MSFIN/MSFIN-S_X4.pt --test_only --save_results --chop --self_ensemble --save "MSFIN_Plus" --testpath ../LR/LRBI --testset Set5

```

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{wang2021lightweight,
  title={Lightweight Image Super-Resolution with Multi-scale Feature Interaction Network},
  author={Wang, Zhengxue and Gao, Guangwei and Li, Juncheng and Yu, Yi and Lu, Huimin},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch).
