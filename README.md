## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/Songyeyaosong/U-TRSOR.git
cd U-TRSOR
pip install -v -e .
```

## Train

```shell
python tools/train.py configs/oriented_rcnn_teef_v2_edl_kl_xcor_corall_le0.15_lt0.25_r50_fpn_fp16_1x_dota_ms_rr_le90.py
```