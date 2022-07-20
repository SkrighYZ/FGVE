# Installation

## Requirements
- Python 3.7
- PyTorch 1.8.0
- Torchvision 0.9.0
- CUDA 11.1

## Setup with Conda
```bash
conda create --name fgve python=3.7
conda activate fgve

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone --recursive https://github.com/SkrighYZ/FGVE.git
cd FGVE
python setup.py build develop

pip install -r requirements.txt

unset INSTALL_DIR
```
