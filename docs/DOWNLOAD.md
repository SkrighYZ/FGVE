# Download
We store our files on google drive. The easiest way to download them is to use [gdown](https://pypi.org/project/gdown/).

Please see [DATA.md](DATA.md) for detailed data formats.

## Dataset Files
Extract the data files into `FGVE/data`.

```bash
cd FGVE/data
gdown 1oODKH_GKygdkfNxzZLzJOfGQMoAVoXcv
unzip data.zip
mv data/* .
rm -r data
rm data.zip
```

## Pre-Extracted Image Features
We extract the Flickr30K image features using a pretrained Faster R-CNN ResNeXt152-C4 detector. The image features are extracted following instructions in [pzzhang/VinVL](https://github.com/pzzhang/VinVL) and [microsoft/scene\_graph\_benchmark](https://github.com/microsoft/scene_graph_benchmark) and we reformat the resulting files to `.npy` format.

Note that the Flickr30K dataset includes images obtained from [Flickr](https://www.flickr.com/). Use of the images must abide by the [Flickr Terms of Use](http://www.flickr.com/help/terms/). 

You can download pre-extracted image features with the following script.

```bash
cd $FEATURE_DIR
gdown 1A0YLy6GoWZcCAq_mwZB9oWsKpPEr0cwP
unzip f30k_features.zip
rm f30k_features.zip
```

## Model Checkpoints
Regardless of whether you need to do the training, you need the checkpoint pretrained by Oscar+. In fact, only its vocab (tokenizer) files are needed if you only want to do evaluation. Please see [here](https://github.com/SkrighYZ/FGVE/blob/65ef32b16b00dfb1ac89d88064a938f992625ca7/oscar/run_ve_amr.py#L1052) for details.

We modified it to have an additional (3rd) token type embedding initialized with the text's token type embedding. You can download the checkpoint with the script below.

```bash
cd $CHECKPOINT_DIR
gdown 1RyqP_uv04SuKLAf5BM4_x9SxEnuA0aW2
unzip pretrained_ckpt.zip
rm pretrained_ckpt.zip
```

We also release our best model's checkpoint (denoted *Ours* or *Ours+CLS* in the paper).

```bash
cd $CHECKPOINT_DIR
gdown 1-CepLyuATWQKEZUPg6Jfd_uJ6Uv2PE5j
unzip final_ckpt.zip
rm final_ckpt.zip
```
