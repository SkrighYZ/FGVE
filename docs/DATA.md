# Data Formats
Here are some explanations for the data files needed by our model.

## Text Files
The following files contains the visual entailment text data splits from [e-ViL](https://openaccess.thecvf.com/content/ICCV2021/html/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.html). We augment the dataset with raw AMR sequences for each hypothesis, each image's detected object tags (we only take object labels and not their attributes), and each image's object detection confidence scores. We also add the preprocessed AMR sequences to the files.

- `ve_train.json`
- `ve_test.json`
- `ve_dev.json`

Each file is a list of dictionaries of format:

```
{
'pair_id': [Example identifier in VE],
'prem': [Premise], 
'hyp': [Text hypothesis], 
'img_id': [Flickr30K image ID], 
'conf': [Prediction confidence],
'ans': [Sample-level label], 
'hyp_amr': [Raw hypothesis AMR], 
'hyp_amr_cleaned': [Preprocessed hypothesis AMR]
}
```

## AMR Files
- `amr_annotations.json`: Fine-grained KE annotations for our FGVE test set. Labels: entailment (0), neutral (1), contradiction (2), opt-out (3).
- `amr_substitute.json`: AMR role string substitute used for preprocessing to prevent BertToknizer breaking up AMR role tokens.
- `amr_vocab.txt`: New AMR tokens to be added to the tokenizer.
- `amr_special_tokens.txt`: `[amr-unknown]` tokens in AMRs are treated as `[UNK]` for BertToknizer.

## Token Index Files
- `node_edge_indices.pkl`: A dictionary provided for convenient token mapping when calculating our losses. Each example in the dataset has `pair_id` as key and the value is a dictionary of the following form:

```
{
'tokens': [Tokenized string, convenient for retrieving original KE string given token indices], 
'node_indices': [Token indices for each node], 
'edge_indices': [Token indices for each edge],
'edges': [Indices of nodes and edges that form tuples]
}
```

- `tag2region.pkl`: Token index mapping from each object tag back to each object region. Each example in the dataset has `pair_id` as key.

See [here](https://github.com/SkrighYZ/FGVE/blob/65ef32b16b00dfb1ac89d88064a938f992625ca7/preprocess_utils.py#L142) and [here](https://github.com/SkrighYZ/FGVE/blob/65ef32b16b00dfb1ac89d88064a938f992625ca7/preprocess_utils.py#L172) for more details.

## Image Features
We use one file for each image. Each compressed `.npz` file contains a feature matrix in field `x`, height in field `image_h`, width in field `image_w`, and prediction confidences in field `obj_conf`. You can access the features by  `np.load('xxx.npz')['x']`.

The feature matrix has a size of `(N, 2054)` where `N` is the number of objects whose features are extracted in this image. Among the 2054 feature dimensions, the first 2048 are the CNN-extracted region features; four dimensions are the bounding box coordinates (left, top, right, bottom) normalized to `[0, 1]` by image size; and the rest two dimensions are the normalized width and height of the object.





# Custom Data

## AMR Generation
We generate the AMRs from text hypotheses using [SPRING](https://ojs.aaai.org/index.php/AAAI/article/view/17489), with the help of [bjascob/amrlib](https://github.com/bjascob/amrlib) API.

## Image Feature Extraction
We extract the Flickr30K image features using a pretrained Faster R-CNN ResNeXt152-C4 detector. The image features are extracted following instructions in [pzzhang/VinVL](https://github.com/pzzhang/VinVL) and [microsoft/scene\_graph\_benchmark](https://github.com/microsoft/scene_graph_benchmark) and we reformat the resulting files to `.npy` format.

## Other Files
We provide some utility functions in [preprocess_utils.py](https://github.com/SkrighYZ/FGVE/blob/main/preprocess_utils.py) to generate the other files needed by our model. Note that the functions are specific to the BertTokenizer used by our model.