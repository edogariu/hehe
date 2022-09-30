# hehe
This is code for **da pengster** (bio ML stuffs)

The framework for this project is that we are given two datasets:
1. coupled single-cell measurements of the form `(DNA, RNA)`, where `DNA` is short for a ~220k dimensional vector of ATAC-seq chromatin accessibility measurements for 220k genes and `RNA` is a ~24k vector of measured gene expression
2. coupled single-cell measurements of the form `(RNA, protein)`, where `RNA` is defined as before and `protein` is a 140 dimensional vector of surface level protein measurements for 140 different proteins

The idea is as follows (*please nobody write a paper about it before i do, if you do and dont cite me i will litigate immediately*): create a joint embedding space that encodes `DNA`, `RNA`, and `protein` together using very cool update steps and contrastive losses and such. The fact that we have two different `RNA` distributions (one as targets and one as inputs) from the two datasets makes it useful to unify the datasets with an approach like this. Also, this framework is very general, can be expanded and built on without starting from scratch, and somehow feels like the future of such a data-driven field like bio ML.

## Table of Contents
- `jepa.py` contains a Joint Embedding (Predictive) Architecture framework that wraps several individual models (joint embeddings work like [CLIP](https://openai.com/blog/clip/) or [this](https://openaccess.thecvf.com/content/WACV2021/papers/VidalMata_Joint_Visual-Temporal_Embedding_for_Unsupervised_Learning_of_Actions_in_Untrimmed_WACV_2021_paper.pdf), but what I am doing has more moving parts) 
- `models.py` contains the actual models implemented (currently [Enformer](https://www.nature.com/articles/s41592-021-01252-x) and a better version of [dilated nets](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5))
- `trainer.py` contains a big boy training framework, complete with lr scheduling, patience algorithms, etc.
- `datasets.py` implements custom PyTorch Dataset objects to allow for efficient dataloading of the massive `.h5` files we use
- `env_init.sh` sets up the python virtual environment to make things nice

The rest of the code is exploratory and not finalized (well, nothing is finalized but you know what i mean)
