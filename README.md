<div align="center">
<h1> XAMI: XMM-Newton optical Artefact Mapping for astronomical Instance segmentation </h1>
<h2> <i> The Dataset </i> </h12>
</div>

## 💫 Introduction

The **HuggingFace🤗** repository for this dataset can be found **[here](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)**. 

The XAMI dataset contains 1000 annotated images of observations from diverse sky regions of the XMM-Newton Optical Monitor (XMM-OM) image catalog. An additional 50 images with no annotations are included to help decrease the amount of False Positives or Negatives that may be caused by complex objects (e.g., large galaxies, clusters, nebulae).

# 📊 Downloading the dataset

### Cloning the repository

```bash
git clone https://github.com/ESA-Datalabs/XAMI-dataset.git
cd XAMI-dataset
conda env create -f environment.yaml # create an environment with the package requirements
```

Then

### Downloading the dataset from HuggingFace🤗

- using a python script (see [load_and_visualise_dataset.pynb](https://github.com/ESA-Datalabs/XAMI-dataset/blob/main/load_and_visualise_dataset.ipynb))

```python
from xami_dataset import XAMIDataset

# Download the dataset
xami_dataset = XAMIDataset(
    repo_id="iulia-elisa/XAMI-dataset", 
    dataset_name="xami_dataset", 
    dest_dir='./data')
```

- Or you can simply download only the dataset and unarchive it using a CLI command

```bash
DEST_DIR='/path/to/local/dest'

huggingface-cli download iulia-elisa/XAMI-dataset xami_dataset.zip --repo-type dataset --local-dir "$DEST_DIR" && unzip "$DEST_DIR/xami_dataset.zip" -d "$DEST_DIR" && rm "$DEST_DIR/xami_dataset.zip"
```

# 👀 About

The dataset is splited into train and validation categories and contains annotated artefacts in COCO format for Instance Segmentation. We use multilabel Stratified K-fold (**k=4**) to balance class distributions across splits. We choose to work with a single dataset splits version (out of 4) but also provide means to work with all 4 versions. 

Please check [Dataset Structure](Datasets-Structure.md) for a more detailed structure of our dataset in COCO-IS and YOLOv8-Seg format.

### Artefacts

A particularity of our XAMI dataset compared to every-day images datasets are the locations where artefacts usually appear. 
<img src="https://github.com/ESA-Datalabs/XAMI-dataset/blob/main/plots/artefact_distributions.png" alt="Examples of an image with multiple artefacts." />

Here are some examples of common artefacts in the dataset:

<img src="https://github.com/ESA-Datalabs/XAMI-dataset/blob/main/plots/artefacts_examples.png" alt="Examples of common artefacts in the OM observations." width="400"/>

### Annotation platforms

The images have been annotated using the following projects:

- [Zooniverse project](https://www.zooniverse.org/projects/ori-j/ai-for-artefacts-in-sky-images), where the resulted annotations are not externally visible. 
- [Roboflow project](https://universe.roboflow.com/iuliaelisa/xmm_om_artefacts_512/), which allows for more interactive and visual annotation projects. 

# © Licence 
**[CC BY-NC 3.0 IGO](https://creativecommons.org/licenses/by-nc/3.0/igo/deed.en).**