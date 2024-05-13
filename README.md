# XAMI-dataset

*The HuggingFace repository for this dataset can be found [here](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)*. 

The XAMI dataset contains 1000 annotated images of observations from diverse sky regions of the XMM-Newton Optical Monitor (XMM-OM) image catalog. An additional 50 images with no annotations are included to help decrease the amount of False Positives or Negatives that may be caused by complex objects (e.g., large galaxies, clusters, nebulae).

### Artefacts

A particularity of our XAMI dataset compared to every-day images datasets are the locations where artefacts usually appear. 
<img src="https://huggingface.co/datasets/iulia-elisa/XAMI-dataset/resolve/main/plots/artefact_distributions.png" alt="Examples of an image with multiple artefacts." />

Here are some examples of common artefacts:

<img src="https://huggingface.co/datasets/iulia-elisa/XAMI-dataset/resolve/main/plots/artefacts_examples.png" alt="Examples of common artefacts in the OM observations." width="400"/>

# Annotation platforms

The dataset images have been annotated using the following project:

- [Zooniverse project](https://www.zooniverse.org/projects/ori-j/ai-for-artefacts-in-sky-images), where the resulted annotations are not externally visible. 
- [Roboflow project](https://universe.roboflow.com/iuliaelisa/xmm_om_artefacts_512/), which allows for more interactive and visual annotation projects. 

# The dataset format
The XAMI dataset is splited into train and validation categories and contains annotated artefacts in COCO format for Instance Segmentation. We use multilabel Stratified K-fold technique (**k=4**) to balance class distributions across training and validation splits. We choose to work with a single dataset splits version (out of 4), but also provide means to train all 4 versions. 

A more detailed structure of our dataset in COCO and YOLOformat can be found in [Dataset Structure](Datasets-Structure.md).

# Downloading the dataset

The dataset repository on can be found on [HuggingFace](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset) and [Github](https://github.com/IuliaElisa/XAMI-dataset).

### Downloading the dataset archive from HuggingFace:

```python
from huggingface_hub import hf_hub_download
import pandas as pd

dataset_name = 'dataset_archive' # the dataset name of Huggingface
images_dir = '.' # the output directory of the dataset images
annotations_path = os.path.join(images_dir, dataset_name, '_annotations.coco.json')

for filename in [dataset_name, utils_filename]:
  hf_hub_download(
    repo_id="iulia-elisa/XAMI-dataset", # the Huggingface repo ID
    repo_type='dataset', 
    filename=filename, 
    local_dir=images_dir
  );

# Unzip file
!unzip "dataset_archive.zip"

# Read the json annotations file
with open(annotations_path) as f:
    data_in = json.load(f)
```
or

```
- using a CLI command:
```bash
huggingface-cli download iulia-elisa/XAMI-dataset dataset_archive.zip --repo-type dataset --local-dir '/path/to/local/dataset/dir'
```

### Cloning the repository for more visualization tools

<!-- The dataset can be generated to match our baseline (this is helpful for recreating dataset and model results).  -->

Clone the repository locally:

```bash
# Github
git clone https://github.com/IuliaElisa/XAMI-dataset.git
cd XAMI-dataset
```
or 
```bash
# HuggingFace
git clone https://huggingface.co/datasets/iulia-elisa/XAMI-dataset.git
cd XAMI-dataset
```

# Dataset Split with SKF (Optional)

- The below method allows for dataset splitting, using the pre-generated splits in CSV files. This step is useful when training multiple dataset splits versions to gain mor generalised view on metrics. 
```python
import utils

# run multilabel SKF split with the standard k=4
csv_files = ['mskf_0.csv', 'mskf_1.csv', 'mskf_2.csv', 'mskf_3.csv'] 

for idx, csv_file in enumerate(csv_files):
    mskf = pd.read_csv(csv_file)
    utils.create_directories_and_copy_files(images_dir, data_in, mskf, idx)
```

## Licence 
...
