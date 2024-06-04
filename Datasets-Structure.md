# Datasets-Structure

**(YOLOv8 format)**

The Yolov8 dataset for segmentation is usually structured as follows:
```
yolo_dataset/
│
├── train/
│ ├── images/
│ │ ├── 🖼️ <train_img_1>.<ImageFormat> 
| | ├── 🖼️ <train_img_2>.<ImageFormat> 
| |  ...
│ └── labels/
│   ├── 📄 <train_img_1_labels>.txt 
│   ├── 🖼️ <train_img_2_labels>.txt
|   ...
├── valid/
│ ├── images/
│ │ ├── 🖼️ <valid_img_1>.<ImageFormat> 
| | ├── 🖼️ <valid_img_2>.<ImageFormat> 
| |  ...
│ └── labels/
│   ├── 📄 <valid_img_1_labels>.txt
│   ├── 🖼️ <valid_img_2_labels>.txt
│
└── 📄 data.yaml
```

Each ```xyz_img_xyz_labels.txt``` file contains multiple annotations (one per line) with corresponding class ID and segmentation coordinates for the corresponding image:

`<class-index1> <x11> <y11> <x12> <y12> ... <x1n> <y1n>`
`...`

The file `data.yaml` contains keys such as:
 - `names` (the class names)
 - `nc` (number or classes)
 - `train` (path/to/train/images/)
 - `val` (path/to/val/images/)


**(COCO Instance Segmentation format)**
```
coco_dataset/
│
├── train/
│  ├── 🖼️ <train_img_1>.<ImageFormat> 
|  ├── 🖼️ <train_img_2>.<ImageFormat> 
|  ...
│  └── 📄 <train_annotations_file>.json # The annotations json file
│
└── valid/ 
|  ├── 🖼️ <valid_img_1>.<ImageFormat> 
|  ├── 🖼️ <valid_img_2>.<ImageFormat> 
|  ...
│  └── 📄 <valid_annotations_file>.json # The annotations json file
```

Each annotations json file in splits contains a dictionary of lists:

- images - a list of dictionaries with keys:
  - `id` - image ID
  - `file_name` 
  - `height`
  - `width`


- annotations - a list of dictionaries with keys:
  - `id`
  - `image_id`
  - `category_id`
  - `bbox` 
  - `area`
  - `segmentation` (a segmentation polygon)
  - `iscrowd`

 
- categories - a dictionary with keys:
  - `id`
  - `name`