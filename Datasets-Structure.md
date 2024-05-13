# Datasets-Structure

**(YOLOv8 format)**

The Yolov8 dataset for segmentation is usually structured as follows:
```
yolo_dataset/
│
├── train/
│ ├── images/
│ │ └── 🖼️ img_n # Example image 
│ │
│ └── labels/
│   └── 📄 img_n_labels.txt # Example labels file 
│
├── valid/
│ │ ... (similar)
│
└── 📄 data.yaml
```

Each ```img_x_labels.txt``` file contains multiple annotations (one per line) with corresponding class ID and segmentation coordinates:

`<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>`

The file `data.yaml` contains keys such as:
 - names (the class names)
 - nc (number or classes)
 - train (path/to/train/images/)
 - val (path/to/val/images/)


**(COCO Instance Segmentation format)**
```
coco_dataset/
│
├── train/
│  ├── 🖼️ img_n # Example image 
│  └── 📄 annotations.json # The annotations json file
│
└── valid/ 
   └── ... (similar)

```

The annotations json file contains a dictionary of lists:

- images (a list of dictionaries)
  - id - image ID
  - file_name 
  - height
  - width


- annotations (a list of dictionaries)
  - id
  - image_id
  - category_id
  - bbox 
  - area
  - segmentation (a segmentation polygon)
  - iscrowd

 
- categories (a list of dictionaries)
  - id
  - name