import os
import json
from shutil import copy
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
import re
import datasets
from datasets import Value
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_directories_and_copy_files(images_dir, coco_data, image_data, k):
    base_dir = os.path.join(images_dir, f'mskf_{k}')
    os.makedirs(base_dir, exist_ok=True)

    for split in np.unique(image_data['SPLIT']):
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Filter the annotations
        split_ids = image_data[image_data['SPLIT'] == split]['IMADE_ID'].tolist()
        annotations = {
            'images': [img for img in coco_data['images'] if img['id'] in split_ids],
            'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in split_ids],
            'categories': coco_data['categories']
        }
            
        # Write the filtered annotations to a file
        with open(os.path.join(split_dir, '_annotations.coco.json'), 'w') as f:
            json.dump(annotations, f, indent=4)
        
        # Copy the images
        split_data = image_data[image_data['SPLIT'] == split]
        for _, row in split_data.iterrows():
            source = row['IMAGE_PATH']
            destination = os.path.join(split_dir, os.path.basename(source))
            copy(source, destination)
            
    print(f'Dataset split for mskf_{k} was successful.')

def split_to_df(dataset_dir, split):
    annotations_path = Path(dataset_dir+split+'/_annotations.coco.json')
    
    with annotations_path.open('r') as f:
        coco_data = json.load(f)
    
    def image_from_path(file_path):
        image = cv2.imread(file_path)
        return image
    
    def gen_segmentation(segmentation, width, height):
        mask_img = np.zeros((height, width, 3), dtype=np.uint8)
        for segment in segmentation:
            pts = np.array(segment, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_img, [pts], (255, 255, 255))  # Fill color in BGR
        
        return mask_img
        
    images_df = pd.DataFrame(coco_data['images'][5:25], columns=['id', 'file_name', 'width', 'height'])
    annotations_df = pd.DataFrame(coco_data['annotations'])
    df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')
    image_folder = annotations_path.parent 
    df['file_path'] = df['file_name'].apply(lambda x: str(image_folder / x))
    df['observation'] = df['file_name'].apply(lambda x: x.split('.')[0].replace('_png', ''))
    df['image'] = df['file_path'].apply(image_from_path)
    df['segmentation'] = df.apply(lambda row: gen_segmentation(row['segmentation'], row['width'], row['height']), axis=1)
    df = df.drop('file_path', axis=1)
    df = df.drop('file_name', axis=1)
    df['annot_id'] = df['id_x'] 
    df = df.drop('id_x', axis=1)
    df = df.drop('id_y', axis=1)
    
    # take image fro df, and the corresponging annotations and plot them on image
    # for i in range(5):
    #     img = df['image'][i]
    #     annot_id = df['annot_id'][i]
    #     # plot the image with the annotation using plt
    #     if img.dtype != np.uint8:
    #         img = img.astype(np.uint8)
    #     # plot
    #     segm_polygon = df['segmentation'][i]
    #     plt.imshow(segm_polygon)
    #     plt.axis('off')
    #     plt.show()
    #     plt.close()
    
    return df, coco_data
	
def df_to_dataset_dict(df, coco_data, cats_to_colours):

    def annot_on_image(annot_id, img_array, cat_id, annot_type='segm'):
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
    
        pil_image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(pil_image)
        if annot_type=='bbox':
            bbox = [annot for annot in coco_data['annotations'] if annot['id'] == annot_id][0]['bbox']
            x_min, y_min, width, height = bbox
            top_left = (x_min, y_min)
            bottom_right = (x_min + width, y_min + height)
            
            draw.rectangle([top_left, bottom_right], outline=cats_to_colours[cat_id][1], width=2)
        else:
            # look for the annotation in coco_data that corresponds to the annot_id
            segm_polygon = [annot for annot in coco_data['annotations'] if annot['id'] == annot_id][0]['segmentation'][0]
            polygon = [(segm_polygon[i], segm_polygon[i+1]) for i in range(0, len(segm_polygon), 2)]
            draw.polygon(polygon, outline=cats_to_colours[cat_id][1], width=2)  
         
        # plt.imshow(pil_image)
        # plt.axis('off')
        # plt.show()
        # plt.close()
           
        byte_io = BytesIO()
        pil_image.save(byte_io, 'PNG')
        byte_io.seek(0)
        png_image = Image.open(byte_io)
    
        return png_image
    
    dictionary = df.to_dict(orient='list')
    feats=datasets.Features({"observation id":Value(dtype='string'), \
                             'segmentation': datasets.Image(), \
                             'bbox':datasets.Image() , \
                             'label': Value(dtype='string'),\
                             'area':Value(dtype='string'), 
                             'image shape':Value(dtype='string')})
    
    dataset_data = {"observation id":dictionary['observation'],  \
                  'segmentation': [annot_on_image(dictionary['annot_id'][i], dictionary['image'][i], dictionary['category_id'][i]) \
                      for i in range(len(dictionary['segmentation']))], \
                  'bbox': [annot_on_image(dictionary['annot_id'][i], dictionary['image'][i], dictionary['category_id'][i], annot_type='bbox') \
                      for i in range(len(dictionary['bbox']))], \
                  'label': [cats_to_colours[cat][0] for cat in dictionary['category_id']],\
                  'area':['%.3f'%(value) for value in dictionary['area']], \
                  'image shape':[f"({dictionary['width'][i]}, {dictionary['height'][i]})" for i in range(len(dictionary['width']))]}
    the_dataset=datasets.Dataset.from_dict(dataset_data,features=feats)

    return the_dataset

def merge_coco_jsons(first_json, second_json, output_path):
        
    # Load the first JSON file
    with open(first_json) as f:
        coco1 = json.load(f)

    # Load the second JSON file
    with open(second_json) as f:
        coco2 = json.load(f)

    # Update IDs in coco2 to ensure they are unique and do not overlap with coco1
    max_image_id = max(image['id'] for image in coco1['images'])
    max_annotation_id = max(annotation['id'] for annotation in coco1['annotations'])
    max_category_id = max(category['id'] for category in coco1['categories'])

    # Add an offset to the second coco IDs
    image_id_offset = max_image_id + 1
    annotation_id_offset = max_annotation_id + 1
    # category_id_offset = max_category_id + 1

    # Apply offset to images, annotations, and categories in the second JSON
    for image in coco2['images']:
        image['id'] += image_id_offset

    for annotation in coco2['annotations']:
        annotation['id'] += annotation_id_offset
        annotation['image_id'] += image_id_offset  # Update the image_id reference

    # Merge the two datasets
    merged_coco = {
        'images': coco1['images'] + coco2['images'],
        'annotations': coco1['annotations'] + coco2['annotations'],
        'categories': coco1['categories']  # If categories are the same; otherwise, merge as needed
    }

    # Save the merged annotations to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(merged_coco, f)

def percentages(n_splits, image_ids, labels):
    labels_percentages = {}
    for i in range(n_splits):
        train_k, valid_k = 0, 0
        train_labels_counts = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
        valid_labels_counts = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
        for j in range(len(image_ids[i]['train'])):
            for cat in list(labels[i]['train'][j]):
                train_labels_counts[cat] += 1
                train_k+=1
        
        for j in range(len(image_ids[i]['valid'])):
            for cat in list(labels[i]['valid'][j]):
                valid_labels_counts[cat] += 1
                valid_k+=1
    
        train_labels_counts = {cat:counts * 1.0/train_k for cat, counts in train_labels_counts.items()}
        valid_labels_counts = {cat:counts * 1.0/valid_k for cat, counts in valid_labels_counts.items()}
                
        labels_percentages[i] = {'train':train_labels_counts, 'valid':  valid_labels_counts}
        
    return labels_percentages

def make_split(data_in, train_index, valid_index):
    
    data_in_train = data_in.copy()
    data_in_valid = data_in.copy()
    
    data_in_train['images'] = [data_in['images'][train_index[i][0]] for i in range(len(train_index))]
    data_in_valid['images'] = [data_in['images'][valid_index[i][0]] for i in range(len(valid_index))]
    train_annot_ids, valid_annot_ids = [], []
    
    for img_i in data_in_train['images']:
        annotation_ids = [annot['id'] for annot in data_in_train['annotations'] if annot['image_id'] == img_i['id']]
        train_annot_ids +=annotation_ids
        
    for img_i in data_in_valid['images']:
        annotation_ids = [annot['id'] for annot in data_in_valid['annotations'] if annot['image_id'] == img_i['id']]
        valid_annot_ids +=annotation_ids
        
    data_in_train['annotations'] = [data_in_train['annotations'][id] for id in train_annot_ids]
    data_in_valid['annotations'] = [data_in_valid['annotations'][id] for id in valid_annot_ids]
    
    print(len(data_in_train['images']), len(data_in_valid['images']))
    return data_in_train, data_in_valid

def correct_bboxes(annotations):
        for ann in annotations:
            # If the segmentation is in polygon format (COCO polygon)
            if isinstance(ann['segmentation'], list):
    
                points = np.array(ann['segmentation']).reshape(-1, 2)
                x_min, y_min = np.inf, np.inf
                x_max, y_max = -np.inf, -np.inf
                x_min = min(x_min, points[:, 0].min())
                y_min = min(y_min, points[:, 1].min())
                x_max = max(x_max, points[:, 0].max())
                y_max = max(y_max, points[:, 1].max())
        
                width = x_max - x_min
                height = y_max - y_min
            
                # The bbox in COCO format [x_min, y_min, width, height]
                bbox = [x_min, y_min, width, height]
                x, y, w, h = map(int, bbox)
                ann['bbox'] = [x, y, w, h]
    
        return annotations
    
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_max_str(s):
    
    cats = []
    for cat in s:
        cats.append([float(match) for match in re.findall(r"[-+]?[0-9]*\.?[0-9]+", cat)][0])
        
    is_max = cats == np.max(cats)
    return ['background-color: yellow' if v else '' for v in is_max]

def read_yolo_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = list(map(float, parts[1:]))
        annotations.append((class_id, points))
    
    return annotations

def display_image_with_annotations(coco, cat_names, image_id):
    img = coco.loadImgs(image_id)[0]
    image_path = os.path.join('./mskf_0/train/', img['file_name'])
    I = Image.open(image_path)
    plt.imshow(I); plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    ax = plt.gca()
    
    for ann in anns:
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 5, cat_names[ann['category_id']],
                color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

def plot_yolo_segmentations(image_path, annotations, category_mapping):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)  # Load a font
    except IOError:
        font = ImageFont.load_default()

    for class_id, points in annotations:
        # Scale points from normalized coordinates to image dimensions
        scaled_points = [(p[0] * width, p[1] * height) for p in zip(points[0::2], points[1::2])]
        draw.polygon(scaled_points, outline='green', fill=None)  

        category_name = category_mapping[class_id][0]
        centroid_x = sum([p[0] for p in scaled_points]) / len(scaled_points)
        centroid_y = sum([p[1] for p in scaled_points]) / len(scaled_points)
        draw.text((centroid_x, centroid_y), category_name, fill='red', font=font, anchor='ms')

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def plot_coco_segmentations(image_path, annotations, category_mapping):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)  # Load a font
    except IOError:
        font = ImageFont.load_default()

    for class_id, points in annotations:
        # Scale points from normalized coordinates to image dimensions
        scaled_points = [(p[0] * width, p[1] * height) for p in zip(points[0::2], points[1::2])]
        draw.polygon(scaled_points, outline='green', fill=None)  

        category_name = category_mapping[class_id][0]
        centroid_x = sum([p[0] for p in scaled_points]) / len(scaled_points)
        centroid_y = sum([p[1] for p in scaled_points]) / len(scaled_points)
        draw.text((centroid_x, centroid_y), category_name, fill='red', font=font, anchor='ms')

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def find_image_id(coco_data, filename):
    for image in coco_data['images']:
        if image['file_name'] == filename:
            return image['id']
    return None

def extract_coco_annotations(coco_data, image_id):
    annotations = []
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            annotations.append({
                'category_id': annotation['category_id'],
                'segmentation': annotation['segmentation']
            })
    return annotations