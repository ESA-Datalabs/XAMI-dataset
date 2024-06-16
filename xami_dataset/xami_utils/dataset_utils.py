import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Any
import pycocotools.mask as maskUtils
import json
from astropy.io import fits
import pywt
import os

from shutil import copy
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import re
import datasets
from datasets import Value
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from re import split
import zipfile
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from tabulate import tabulate

BOX_COLOR = (255, 0, 0) 
TEXT_COLOR = (0, 0, 255) 

def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def visualize_bbox(img, bbox, color, thickness=1, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def visualize_titles(img, bbox, title, font_thickness = 1, font_scale=0.35, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([65/255, 174/255, 255/255, 0.4]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks(masks, ax, random_color=False, colours=None):
    for i in range(len(masks)):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        elif colours is not None:
            color = np.array([c/255.0 for c in colours[i]]+[0.6])
            # color = np.array(list(colours[i])+[0.6])
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = masks[i].shape[-2:]
        mask_image = masks[i].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1))    

def plot_bboxes_and_masks(image, bboxes, masks):
    
    image_copy = image.copy()
    image_copy = image_copy.astype('uint8')

    for i in range(len(bboxes)): 
        start_point = (int(bboxes[i][0]), int(bboxes[i][1]))  
        end_point = (int(bboxes[i][2]+bboxes[i][0]), int(bboxes[i][3]+bboxes[i][1]))  
        color = random_color()
        cv2.rectangle(image_copy, start_point, end_point, color, 2)

        colored_mask = np.zeros_like(image_copy)
        colored_mask[masks[i] >0] = color

        image_copy = cv2.addWeighted(image_copy, 1, colored_mask, .2, 0.9)

        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        plt.close()
        
def visualize_masks(image_path, image, masks, labels, colors, alpha=0.4, is_gt=True):
    """
    Visualize masks on an image with contours and dynamically sized text labels with a background box.

    Parameters:
    - image_path: The path to the image file.
    - image: The original image (numpy array).
    - masks: A list of masks (numpy arrays), one for each label.
    - labels: A list of labels corresponding to each mask.
    - colors: A list of colors corresponding to each label.
    - alpha: Transparency of masks.
    """
    # Pad the image (when labels are too big and they appear at image corners)
    pad_size = 20  # Padding size
    if len(image.shape) == 3:
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                          cv2.BORDER_CONSTANT, value=[255,255,255])
    else:  # For grayscale images
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                          cv2.BORDER_CONSTANT, value=255)

    # Work on a copy of the padded image
    temp_image = padded_image.copy()

    # Ensure the temp_image is in RGB format
    if len(temp_image.shape) == 2 or temp_image.shape[2] == 1:
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2RGB)
    elif temp_image.shape[2] == 4:
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2RGB)

    # Create a color overlay
    overlay = temp_image.copy()
    for mask, label in zip(masks, labels):
        # Adjust mask for padding
        padded_mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size,
                                         cv2.BORDER_CONSTANT, value=0)

        color = colors[label]
        bg_color = (int(color[0]), int(color[1]), int(color[2]))  # Background color

        overlay[padded_mask == 1] = [color[0], color[1], color[2]]

        contours, _ = cv2.findContours(padded_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_color = [c // 4 for c in color]  # Darker shade of the mask color
        cv2.drawContours(temp_image, contours, -3, contour_color, 2) 

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            draw_label(temp_image, label, (x, y), bg_color)

    cv2.addWeighted(overlay, alpha, temp_image, 1 - alpha, 0, temp_image)

    # Visualization code remains the same
    plt.figure(figsize=(50, 50)) 
    plt.imshow(temp_image)
    if is_gt:
        plt.title(f'Ground truth classes \n{image_path.split(".")[0]}', fontsize=100)
        output_file= './plots/'+image_path.split(".")[0]+'_gt.png'
    else:
        plt.title(f'Predicted classes \n{image_path.split(".")[0]}', fontsize=100)
        output_file= './plots/'+image_path.split(".")[0]+'_pred.png'

    plt.axis('off')
    plt.show()
    plt.imsave(output_file, temp_image, dpi=1000)
    plt.close()
    
def isolate_background(image, decomposition='db1', level=2, sigma=1):
    """
    Isolates and visualizes parts of an image that are close to the estimated background.
    
    Parameters:
    - image: numpy.ndarray. The input image for background isolation.
    - decomposition: str, optional. The name of the wavelet to use for decomposition. Default is 'db1'.
    - level: int, optional. The level of wavelet decomposition to perform. Default is 2.
    - sigma: float, optional. The number of standard deviations from the background mean to consider as background. Default is 1.
    
    Returns:
    - mask:  numpy.ndarray. A binary mask indicating parts of the image close to the background.
    - close_to_background: numpy.ndarray. The parts of the original image that are close to the background.
    """
    
    # Perform a multi-level wavelet decomposition
    coeffs = pywt.wavedec2(image, decomposition, level=level)
    if len(coeffs) == 3:
        cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
        # Reconstruct the background image from the approximation coefficients
        background = pywt.waverec2((cA2, (None, None, None), (None, None, None)), decomposition)
    
    elif len(coeffs) == 2:
        cA1, (cH1, cV1, cD1) = coeffs
        background = pywt.waverec2((cA1, (None, None, None), (None, None, None)), decomposition)
        

    # Calculate the mean and standard deviation of the background
    mean_bg = np.mean(background)
    std_bg = np.std(background)
    
    # Define a threshold based on mean and standard deviation
    lower_bound = mean_bg - std_bg * sigma
    upper_bound = mean_bg + std_bg * sigma
    
    # Create a mask where pixel intensities are close to the background
    mask = (image >= lower_bound) & (image <= upper_bound)
    
    # Apply the mask to the image
    close_to_background = image * mask
    
    # # Visualize the results
    # plt.imshow(mask, cmap='gray')
    # plt.title("Binary Mask")
    # plt.show()
    # plt.close()
    
    # plt.imshow(close_to_background, cmap='gray')
    # plt.title("Image Close to Background")
    # plt.show()
    # plt.close()
    
    return mask, close_to_background

def get_coords_and_masks_from_json(input_dir, data_in, image_key=None):
    """
    Extracts masks and bounding box coordinates from a JSON object containing image annotations.
    
    Parameters:
    - data_in (dict): The JSON dataset split containing image and annotation details.
    - image_key (str, optional): A string key that identifies the specific image for which annotations are required. 
    If provided, the function will only process the image with the matching key.
    
    Returns:
    - result_masks (dict): A dictionary of type {mask: mask array}.
    - bbox_coords (dict): A dictionary of type {mask:bounding box coordinates corresponding to that mask}.
    
    For each annotation, it keeps it only if its bounding box size is significant (h,w) > (5,5).
    """
    result_masks, bbox_coords, result_class = {}, {}, {}
    class_categories = {data_in['categories'][a]['id']:data_in['categories'][a]['name'] for a in range(len(data_in['categories']))}

    for im in data_in['images']:  

        masks = [data_in['annotations'][a] for a in range(len(data_in['annotations'])) 
                 if data_in['annotations'][a]['image_id'] == im['id']]
        temp_img = cv2.imread(input_dir+im["file_name"])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        classes = [data_in['annotations'][a]['category_id'] for a in range(len(data_in['annotations'])) 
                   if data_in['annotations'][a]['image_id'] == im['id']]
        
        for i in range(len(masks)):
            segmentation = masks[i]['segmentation']
            if isinstance(segmentation, list):
                if len(segmentation) > 0 and isinstance(segmentation[0], list):
                    points = segmentation[0]
                    h_img, w_img = temp_img.shape[:2]
                    mask = create_mask(points, (h_img, w_img)) # COCO segmentations are polygon points, and must be converted to masks
                    bbox = mask_to_bbox(mask)  #xyhw[0], xyhw[1], xyhw[2]+xyhw[0], xyhw[3]+xyhw[1] 
                    # binary mask to RLE 
                    result_masks[f'{im["file_name"]}_mask{i}'] = maskUtils.encode(np.asfortranarray(mask)) #mask
                    bbox_coords[f'{im["file_name"]}_mask{i}'] = bbox
                    result_class[f'{im["file_name"]}_mask{i}'] = classes[i]

            elif isinstance(segmentation, dict): # TODO: handle this
                # Handle RLE segmentation
                if 'counts' in segmentation and 'size' in segmentation:
                    rle = maskUtils.frPyObjects([segmentation], segmentation['size'][0], segmentation['size'][1])
                    mask = maskUtils.decode(rle)
                    # result_masks[f'{im["file_name"]}_mask{i}'] = mask
                    # bbox_coords[f'{im["file_name"]}_mask{i}'] = [xyhw[0], xyhw[1], xyhw[2]+ xyhw[0], xyhw[3]+xyhw[1]]
                    # result_class[f'{im["file_name"]}_mask{i}'] = classes[i]
                    # Now `mask` is a binary mask of shape `(height, width)` where `segmentation['size']` = [height, width]
                    
        del temp_img
	
    return result_masks, bbox_coords, result_class, class_categories

def mask_to_bbox(mask):
    """
    Calculate the bounding box from the mask.
    mask: binary mask of shape (height, width) with non-zero values indicating the object
    Returns: bbox in the format [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max, y_max]

def create_mask(points, image_size):
    polygon = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    mask = np.zeros(image_size, dtype=np.uint8)
    
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

def create_mask_0_1(points, image_size):
    """
    Create a binary mask from polygon points.

    :param points: List of normalized points (x, y) of the polygon, values between 0 and 1.
    :param image_size: Tuple of (height, width) of the image.
    :return: A binary mask as a numpy array.
    """
    height, width = image_size
    
    # Scale points from normalized coordinates to pixel coordinates
    polygon = [(int(x * width), int(y * height)) for x, y in zip(points[::2], points[1::2])]
    
    mask = np.zeros(image_size, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask

def split_list(input_list, percentages):
    size = len(input_list)
    idx = 0
    output = []
    for percentage in percentages:
        chunk_size = round(percentage * size)
        chunk = input_list[idx : idx + chunk_size]
        output.append(chunk)
        idx += chunk_size
    return output

def create_dataset(image_paths, ground_truth_masks, bbox_coords):
        
    d_gt_masks, d_bboxes = {}, {}
    for img_path in image_paths:
        id = img_path.split('/')[-1]

        d_gt_masks.update({mask_id:mask 
				   for mask_id, mask in ground_truth_masks.items() if mask_id.startswith(id)})
        d_bboxes.update({bbox_id:bbox for bbox_id, bbox in bbox_coords.items() if bbox_id.startswith(id)}) 

    return d_gt_masks, d_bboxes

def load_json(path):
    with open(path) as f:
        return json.load(f)

def merge_coco_jsons(first_json, second_json, output_path):
        
    with open(first_json) as f:
        coco1 = json.load(f)

    with open(second_json) as f:
        coco2 = json.load(f)

    max_image_id = max(image['id'] for image in coco1['images'])
    max_annotation_id = max(annotation['id'] for annotation in coco1['annotations'])
    max_category_id = max(category['id'] for category in coco1['categories'])

    image_id_offset = max_image_id + 1
    annotation_id_offset = max_annotation_id + 1

    for image in coco2['images']:
        image['id'] += image_id_offset

    for annotation in coco2['annotations']:
        annotation['id'] += annotation_id_offset
        annotation['image_id'] += image_id_offset  # Update the image_id reference

    merged_coco = {
        'images': coco1['images'] + coco2['images'],
        'annotations': coco1['annotations'] + coco2['annotations'],
        'categories': coco1['categories']  # If categories are the same;
    }

    with open(output_path, 'w') as f:
        json.dump(merged_coco, f)

def get_category_table(train_annotations_path, valid_annotations_path):
		table_data = []

		for annotations_file in [train_annotations_path, valid_annotations_path]:
			data = get_data_from_json(annotations_file)
			_, cat_names, cat_counts = get_categories(data)
			total_count = sum(cat_counts)
			split = annotations_file.split('/')[-2]
			for cat_name, count in zip(cat_names, cat_counts):
				percentage = (count / total_count) * 100 if total_count > 0 else 0
				if percentage > 0:
					table_data.append({
						'Split': '#'+split.capitalize()+' (%)',
						'Category': cat_name,
						'Count': count,
						'Percentage': np.round(percentage, 3)
					})

		df = pd.DataFrame(table_data)
		pivot_count = df.pivot_table(index='Category', columns='Split', values='Count', fill_value=0)
		pivot_percentage = df.pivot_table(index='Category', columns='Split', values='Percentage', fill_value=0.0)
		combined_df = pivot_count.astype(str) + " (" + pivot_percentage.astype(str) + "%)"

		# Apply styles to manage column widths and alignment
		styled_df = combined_df.style.apply(highlight_max_str, subset=['#Train (%)', '#Valid (%)']
                                      ).set_properties(**{
			'text-align': 'center', 
			'white-space': 'nowrap', 
			'font-size': '10pt'
		}).set_table_styles([
			{'selector': 'th', 'props': [('font-size', '12pt'), ('text-align', 'center')]},
			{'selector': 'td', 'props': [('text-align', 'center'), ('padding', '5px'), ('width', '100px')]}
		]).set_caption("Annotation Counts and Percentages by Category and Split")

		return styled_df

def get_data_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_categories(split_set):
    cat_ids = [cat['id'] for cat in split_set['categories']]
    cat_names = [cat['name'] for cat in split_set['categories']]
    cat_counts = [len([ann for ann in split_set['annotations'] \
                    if ann['category_id'] == cat_id]) for cat_id in cat_ids]
    return cat_ids, cat_names, cat_counts

def get_filters_table(train_annotations_path, valid_annotations_path, filter_count):
    filter_annots_count = filter_count.copy()
    
    for annotations_file in [train_annotations_path, valid_annotations_path]:
        data_in = get_data_from_json(annotations_file)
        for img in data_in['images']:
            filter = img['file_name'][:13][-1]
            filter_count[filter] = filter_count.get(filter, 0) + 1
            image_annots = [annot for annot in data_in['annotations'] if annot['image_id'] == img['id']]
            for _ in image_annots:
                filter_annots_count[filter] = filter_annots_count.get(filter, 0) + 1
    
    df_counts = pd.DataFrame(list(filter_count.items()), columns=['Observing Filter', 'Image Count'])
    df_annot_counts = pd.DataFrame(list(filter_annots_count.items()), columns=['Observing Filter', 'Annotation Count'])
    df_merged = pd.merge(df_counts, df_annot_counts, on='Observing Filter')
    filters_df = df_merged.style.apply(highlight_max, subset=['Image Count', 'Annotation Count']
                                    ).set_properties(**{
        'text-align': 'center', 
        'white-space': 'nowrap', 
        'font-size': '10pt'
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-size', '12pt'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '5px'), ('width', '100px')]}
    ]).set_caption("Counts of Images and Annotations per Filter")
    
    return filters_df

def generate_heatmap(train_annotations_path, valid_annotations_path, output_path=None):
    
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import style
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    train_data = get_data_from_json(train_annotations_path)
    valid_data = get_data_from_json(valid_annotations_path)

    _, cat_names, _ = get_categories(train_data)
    categories = [cat_names[-1], cat_names[3], cat_names[2], cat_names[0], cat_names[1]]
    img_width, img_height = 512, 512

    style.use('ggplot')
    fig, axes = plt.subplots(2, len(categories), figsize=(3 * len(categories), 6), squeeze=False)
    splits = ['train', 'valid']

    for i, cat_name in enumerate(categories):
        for j, data in enumerate([train_data, valid_data]):
            cat_id = [cat['id'] for cat in data['categories'] if cat['name'] == cat_name]
            if not cat_id:
                continue
            annotations = [ann for ann in data['annotations'] if ann['category_id'] == cat_id[0]]
            heatmap = np.zeros((img_height, img_width))

            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = map(int, bbox)
                heatmap[y:y + h, x:x + w] += 1

            colors = ['white', 'skyblue', 'navy']
            cm = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
            ax = axes[j, i]
            im = ax.imshow(heatmap, cmap=cm, interpolation='nearest')
            ax.set_title(f'{splits[j].capitalize()} - {cat_name}', fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2, wspace=0.07)

    if output_path:
        plt.savefig(output_path)  
    plt.show()
    plt.close()

def generate_galactic_distribution_plot(dest_dir, dataset_name, obs_coords_file, splits=['train', 'valid'], output_path=None):
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.visualization import astropy_mpl_style

    with open(obs_coords_file, 'r') as file:
        coords_data = json.load(file)

    ra = []
    dec = []
    exposures = []

    for split in splits: 
        for image_file in os.listdir(os.path.join(dest_dir, dataset_name, split)):
            obs = image_file[:13]+'.fits'
            if obs in coords_data:
                ra.append(coords_data[obs]['RA'])
                dec.append(coords_data[obs]['DEC'])
                exposures.append(coords_data[obs]['EXPOSURE'])

    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galactic = coords.galactic

    exposure_norm = (exposures - np.min(exposures)) / (np.max(exposures) - np.min(exposures))

    plt.style.use(astropy_mpl_style)
    plt.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="aitoff")
    sc = ax.scatter(galactic.l.wrap_at(180*u.deg).radian, galactic.b.wrap_at(180*u.deg).radian,
                    c=exposure_norm, cmap='magma_r', alpha=0.7, edgecolor='none', zorder=1)

    ax.grid(True, color='silver', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14, colors='black', zorder=2)
    ax.tick_params(axis='y', labelsize=14, colors='black', zorder=2)
    ax.set_facecolor('aliceblue')

    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label('Normalized Exposure', fontsize=25)
    cbar.ax.tick_params(labelsize=17) 
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_tick_params(color='black')  

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.3)
    plt.show()
    if output_path:
        plt.savefig(output_path, dpi=200)  
    plt.close()

def exposure_per_filter(dest_dir, dataset_name, filters_dict, obs_coords_file, splits=['train', 'valid']):
    with open(obs_coords_file, 'r') as file:
        coords_data = json.load(file)

    exposure_per_filter = {filter: 0 for filter in filters_dict.keys()}
    exposure_per_filter = {'train': exposure_per_filter.copy(), 'valid': exposure_per_filter.copy()}

    for split in splits:
        split_filter_count = {filter: 0.0 for filter in filters_dict.keys()}

        for image_file in os.listdir(os.path.join(dest_dir, dataset_name, split)):
            obs = image_file[:13]
            obs_name_in_json = obs+'.fits'
            if obs_name_in_json in coords_data:
                filter = obs[-1]
                exposure_per_filter[split][filter] += coords_data[obs_name_in_json]['EXPOSURE']
                split_filter_count[filter] += 1

        # mean exposure per filter
        for filter in exposure_per_filter[split].keys():
            if split_filter_count[filter] > 0:
                exposure_per_filter[split][filter] /= split_filter_count[filter]

    return exposure_per_filter

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

def correct_bboxes(coco_data):
    
    annotations = coco_data['annotations']
    for ann_i in range(len(annotations)):
        # If the segmentation is in polygon format (COCO polygon)
        if isinstance(annotations[ann_i]['segmentation'], list):

            points = np.array(annotations[ann_i]['segmentation']).reshape(-1, 2)
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
            
            orig_bbox = annotations[ann_i]['bbox']
            orig_x, orig_y, orig_w, orig_h = map(int, orig_bbox)
            annotations[ann_i]['bbox'] = [x, y, w, h]
            
            # if annotations[ann_i]['category_id'] in [3, 0] and (orig_w>40 or orig_h>40):
            #     # take all annotations of that image and plot themon top of it
            #     img_annots = [ann for ann in coco_data['annotations'] \
            #         if ann['image_id'] == annotations[ann_i]['image_id'] and ann['category_id'] in [3, 0]]
            #     img = [img for img in coco_data['images'] if img['id'] == annotations[ann_i]['image_id']][0]
            #     img_path = img['file_name']
            #     image = cv2.imread('../../XAMI-model/data/xami_dataset/train/'+img_path)
            #     for annot in img_annots:
            #         bbox = annot['bbox']
            #         # points = np.array(annot['segmentation']).reshape(-1, 2)
            #         # x_min, y_min = np.inf, np.inf
            #         # x_max, y_max = -np.inf, -np.inf
            #         # x_min = min(x_min, points[:, 0].min())
            #         # y_min = min(y_min, points[:, 1].min())
            #         # x_max = max(x_max, points[:, 0].max())
            #         # y_max = max(y_max, points[:, 1].max())
            
            #         # width = x_max - x_min
            #         # height = y_max - y_min
                
            #         # # The bbox in COCO format [x_min, y_min, width, height]
            #         # bbox = [x_min, y_min, width, height]
            #         x, y, w, h = map(int, bbox)
            #         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            #     plt.imshow(image)
            #     plt.axis('off')
            #     plt.show()
            #     plt.close()
                
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

def display_image_with_annotations(coco, cat_names, dir, image_id):
    img = coco.loadImgs(image_id)[0]
    image_path = os.path.join(dir, img['file_name'])
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

