import os
import json
from re import split
import zipfile
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from xami_utils import utils

class XAMIDataset:
	def __init__(self, repo_id, dataset_name, dest_dir='.'):
		self.repo_id = repo_id
		self.dataset_name = dataset_name
		self.dest_dir = dest_dir
		self.train_annotations_path = os.path.join(dest_dir, dataset_name, 'train', '_annotations.coco.json')
		self.valid_annotations_path = os.path.join(dest_dir, dataset_name, 'valid', '_annotations.coco.json')
		self.download_dataset()
		print("Dataset downloaded.")
		self.unzip_dataset()
		print("Dataset unzipped.")
		print(f"Train annotations file: \033[95m{self.train_annotations_path}\033[0m")
		print(f"Valid annotations file: \033[95m{self.valid_annotations_path}\033[0m")
  
	def download_dataset(self):
		hf_hub_download(
			repo_id=self.repo_id,
			repo_type='dataset',
			filename=self.dataset_name + '.zip',
			local_dir=self.dest_dir
		)

	def unzip_dataset(self):
		zip_path = os.path.join(self.dest_dir, self.dataset_name + '.zip')
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(self.dest_dir)

		os.remove(zip_path)
  
	def get_data_from_json(self, path):
		with open(path, 'r') as f:
			data = json.load(f)
		return data

	def get_categories(self, split_set):
		cat_ids = [cat['id'] for cat in split_set['categories']]
		cat_names = [cat['name'] for cat in split_set['categories']]
		cat_counts = [len([ann for ann in split_set['annotations'] \
      					if ann['category_id'] == cat_id]) for cat_id in cat_ids]
		return cat_ids, cat_names, cat_counts

	def get_category_table(self):
		table_data = []

		for annotations_file in [self.train_annotations_path, self.valid_annotations_path]:
			data = self.get_data_from_json(annotations_file)
			_, cat_names, cat_counts = self.get_categories(data)
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
		styled_df = combined_df.style.apply(utils.highlight_max_str, subset=['#Train (%)', '#Valid (%)']
                                      ).set_properties(**{
			'text-align': 'center', 
			'white-space': 'nowrap', 
			'font-size': '10pt'
		}).set_table_styles([
			{'selector': 'th', 'props': [('font-size', '12pt'), ('text-align', 'center')]},
			{'selector': 'td', 'props': [('text-align', 'center'), ('padding', '5px'), ('width', '100px')]}
		]).set_caption("Annotation Counts and Percentages by Category and Split")

		return styled_df

	def get_filters_table(self, filter_count):
		filter_annots_count = filter_count.copy()
		
		for annotations_file in [self.train_annotations_path, self.valid_annotations_path]:
			data_in = self.get_data_from_json(annotations_file)
			for img in data_in['images']:
				filter = img['file_name'][:13][-1]
				filter_count[filter] = filter_count.get(filter, 0) + 1
				image_annots = [annot for annot in data_in['annotations'] if annot['image_id'] == img['id']]
				for _ in image_annots:
					filter_annots_count[filter] = filter_annots_count.get(filter, 0) + 1
		
		df_counts = pd.DataFrame(list(filter_count.items()), columns=['Observing Filter', 'Image Count'])
		df_annot_counts = pd.DataFrame(list(filter_annots_count.items()), columns=['Observing Filter', 'Annotation Count'])
		df_merged = pd.merge(df_counts, df_annot_counts, on='Observing Filter')
		filters_df = df_merged.style.apply(utils.highlight_max, subset=['Image Count', 'Annotation Count']
                                     ).set_properties(**{
			'text-align': 'center', 
			'white-space': 'nowrap', 
			'font-size': '10pt'
		}).set_table_styles([
			{'selector': 'th', 'props': [('font-size', '12pt'), ('text-align', 'center')]},
			{'selector': 'td', 'props': [('text-align', 'center'), ('padding', '5px'), ('width', '100px')]}
		]).set_caption("Counts of Images and Annotations per Filter")
		
		return filters_df

	def generate_heatmap(self, output_path=None):
		
		from matplotlib.colors import LinearSegmentedColormap
		from matplotlib import style
		plt.rcParams["font.family"] = "serif"
		plt.rcParams["mathtext.fontset"] = "dejavuserif"

		train_data = self.get_data_from_json(self.train_annotations_path)
		valid_data = self.get_data_from_json(self.valid_annotations_path)
  
		_, cat_names, _ = self.get_categories(train_data)
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

	def generate_galactic_distribution_plot(self, obs_coords_file, splits=['train', 'valid'], output_path=None):
		
		from astropy.coordinates import SkyCoord
		import astropy.units as u
		from astropy.visualization import astropy_mpl_style

		with open(obs_coords_file, 'r') as file:
			coords_data = json.load(file)

		ra = []
		dec = []
		exposures = []

		for split in splits: 
			for image_file in os.listdir(os.path.join(self.dest_dir, self.dataset_name, split)):
				obs = image_file.split('.')[0].replace('_png', '.fits')
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