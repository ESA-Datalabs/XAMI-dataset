import os
import json
from re import split
import zipfile
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from .xami_utils import dataset_utils

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