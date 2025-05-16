from __future__ import print_function, division
import torch
from skimage import io
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import os
import cv2

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		h, w = image.shape[-2:]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[:, top: top + new_h, left: left + new_w]
		label = label[:, top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}
class RandomHorizontalFlip(object):
	def __init__(self, p=0.5):
		assert 0 <= p <= 1
		self.p = p

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() < self.p:
			image = np.flip(image, dims=[2])
			label = np.flip(label, dims=[2])

		return {'imidx': imidx, 'image': image, 'label': label}
class RandomVerticalFlip(object):
	def __init__(self, p=0.5):
		assert 0 <= p <= 1
		self.p = p

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() < self.p:
			image = torch.flip(image, dims=[1])
			label = torch.flip(label, dims=[1])

		return {'imidx': imidx, 'image': image, 'label': label}
class RandomScale(object):
	def __init__(self, scale_range=(1.0, 1.4)):
		self.scale_range = scale_range

	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

		h, w = image.shape[-2:]

		new_h, new_w = int(h * scale_factor), int(w * scale_factor)
		image = cv2.resize(image.transpose(1, 2, 0), (new_w, new_h))
		image = image.transpose(2, 0, 1)

		label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
		label = np.expand_dims(label, axis=-1)

		top = (new_h - h) // 2
		left = (new_w - w) // 2

		image = image[:, top:top + h, left:left + w]
		label = label[top:top + h, left:left + w, :]

		return {'imidx': imidx, 'image': image, 'label': label}
class RandomFourSplitFlipAndRecombine(object):
	def __init__(self, p=0.5, flip_p=0.5):
		assert 0 <= p <= 1
		assert 0 <= flip_p <= 1
		self.p = p
		self.flip_p = flip_p
	def __call__(self, sample):
		if random.random() >= self.p:
			return sample
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		h, w = image.shape[1:]

		top_left_img, top_right_img = image[:, :h // 2, :w // 2], image[:, :h // 2, w // 2:]
		bottom_left_img, bottom_right_img = image[:, h // 2:, :w // 2], image[:, h // 2:, w // 2:]

		top_left_label, top_right_label = label[:, :h // 2, :w // 2], label[:, :h // 2, w // 2:]
		bottom_left_label, bottom_right_label = label[:, h // 2:, :w // 2], label[:, h // 2:, w // 2:]

		# 定义四个部分的图像和标签
		parts_img = [top_left_img, top_right_img, bottom_left_img, bottom_right_img]
		parts_label = [top_left_label, top_right_label, bottom_left_label, bottom_right_label]

		# 随机翻转每个部分
		for i in range(4):
			if random.random() < self.p:
				# 随机选择翻转方向
				if random.random() < 0.5:
					# 垂直翻转
					parts_img[i] = torch.flip(parts_img[i], dims=[1])
					parts_label[i] = torch.flip(parts_label[i], dims=[1])
				else:
					# 水平翻转
					parts_img[i] = torch.flip(parts_img[i], dims=[2])
					parts_label[i] = torch.flip(parts_label[i], dims=[2])

		parts = list(zip(parts_img, parts_label))
		random.shuffle(parts)
		parts_img, parts_label = zip(*parts)
		top_img = torch.cat((parts_img[0], parts_img[1]), dim=1)
		bottom_img = torch.cat((parts_img[2], parts_img[3]), dim=1)
		recombined_image = torch.cat((top_img, bottom_img), dim=2)

		top_label = torch.cat((parts_label[0], parts_label[1]), dim=1)
		bottom_label = torch.cat((parts_label[2], parts_label[3]), dim=1)
		recombined_label = torch.cat((top_label, bottom_label), dim=2)
		return {'imidx': imidx, 'image': recombined_image, 'label': recombined_label}
class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag
	def __call__(self, sample):
		imidx, image, label =sample['imidx'], sample['image'], sample['label']
		tmpImg = np.zeros((2,image.shape[1], image.shape[2]))
		tmpImg[0, :, :] = (image[0, :, :] + 10.378) / 3.286
		tmpImg[1, :, :] = (image[1, :, :] + 17.224) / 3.738

		tmpImg[0, :, :] = np.nan_to_num(tmpImg[0, :, :], nan=0)
		tmpImg[1, :, :] = np.nan_to_num(tmpImg[1, :, :], nan=0)

		# mean = (-10.378, -17.224)  std = (3.286, 3.738)
		tmpImg = tmpImg.transpose((0, 1, 2))
		tmpLbl = label.transpose((2, 0, 1))
		tmpImg  = tmpImg.copy()
		tmpLbl = tmpLbl.copy()

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}
class FloodDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		label = io.imread(self.label_name_list[idx])
		label = label[:,:,np.newaxis]

		filename = os.path.basename(imname)
		region_name = filename.split("_")[0]
		sample = {'imidx': imidx, 'image': image, 'label': label, 'region_name': region_name}

		if self.transform:
			sample = self.transform(sample)

		return sample
