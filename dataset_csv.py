from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import torch
import cv2
from utils import get_box
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from torchvision import transforms
from PIL import Image
import random

def collate_fn(batch):
	'''
	In order to return bbox in batch, we add zero tensor if there is not enough bbox
	Example: if batch size if 4, and bbox size is [1, 6], [1, 6], [1, 6], [2, 6] seperately, then the returned bbox size
	will be [4(batch size), 2(max num bbox), 6(coordinate, class, instance)]
	:param batch:
	:return:
	'''
	image, bbox, image_ = zip(*batch)
	max_len = 0
	for box in bbox:
		if len(box) > max_len:
			max_len = len(box)

	if max_len > 0:
		bbox_padded = np.ones((len(bbox), max_len, 5)) * -1

		for index, box in enumerate(bbox):
			if box.shape[0] > 0:
				if box.shape[1] == 4:
					bbox_padded[index, :len(box), :4] = box
					bbox_padded[index, : len(box), 4] = 1
				else:
					bbox_padded[index, :len(box), :] = box

		bbox_padded = torch.Tensor(bbox_padded)
	else:
		bbox_padded = torch.ones((len(bbox), 1, 5)) * -1

	image = torch.stack(image, 0)
	image_ = torch.stack(image_, 0)

	return image, bbox_padded, image_

def random_crop(image, bbox=None, crop_w=512, crop_h=512):
	'''
	:param image: numpy.ndarry -> [H, W, 3]
	:param bbox:  list -> [[x1, y1, x2, y2], ...]
	:return:
	'''
	h, w = image.shape[:2]
	xmin = np.random.randint(0, w - 1 - crop_w)
	ymin = np.random.randint(0, h - 1 - crop_h)
	xmax = xmin + crop_w
	ymax = ymin + crop_h
	image = image[ymin: ymax, xmin: xmax]
	if bbox is not None:
		bbox = np.array(bbox)
		# exclude bbox out of range
		bbox = bbox[bbox[:, 0] < xmax - 8]
		bbox = bbox[bbox[:, 1] < ymax - 8]
		bbox = bbox[bbox[:, 2] > xmin + 7]
		bbox = bbox[bbox[:, 3] > ymin + 7]

		bbox[:, 0] = np.clip(bbox[:, 0] - xmin, 0, crop_w - 1)
		bbox[:, 1] = np.clip(bbox[:, 1] - ymin, 0, crop_h - 1)
		bbox[:, 2] = np.clip(bbox[:, 2] - xmin, 0, crop_w - 1)
		bbox[:, 3] = np.clip(bbox[:, 3] - ymin, 0, crop_h - 1)

		return image, bbox
	else:
		return image

def get_label(csv_path):
	pred_dict_box = {}
	with open(csv_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line[:-1]
			line = line.split(',')
			image_name = line[0]
			pred_dict_box[image_name] = []
			if len(line[1]) != 0:
				preds = line[1].split(';')[:-1]
				for pred in preds:
					pred = pred.split(' ')
					box = []
					for elemet in pred[:-1]:
						box.append(float(elemet))
					pred_dict_box[image_name].append(box)

	return pred_dict_box

seq = iaa.Sequential([
    # iaa.Affine(scale=(0.8, 1.2),
    #            rotate=(-10, 10)),
    # iaa.GaussianBlur((0, 1.0)),
    # iaa.Add((-20, 30)),
    # iaa.GammaContrast((0.8, 1.2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((0,3))
], random_order=True)


class Ring_Cell_random_crop_all(Dataset):
	# include pos and neg samples
	# pos:neg = 1:5
	def __init__(self, txt_path, label_csv_path, training=True, mixup=False):
		with open(txt_path, 'r') as f:
			lines = f.readlines()
		# balance the pos and neg samples
		# lines_pos = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 25
		# lines_neg = [x for x in lines if not os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 5

		lines_pos = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 50
		lines_neg = [x for x in lines if not os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 10

		# random crop 25 times every epoch
		self.lines = lines_pos
		self.lines.extend(lines_neg)
		self.training = training
		self.to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])
		self.label_dict = get_label(label_csv_path)
		# random.shuffle(self.lines)
		self.mixup = mixup

	def __getitem__(self, index):
		image_path = self.lines[index][:-1]
		image = cv2.imread(image_path)
		image_name = image_path.split('/')[-1].split('.')[0]
		if image_name not in self.label_dict:
			bbox = []
		else:
			bbox = self.label_dict[image_name]

		if len(bbox) != 0:
			image, bbox = random_crop(image, bbox)
			bbox = bbox.tolist()
		else:
			image = random_crop(image)

		if self.mixup == True:
			index_mixup = np.random.randint(0, len(self.lines))
			image_path_mixup = self.lines[index_mixup][:-1]
			image_mixup = cv2.imread(image_path_mixup)
			image_name_mixup = image_path_mixup.split('/')[-1].split('.')[0]
			if image_name_mixup not in self.label_dict:
				bbox_mixup = []
			else:
				bbox_mixup = self.label_dict[image_name_mixup]

			if len(bbox_mixup) != 0:
				image_mixup, bbox_mixup = random_crop(image_mixup, bbox_mixup)
				bbox_mixup = bbox_mixup.tolist()
			else:
				image_mixup = random_crop(image_mixup)

			ratio_mixup = np.random.beta(a=1.5, b=1.5)
			image = image * ratio_mixup + image_mixup * (1-ratio_mixup)

			score = np.zeros((len(bbox) + len(bbox_mixup)))
			score[:len(bbox)] = ratio_mixup
			score[len(bbox):] = 1 - ratio_mixup
			bbox.extend(bbox_mixup)


		if self.training:
			# data augmentation
			seq_det = seq.to_deterministic()
			if len(bbox) != 0:
				bbs = []
				for box in bbox:
					bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
				bbs = ia.BoundingBoxesOnImage(bbs, shape=image.shape)
				bbs_ = seq_det.augment_bounding_boxes([bbs])
				bbox = []
				if self.mixup == True:
					for idx, box in enumerate(bbs_[0].bounding_boxes):
						bbox.append([box.x1, box.y1, box.x2, box.y2, score[idx]])
				else:
					for idx, box in enumerate(bbs_[0].bounding_boxes):
						bbox.append([box.x1, box.y1, box.x2, box.y2])

			image = seq_det.augment_image(image)

		if len(bbox) != 0:
			bbox = np.clip(bbox, a_min=0, a_max=511)

		image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
		image = self.to_tensor(Image.fromarray(image.astype(np.uint8)))

		return image, np.array(bbox), image_
		# if os.path.exists(xml_path):
		# 	return image, bbox, image_
		# else:
		# 	return image, np.array([]), image_


	def __len__(self):
		return len(self.lines)

if __name__ == '__main__':
	dataset = Ring_Cell_random_crop_all(txt_path='../train_test_4/train_0.txt',
										label_csv_path='./bbox/retinanet_resnet_18_train_0_nms_0.4_scorethreshold_0.2.csv',
										mixup=True)

	dataloader = DataLoader(
		dataset,
		batch_size=8,
		shuffle=True,
		num_workers=4,
		collate_fn=collate_fn
	)
	for i, (image, bbox, image_) in enumerate(dataset):
		# print(bbox)
		image_ = np.array(image_).transpose((1, 2, 0))
		for box in bbox:
			image_ = cv2.rectangle(image_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
		cv2.imwrite(os.path.join('dataset_test', '{}.jpg'.format(i)), image_)