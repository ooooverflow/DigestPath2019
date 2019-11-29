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
import time

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
		bbox_padded = np.ones((len(bbox), max_len, 6)) * -1

		for index, box in enumerate(bbox):
			if box.shape[0] > 0:
				bbox_padded[index, :len(box), :5] = box
				bbox_padded[index, : len(box), 4] = 1

		bbox_padded = torch.Tensor(bbox_padded)
	else:
		bbox_padded = torch.ones((len(bbox), 1, 6)) * -1

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

def random_crop_with_confidence_score(image, bboxs, pm, crop_w=512, crop_h=512):
	h, w = image.shape[:2]
	# probability_map = np.ones([h, w]).astype(np.float)
	# # generate the probability map of the center point of the bbox
	# for i, bbox in enumerate(bboxs):
	# 	score = bbox[4]
	# 	if score == 0:
	# 		weight = 20
	# 	else:
	# 		weight = 1 / score
	# 		if weight >= 20:
	# 			weight = 20
	#
	# 	x1 = int(bbox[2]) - 253
	# 	y1 = int(bbox[3]) - 253
	# 	x2 = int(bbox[0]) + 253
	# 	y2 = int(bbox[1]) + 253
	#
	# 	x1 = np.clip(x1, a_min=256, a_max=w - 256)
	# 	y1 = np.clip(y1, a_min=256, a_max=h - 256)
	# 	x2 = np.clip(x2, a_min=256, a_max=w - 256)
	# 	y2 = np.clip(y2, a_min=256, a_max=h - 256)
	# 	probability_map[int(y1): int(y2) + 1, int(x1): int(x2) + 1] += weight
	probability_map_view = pm.reshape(-1)
	# probability_map = probability_map.reshape(-1)
	# probability_map_view = probability_map_view / np.sum(probability_map_view)
	coordinates_range = np.arange(probability_map_view.shape[0])
	coordinate = np.random.choice(coordinates_range, p=probability_map_view)
	# x, y is the center point of the bbox
	y = coordinate // w
	x = coordinate % w
	x = np.clip(x, a_min=256, a_max=w - 256)
	y = np.clip(y, a_min=256, a_max=h - 256)

	xmin = x - 256
	ymin = y - 256
	xmax = x + 256
	ymax = y + 256

	image = image[ymin: ymax, xmin: xmax]

	if bboxs is not None:
		bbox = np.array(bboxs)
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
    iaa.Affine(scale=(0.8, 1.2),
               rotate=(-10, 10)),
    iaa.GaussianBlur((0, 1.0)),
    iaa.Add((-20, 30)),
    iaa.GammaContrast((0.8, 1.2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((0,3))
], random_order=True)


class Ring_Cell_random_crop_all(Dataset):
	# include pos and neg samples
	# pos:neg = 1:5
	def __init__(self, txt_path, confidence_csv_path, pm_dir, training=True):
		with open(txt_path, 'r') as f:
			lines = f.readlines()
		# balance the pos and neg samples
		lines_pos = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 25
		lines_neg = [x for x in lines if not os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 5
		# random crop 25 times every epoch
		self.lines = lines_pos
		self.lines.extend(lines_neg)
		self.training = training
		self.to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])
		self.label_dict = get_label(confidence_csv_path)
		self.pm_dir = pm_dir


		# random.shuffle(self.lines)

	def __getitem__(self, index):
		image_path = self.lines[index][:-1]
		image = cv2.imread(image_path)
		image_name = image_path.split('/')[-1].split('.')[0]
		bbox = self.label_dict[image_name]

		if len(bbox) != 0:
			# t1 = time.time()
			pm = np.load(os.path.join(self.pm_dir, image_name + '.npy'))
			image, bbox = random_crop_with_confidence_score(image, bbox, pm)
			# t2 = time.time()
			# image, bbox = random_crop(image, bbox)
		else:
			image = random_crop(image)
		if self.training:
			# data augmentation
			seq_det = seq.to_deterministic()
			if len(bbox) != 0:
				bbs = []
				confidence = []
				for box in bbox:
					bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
					confidence.append(box[4])
				bbs = ia.BoundingBoxesOnImage(bbs, shape=image.shape)
				bbs_ = seq_det.augment_bounding_boxes([bbs])
				bbox = []
				for idx, box in enumerate(bbs_[0].bounding_boxes):
					bbox.append([box.x1, box.y1, box.x2, box.y2, confidence[idx]])

			image = seq_det.augment_image(image)

		if len(bbox) != 0:
			bbox = np.clip(bbox, a_min=0, a_max=511)

		image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
		image = self.to_tensor(Image.fromarray(image))

		return image, np.array(bbox), image_
		# if os.path.exists(xml_path):
		# 	return image, bbox, image_
		# else:
		# 	return image, np.array([]), image_


	def __len__(self):
		return len(self.lines)

if __name__ == '__main__':
	dataset = Ring_Cell_random_crop_all(txt_path='../train_test_4/train_0.txt',
						confidence_csv_path='./bbox/retinanet_resnet18_training_data_with_confidence_score_using_test_data_prediction.csv',
						pm_dir='probability_map/from_round0_test_prediction_new')
	dataloader = DataLoader(
		dataset,
		batch_size=32,
		shuffle=True,
		collate_fn=collate_fn,
		num_workers=8
	)
	for i, (image, bbox, image_) in enumerate(dataloader):
		print(bbox.size())