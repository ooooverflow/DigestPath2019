# from lib_new.nms.gpu_nms import gpu_nms
#
# from lib_new.nms.nums_py import py_cpu_nms
# import torch
# import random
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# import time
#
# def nms(dets, thresh):
# 	"Dispatch to either CPU or GPU NMS implementations.\
# 	Accept dets as tensor"""
# 	dets = dets.cpu().detach().numpy()
# 	return py_cpu_nms(dets, thresh)
#
# import torchvision
# # torchvision.ops.nms(boxes, scores, iou_threshold)
# # from lib.nms.pth_nms import pth_nms
# #
# #
# # def nms(dets, thresh):
# # 	"Dispatch to either CPU or GPU NMS implementations.\
# # 	Accept dets as tensor"""
# # 	return pth_nms(dets, thresh)
# #
# num = 100000
#
# x = [[random.randint(0,200), random.randint(0,200), random.randint(200, 1000), random.randint(200, 1000)] for y in range(num)]
# transformed_anchors = torch.Tensor(x)
# # print(x.size())
# # x = torch.cat([x, x, x], 0)
# # print(x.size())
# # transformed_anchors = torch.Tensor(x).cuda()
# # x = torch.Tensor(x)
# # transformed_anchors = torch.Tensor(x).unsqueeze(0).cuda()
# x = [random.random() for y in range(num)]
# scores = torch.Tensor(x)
# # scores = torch.cat([x, x, x], 0).cuda()
# t1 = time.time()
# anchors_nms_idx = torchvision.ops.nms(transformed_anchors, scores, 0.3)
# t2 = time.time()
# print(t2 - t1)
# print(len(anchors_nms_idx))
# print(anchors_nms_idx)
#
#
# #
# #
# # # transformed_anchors = torch.Tensor([[10,10, 40, 40], [15, 15, 40, 40]]).unsqueeze(0)
# # # scores = torch.Tensor([[1, 0.5]]).unsqueeze(2)
# transformed_anchors = transformed_anchors.unsqueeze(0)
# scores = torch.Tensor(x).unsqueeze(0).unsqueeze(2)
# t3 = time.time()
# anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.3)
# t4 = time.time()
# print(t4 - t3)
# print(len(anchors_nms_idx))
# print(anchors_nms_idx)
# # print(len(anchors_nms_idx))
# # # print(anchors_nms_idx)
# # print(type(anchors_nms_idx))
# import os
# # import glob
# # x = 'ckpt_new/best_recall_*'
# # print(glob.glob(x))

def generate_probablity_map(label_dict):
	for image_name in label_dict:
		for bbox in label_dict[image_name]:
			pass

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

csv_path = './bbox/retinanet_resnet18_training_data_with_confidence_score_using_test_data_prediction.csv'

pred_dict_box = get_label(csv_path)

import numpy as np
import cv2
import os
image_name = 'D20180628401_2019-05-21 18_36_19-lv0-16939-29532-2036-2080'
bboxs_with_scores = pred_dict_box[image_name]
bboxs_with_scores = np.array(bboxs_with_scores)
# scores = bboxs_with_scores[:, 4]
# bboxs = bboxs_with_scores[:, :4]
bboxs = bboxs_with_scores

image = cv2.imread(os.path.join('/data/sqy/challenge/MICCAI2019/Signet_ring_cell_dataset/sig-train-pos', image_name+'.jpeg'))
# cv2.imwrite('test1.jpg', image)
# probability_map = np.ones_like(image)[:, :, 0].astype(np.float)
# probability_map[bboxs[:, 0]: bboxs[:, 1], bboxs[:, 2]: bboxs[:,3]] = 0
h, w = image.shape[:2]
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

t1 = time.time()

probability_map = np.ones([h, w]).astype(np.float)

for i, bbox in enumerate(bboxs):
	score = bbox[4]
	if score == 0:
		weight = 20
	else:
		weight = 1 / score
		if weight >= 20:
			weight = 20

	x1 = int(bbox[2]) - 253
	y1 = int(bbox[3]) - 253
	x2 = int(bbox[0]) + 253
	y2 = int(bbox[1]) + 253

	x1 = np.clip(x1, a_min=255, a_max=w-256)
	y1 = np.clip(y1, a_min=255, a_max=h-256)
	x2 = np.clip(x2, a_min=255, a_max=w-256)
	y2 = np.clip(y2, a_min=255, a_max=h-256)

	probability_map[int(y1): int(y2)+1, int(x1): int(x2)+1] += weight

t2 = time.time()
print(probability_map.shape)
print(t2-t1)
print(np.min(probability_map))

probability_map_view = probability_map.reshape(-1)
# probability_map = probability_map.reshape(-1)
probability_map_view = probability_map_view / sum(probability_map_view)
# print(probability_map)
print(np.max(probability_map_view))
print(np.min(probability_map_view))

cord = np.arange(probability_map_view.shape[0])
f, ax = plt.subplots(figsize=(12, 9))
ax = sns.heatmap(probability_map, cmap='rainbow', xticklabels=False, yticklabels=False)


# for i in range(1000):
# 	# choice = np.random.choice(cord, p=probability_map_view)
# 	choice = np.random.choice(cord)
# 	y = choice // w
# 	x = choice % w
# 	x = np.clip(x, a_min=255, a_max=w - 256)
# 	y = np.clip(y, a_min=255, a_max=h - 256)
#
# 	plt.scatter(x, y)


plt.savefig('test7.jpg')



import random

def __in_which_part(n, w):
	for i, v in enumerate(w):
		if n < v:
			return i
	return len(w) - 1


def weighting_choice(data, weightings):
	s = sum(weightings)
	w = [float(x) / s for x in weightings]

	t = 0
	for i, v in enumerate(w):
		t += v
		w[i] = t

	c = __in_which_part(random.random(), w)
	try:
		return data[c]
	except IndexError:
		return data[-1]


# print('weighting_choice', weighting_choice(['a', 'b'], [10, 90]))