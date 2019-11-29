from retinanet.dataset import Ring_Cell_all_dataset
from tqdm import tqdm
import numpy as np
import torch
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.gpu_nms import gpu_nms
from metric import compute_overlap
import matplotlib.pyplot as plt
import matplotlib
import os
import glob


result_str = ''

for test_fold in range(0,4):
	test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/test_{}.txt'.format(test_fold))

	pred_dict_box = {}
	pred_dict_score = {}
	pred_dict_max_valid_recall = {}

	csv_path = './test_result_new/retinanet_resnet18_round0_fold_{}_weight_loss_1_on_test_data_latest.csv'.format(test_fold)

	with open(csv_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line[:-1]
			line = line.split(',')
			image_name = line[0]
			pred_dict_box[image_name] = []
			pred_dict_score[image_name] = []
			if len(line[1]) != 0:
				preds = line[1].split(';')[:-1]
				for pred in preds:
					pred = pred.split(' ')
					box = []
					for elemet in pred[:-1]:
						box.append(float(elemet))
					pred_dict_box[image_name].append(box)
					pred_dict_score[image_name].append(float(pred[-1]))


	result_dict = {}
	nms_threshold = 0.4
	scores = np.zeros((0,))
	iou_threshold = 0.3
	scores_all_prediction = []




	for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
		result_dict[image_name] = []
		detected_annotations = []
		scores_all_prediction.extend(pred_dict_score[image_name])

		false_positives = np.zeros((0,))
		true_positives = np.zeros((0,))
		scores = np.zeros((0,))
		num_annotations = len(bbox)

		if len(bbox) != 0:
			# pos image
			pred_scores = pred_dict_score[image_name]
			detections = pred_dict_box[image_name]
			annotations = np.array(bbox)
			for j, d in enumerate(detections):
				score = pred_scores[j]
				scores = np.append(scores, score)
				overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
				assigned_annotation = np.argmax(overlaps, axis=1)
				max_overlap = overlaps[0, assigned_annotation]

				if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
					detected_annotations.append(assigned_annotation)
					gt_box = bbox[assigned_annotation[0]]
					gt_box.append(score)
					result_dict[image_name].append(gt_box)
					false_positives = np.append(false_positives, 0)
					true_positives = np.append(true_positives, 1)
				else:
					false_positives = np.append(false_positives, 1)
					true_positives = np.append(true_positives, 0)

			indices = np.argsort(-scores)
			scores = scores[indices]
			false_positives = false_positives[indices]
			true_positives = true_positives[indices]

			false_positives = np.cumsum(false_positives)
			true_positives = np.cumsum(true_positives)

			recall = true_positives / num_annotations
			precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

			index_record = np.where(precision >= 0.2)[0]

			if index_record.shape[0] != 0:
				index_record = index_record[-1]
				# recall, precision, FPs when precision is 0.2
				recall_record = recall[:index_record + 1]
				precision_record = precision[:index_record + 1]
				score_record = scores[:index_record + 1]
			else:
				recall_record = [0]
				precision_record = [0]
				score_record = [0]

			pred_dict_max_valid_recall[image_name] = recall_record[-1]


		for j in range(len(bbox)):
			if j not in detected_annotations:
				gt_box = bbox[j]
				gt_box.append(0)
				result_dict[image_name].append(gt_box)


	for image_name in result_dict:
		result_str += image_name
		result_str += ','
		results = result_dict[image_name]
		for result in results:
			box = result
			for element in box:
				result_str += str(element)
				result_str += ' '
			result_str += ';'

		if image_name not in pred_dict_max_valid_recall:
			result_str += '-1'
		else:
			result_str += str(pred_dict_max_valid_recall[image_name])

		result_str += '\n'

result_csv = './bbox/retinanet_resnet18_training_data_with_confidence_score_of_whole_image.csv'

with open(result_csv, 'w') as f:
	f.write(result_str)







