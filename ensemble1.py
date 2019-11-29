import os
from retinanet.dataset import Ring_Cell_all_dataset
from tqdm import tqdm
import numpy as np
import torch
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.nums_py import py_cpu_nms, py_cpu_nms_contain, py_cpu_nms_exclude
from metric import compute_overlap
import glob

def calculate_metric_final(pred_bboxes, gt_bboxes, pred_scores, iou_threshold=0.3, score_threshold=0.5):
	'''
	:param pred_bboxes: list -> [num_pic, num_box, 4] (sorted already, descending order)
	:param gt_bboxes: list -> [num_pic, num_box, 4]
	:param pred_scores: list -> [num_pic, num_box]
	:return:
	'''
	false_positives = np.zeros((0,))
	true_positives = np.zeros((0,))
	scores = np.zeros((0,))
	num_annotations = 0.0

	# scores of predict box in negative image
	scores_normal_region = np.zeros((0,))

	num_pos = 0

	normal_regions = 0
	FPs = 0

	for i in range(len(pred_bboxes)):
		detections = pred_bboxes[i]
		annotations = np.array(gt_bboxes[i])
		num_annotations += len(annotations)
		if len(annotations) != 0:
			num_pos += 1
			# positive region
			# calculate precision and recall
			detected_annotations = []

			for j, d in enumerate(detections):
				score = pred_scores[i][j]
				if score < 0.05:
					# score has been sorted in descending order
					break
				scores = np.append(scores, score)

				overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
				assigned_annotation = np.argmax(overlaps, axis=1)
				max_overlap = overlaps[0, assigned_annotation]

				if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
					false_positives = np.append(false_positives, 0)
					true_positives = np.append(true_positives, 1)
					detected_annotations.append(assigned_annotation)
				else:
					false_positives = np.append(false_positives, 1)
					true_positives = np.append(true_positives, 0)
		else:
			# negative region (normal region)
			# calculate FPs
			normal_regions += 1
			for j, d in enumerate(detections):
				score = pred_scores[i][j]
				if score < 0.05:
					# score has been sorted in descending order
					break
				FPs += 1
				scores_normal_region = np.append(scores_normal_region, score)

	indices = np.argsort(-scores)
	scores = scores[indices]
	false_positives = false_positives[indices]
	true_positives = true_positives[indices]

	indices = np.argsort(-scores_normal_region)
	scores_normal_region = scores_normal_region[indices]

	# compute false positives and true positives
	false_positives = np.cumsum(false_positives)
	true_positives = np.cumsum(true_positives)

	# compute recall and precision
	recall = true_positives / num_annotations
	if len(recall) == 0:
		recall = [0]
	precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
	if len(precision) == 0:
		precision = [0]

	# index where precision greater equal 0.2
	index_record = np.where(precision >= 0.2)[0][-1]
	# recall, precision, FPs when precision is 0.2
	recall_record = recall[:index_record+1]
	precision_record = precision[:index_record+1]
	score_record = scores[:index_record+1]

	scores_normal_region_record = scores_normal_region[scores_normal_region > score_record[-1]]
	FPs_record = scores_normal_region_record.shape[0]
	FPs_record = float(FPs_record / normal_regions)
	FPs_record = max(100 - FPs_record, 0)

	# compute FROC when precision is 0.2
	fps_list = [1, 2, 4, 8, 16, 32]
	recall_list = []
	for fps in fps_list:
		total_fps_num = fps * normal_regions
		if total_fps_num >= len(scores_normal_region_record):
			recall_list.append(float(recall_record[-1]))
		else:
			score_min = scores_normal_region_record[total_fps_num - 1]
			score_index = np.where(score_record >= score_min)[0]
			if score_index.shape[0] == 0:
				recall_list.append(0)
			else:
				score_index = score_index[-1]
				recall_list.append(float(recall_record[score_index]))
	froc_record = np.mean(recall_list)


	recall = recall[scores > score_threshold]
	precision = precision[scores > score_threshold]
	scores = scores[scores > score_threshold]

	scores_normal_region = scores_normal_region[scores_normal_region > score_threshold]

	# compute FROC
	fps_list = [1, 2, 4, 8, 16, 32]
	recall_list = []
	for fps in fps_list:
		total_fps_num = fps * normal_regions
		if total_fps_num >= len(scores_normal_region):
			recall_list.append(float(recall[-1]))
		else:
			score_min = scores_normal_region[total_fps_num-1]
			score_index = np.where(scores>=score_min)[0]
			if score_index.shape[0] == 0:
				recall_list.append(0)
			else:
				score_index = score_index[-1]
				recall_list.append(float(recall[score_index]))
	froc = np.mean(recall_list)

	FPs = float(len(scores_normal_region) / normal_regions)
	FPs = max(100 - FPs, 0)

	return recall, precision, froc, FPs, recall_record, precision_record, froc_record, FPs_record, score_record

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	return py_cpu_nms(dets, thresh)

def nms_contain(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	return py_cpu_nms_contain(dets, thresh)

def nms_exclude(dets, thresh, **kwargs):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	return py_cpu_nms_exclude(dets, thresh, **kwargs)

# ensemble_csv_list = []

result_dir = 'test_result_da'
csv_name = os.path.join(result_dir, 'retinanet_resnet18_round0_fold_0_weight_loss_1_on_train_data_best_valid_recall_*_new1.csv')

test_fold = csv_name.split('_fold_')[1][0]
print(test_fold)
ensemble_csv_list = glob.glob(csv_name)
print(len(ensemble_csv_list))
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round4_fold_0_weight_loss_1_on_test_data_best_valid_recall_original_0.5(round0)_0.4(round1)_0.4(round2)_0.4_multi3(round3).csv'))

for i in ensemble_csv_list:
	print(i)

# ensemble_csv_list = [x for x in ensemble_csv_list if 'new1' not in x]
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round0_test_fold_0_weight_loss_1_on_test_data_latest.csv'))
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round0_test_fold_0_weight_loss_10_on_test_data_latest.csv'))
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round0_fold_0_weight_loss_1_on_train_data_best_valid_recall.csv'))
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round0_fold_0_weight_loss_1_on_train_data_best_valid_recall_new1.csv'))
# ensemble_csv_list.append(os.path.join(result_dir, 'retinanet_resnet18_round0_test_fold_0_weight_loss_1_on_test_data_latest.csv'))

score_threshold = 0.4
nms_threshold = 0.4

def get_info(pred_csv, box_dict, score_dict):
	with open(pred_csv, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line[:-1]
			line = line.split(',')
			image_name = line[0]
			if image_name not in box_dict:
				box_dict[image_name] = []
				score_dict[image_name] = []
			if len(line[1]) != 0:
				preds = line[1].split(';')[:-1]
				for pred in preds:
					pred = pred.split(' ')
					box = []
					for elemet in pred[:-1]:
						box.append(float(elemet))
					box_dict[image_name].append(box)
					score_dict[image_name].append(float(pred[-1]))

	return box_dict, score_dict


pred_dict_box = {}
pred_dict_score ={}

for i, ensemble_csv in enumerate(ensemble_csv_list):
	pred_dict_box, pred_dict_score = get_info(ensemble_csv, pred_dict_box, pred_dict_score)


test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/train_{}.txt'.format(test_fold))

result_dict = {}
score_dict = {}
pred_boxes_total = []
pred_scores_total = []
gt_boxes_total = []


for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
	result_dict[image_name] = []
	score_dict[image_name] = []
	gt_bbox = bbox
	gt_scores = np.ones(len(gt_bbox)).tolist()
	# pred_scores = pred_dict_score[image_name]
	# pred_bboxs = pred_dict_box[image_name]


	if len(bbox) != 0:
		pred_scores = pred_dict_score[image_name]
		pred_bboxs = pred_dict_box[image_name]
		scores = gt_scores
		scores.extend(pred_scores)
		bboxs = gt_bbox
		bboxs.extend(pred_bboxs)

		scores = np.array(scores)
		bboxs = np.array(bboxs)

		bboxs = bboxs[scores >= score_threshold]
		scores = scores[scores >= score_threshold]
		# nms
		pred_bboxs = torch.Tensor(bboxs).unsqueeze(0)  # size -> [1, num_box, 4]
		pred_scores = torch.Tensor(scores).unsqueeze(0).unsqueeze(-1)  # size -> [1, num_box, 1]

		# anchors_nms_idx = nms_exclude(torch.cat([pred_bboxs, pred_scores], dim=2)[0, :, :], nms_threshold, vote_num=3)
		anchors_nms_idx = nms(torch.cat([pred_bboxs, pred_scores], dim=2)[0, :, :], nms_threshold)

		pred_boxes = pred_bboxs[:, anchors_nms_idx, :]
		pred_scores = pred_scores[:, anchors_nms_idx, :]

		anchors_nms_idx = nms_contain(torch.cat([pred_boxes, pred_scores], dim=2)[0, :, :], 0.8)

		bboxs = pred_boxes[0, anchors_nms_idx, :]
		scores = pred_scores[0, anchors_nms_idx, 0]

		# bboxs = bboxs[scores >= score_threshold]
		# scores = scores[scores >= score_threshold]

		bboxs = bboxs.numpy().tolist()
		scores = scores.numpy().tolist()

		result_dict[image_name].extend(bboxs)
		score_dict[image_name].extend(scores)


result_str = ''
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
	result_str += '\n'

result_csv = './bbox/retinanet_resnet18_round1_fold_0_weight_loss_1_on_train_data_best_valid_recall_0.4_ensemble(round0)_new1.csv'.format(score_threshold)
with open(result_csv, 'w') as f:
	f.write(result_str)

# result_csv = result_csv.replace('bbox', 'test_result_da')


# recall, precision, froc, FPs, recall_record, precision_record, froc_record, FPs_record, score_record\
# 	= calculate_metric_final(pred_boxes_total, gt_boxes_total, pred_scores_total, score_threshold=0.2)
#
# print('recall: {}, precision: {}, froc: {}, FPs:{}'.format(recall[-1], precision[-1], froc, FPs))
#
#
# print('recall: {}, FPs: {}, froc:{}, score threshold: {} when precision is {}'.
# 	  format(recall_record[-1], FPs_record, froc_record, score_record[-1], precision_record[-1]))
