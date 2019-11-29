from retinanet.dataset import Ring_Cell_all_dataset
from tqdm import tqdm
import numpy as np
import torch
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.nums_py import py_cpu_nms
from metric import compute_overlap, calculate_metric_final_new
import os


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

pred_dict_box = {}
pred_dict_score = {}
result_str = ''

round = 2

nms_threshold = 0.4
score_threshold = 0.5

test_fold = 0

# pred_csv = './test_result_new/retinanet_resnet18_round0_test_fold_0_weight_loss_5_on_test_data_latest.csv'
# pred_csv = 'test_result/retinanet_resnet18_round1_test_fold_0_using_pretrained_model.csv'
pred_csv = './test_result_da_ct/retinanet_resnet18_round1_train_on_fold_0_(01)_result_on_test_fold_0_weight_loss_1_latest_ensemble_0.5(round0).csv'

with open(pred_csv, 'r') as f:
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

test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/test_{}.txt'.format(test_fold))

result_dict = {}

pred_boxes_total = []
pred_scores_total = []
gt_boxes_total = []

total_num_bbox = 0

max_predict_bbox_on_negative_images = 0
min_predict_bbox_on_positive_images = 100000

for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):

	result_dict[image_name] = []
	gt_bbox = bbox
	gt_scores = np.ones(len(gt_bbox)).tolist()
	pred_scores = pred_dict_score[image_name]
	pred_bboxs = pred_dict_box[image_name]

	if len(pred_bboxs) != 0 and len(bbox) == 0:
		if len(pred_bboxs) > max_predict_bbox_on_negative_images:
			max_predict_bbox_on_negative_images = int(torch.Tensor(pred_scores)[torch.Tensor(pred_scores) >= 0.5].size(0))

	if len(bbox) != 0:
		if len(pred_bboxs) < min_predict_bbox_on_positive_images:
			min_predict_bbox_on_positive_images = int(torch.Tensor(pred_scores)[torch.Tensor(pred_scores) >= 0.5].size(0))

	if len(pred_bboxs) != 0:
		# nmsx
		pred_bboxs = torch.Tensor(pred_bboxs).unsqueeze(0)  # size -> [1, num_box, 4]
		pred_scores = torch.Tensor(pred_scores).unsqueeze(0).unsqueeze(-1)  # size -> [1, num_box, 1]
		anchors_nms_idx = nms(torch.cat([pred_bboxs, pred_scores], dim=2)[0, :, :], nms_threshold)

		pred_bboxs = pred_bboxs[0, anchors_nms_idx, :]
		pred_scores = pred_scores[0, anchors_nms_idx, 0]

		# pred_bboxs = pred_bboxs[pred_scores > score_threshold]
		# pred_scores = pred_scores[pred_scores > score_threshold]

		total_num_bbox += int(pred_scores[pred_scores >= 0.05].size(0))

		pred_bboxs = pred_bboxs.numpy().tolist()
		pred_scores = pred_scores.numpy().tolist()

		pred_boxes_total.append(pred_bboxs)
		pred_scores_total.append(pred_scores)
		gt_boxes_total.append(bbox)
	else:
		pred_boxes_total.append([])
		pred_scores_total.append([])
		gt_boxes_total.append(bbox)

recall, precision, froc, FPs, recall_record, precision_record, froc_record, FPs_record, score_record \
			= calculate_metric_final_new(pred_boxes_total, gt_boxes_total, pred_scores_total, score_threshold=score_threshold)

print('recall: {}, precision: {}, froc: {}, FPs:{}'.format(recall[-1], precision[-1], froc, FPs))

print('recall: {}, FPs: {}, froc:{}, score threshold: {} when precision is {}'.
	  format(recall_record[-1], FPs_record, froc_record, score_record[-1], precision_record[-1]))

print(pred_csv)


print(max_predict_bbox_on_negative_images)
print(min_predict_bbox_on_positive_images)