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
from lib_new.nms.nums_py import py_cpu_nms

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	# return gpu_nms(dets, thresh)
	return py_cpu_nms(dets, thresh)

csv_dir = './test_result_da_ct'
csv_path_list = glob.glob(os.path.join(csv_dir, 'retinanet_resnet18_round0_train_on_fold_0_1_result_on_fold_0_1_weight_loss_1_best_valid_recall_original.csv'))

for pred_csv in csv_path_list:
	# if pred_csv.split('/')[-1] != 'retinanet_resnet18_round0_fold_0_weight_loss_1_on_train_data_best_valid_recall_new1.csv':
	# 	continue
	# pred_csv = './test_result/retinanet_resnet101_round{}_test_fold_{}.csv'.format(round, test_fold)

	# if not os.path.exists(pred_csv):
	# 	# 	continue
	# 	# if os.path.exists(pred_csv.replace('.csv', '.jpg')):
	# 	# 	continue
	# 	#
	# 	# if os.path.exists(ap_image):
	# 	# 	continue


	test_fold = pred_csv.split('_fold_')[1][0]
	cooperative = pred_csv.split('_fold_')[1][2]

	# test_fold = 0

	# dataset = pred_csv.split('_data_')[0][-1]
	# if dataset == 't':
	# 	dataset = 'test'
	# else:
	# 	dataset = 'train'
	dataset = 'train'

	pred_dict_box = {}
	pred_dict_score = {}
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

	test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/train_0_1.txt'.format(dataset, test_fold, cooperative))

	result_dict = {}

	nms_threshold = 0.3
	scores = np.zeros((0,))
	iou_threshold = 0.3
	scores_all_prediction = []

	if not os.path.exists(pred_csv.replace('.csv', '.jpg')):
		for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
			detected_annotations = []
			scores_all_prediction.extend(pred_dict_score[image_name])
			if len(bbox) != 0:
				# pos image
				pred_scores = pred_dict_score[image_name]
				detections = pred_dict_box[image_name]
				annotations = np.array(bbox)
				for j, d in enumerate(detections):
					score = pred_scores[j]

					overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
					assigned_annotation = np.argmax(overlaps, axis=1)
					max_overlap = overlaps[0, assigned_annotation]

					if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
						detected_annotations.append(assigned_annotation)
						scores = np.append(scores, score)

			annotations_not_detected = len(bbox) - len(detected_annotations)
			for j in range(annotations_not_detected):
				scores = np.append(scores, 0)



		x_axis = [ 1. * x /  10 for x in range(0, 10)]
		x_axis = [str(x) for x in x_axis]


		num_record_between = [0 for x in range(10)]
		num_record_over = [0 for x in range(10)]

		for score in scores:
			for i in range(0, 10):
				j = 1. * i / 10
				j_ = 1. * (i + 1) / 10
				if score >= j and score < j_:
					num_record_between[i] += 1
				if score > j:
					num_record_over[i] += 1

		num_record_between_num = num_record_between

		num_record_between = [x / len(scores) for x in num_record_between]
		num_record_over = [x / len(scores) for x in num_record_over]

		x = np.arange(len(x_axis))

		plt.figure(figsize=(15, 15))

		plt.subplot(2,2,1)
		rects = plt.bar(x + 0.5, height=num_record_between, width=0.5)
		plt.title('between (gt)')
		for rect in rects:
			height = rect.get_height()
			value = height * 100
			value = str(value)[:2]
			if value[-1] == '.':
				value = value[0]
			value = value + '%'
			plt.text(rect.get_x() + rect.get_width() / 2, height, value, ha="center", va="bottom")

		plt.ylim(0, 1)

		x_axis_ = x_axis.copy()
		x_axis_.append(str(1.0))
		x_ = np.arange(len(x_axis))
		plt.xticks([index for index in x_], x_axis_)

		plt.subplot(2,2,2)
		rects = plt.bar(x + 0.5, height=num_record_over, width=0.5, color='#FF8006')
		plt.title('over')
		for rect in rects:
			height = rect.get_height()
			value = height * 100
			if value == 100:
				value = str(value)
			else:
				value = str(value)[:2]
				if value[-1] == '.':
					value = value[0]
			value = value + '%'
			plt.text(rect.get_x() + rect.get_width() / 2, height, value, ha="center", va="bottom")

		plt.ylim(0, 1)

		x_axis_ = x_axis.copy()
		x_axis_.append(str(1.0))
		x_ = np.arange(len(x_axis))
		plt.xticks([index for index in x_], x_axis_)



		plt.subplot(2, 2, 3)

		num_record_between_all = [0 for x in range(10)]
		for score in scores_all_prediction:
			for i in range(0, 10):
				j = 1. * i / 10
				j_ = 1. * (i + 1) / 10
				if score >= j and score < j_:
					num_record_between_all[i] += 1

		num_record_between_all_num = num_record_between_all
		num_record_between_all = [x / len(scores_all_prediction) for x in num_record_between_all]

		rects = plt.bar(x + 0.5, height=num_record_between_all, width=0.5, color='#F54545')
		plt.title('between (all prediction)')
		for rect in rects:
			height = rect.get_height()
			value = height * 100
			if value == 100:
				value = str(value)
			else:
				value = str(value)[:2]
				if value[-1] == '.':
					value = value[0]
			value = value + '%'
			plt.text(rect.get_x() + rect.get_width() / 2, height, value, ha="center", va="bottom")

		plt.ylim(0, 1)

		x_axis_ = x_axis.copy()
		x_axis_.append(str(1.0))
		x_ = np.arange(len(x_axis))
		plt.xticks([index for index in x_], x_axis_)




		plt.subplot(2, 2, 4)


		num_record_between_gt_over_all = [num_record_between_num[x] / num_record_between_all_num[x] for x in range(10)]

		rects = plt.bar(x + 0.5, height=num_record_between_gt_over_all, width=0.5, color='#45F545')
		plt.title('between (gt / all)')
		for rect in rects:
			height = rect.get_height()
			value = height * 100
			if value == 100:
				value = str(value)
			else:
				value = str(value)[:2]
				if value[-1] == '.':
					value = value[0]
			value = value + '%'
			plt.text(rect.get_x() + rect.get_width() / 2, height, value, ha="center", va="bottom")

		plt.ylim(0, 1)

		x_axis_ = x_axis.copy()
		x_axis_.append(str(1.0))
		x_ = np.arange(len(x_axis))
		plt.xticks([index for index in x_], x_axis_)

		plt.savefig(pred_csv.replace('.csv', '.jpg'))


