from retinanet.dataset import Ring_Cell_all_dataset
from tqdm import tqdm
import numpy as np
import torch
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.gpu_nms import gpu_nms
from lib_new.nms.nums_py import py_cpu_nms

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	return py_cpu_nms(dets, thresh)

pred_dict_box = {}
pred_dict_score = {}
result_str = ''
round = 0

nms_threshold = 0.4
score_threshold = 0.05

for test_fold in range(0, 4):
	pred_csv = './test_result_new/retinanet_resnet18_round{}_test_fold_{}_best_valid_recall_weight_loss1.csv'.format(round, test_fold)

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



	for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
		result_dict[image_name] = []
		gt_bbox = bbox
		gt_scores = np.ones(len(gt_bbox)).tolist()
		if len(bbox) != 0:
			pred_scores = pred_dict_score[image_name]
			pred_bboxs = pred_dict_box[image_name]
			scores = gt_scores
			scores.extend(pred_scores)
			bboxs = gt_bbox
			bboxs.extend(pred_bboxs)

			# nms
			bboxs = torch.Tensor(bboxs).unsqueeze(0)  # size -> [1, num_box, 4]
			scores = torch.Tensor(scores).unsqueeze(0).unsqueeze(-1)  # size -> [1, num_box, 1]
			anchors_nms_idx = nms(torch.cat([bboxs, scores], dim=2)[0, :, :], nms_threshold)

			bboxs = bboxs[0, anchors_nms_idx, :]
			scores = scores[0, anchors_nms_idx, 0]

			bboxs = bboxs[scores >= score_threshold]
			scores = scores[scores >= score_threshold]

			bboxs = bboxs.numpy().tolist()
			scores = scores.numpy().tolist()

			result_dict[image_name].extend(bboxs)

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

result_csv = './bbox/retinanet_resnet_18_using_test_prediction_best_valid_recall_nms_{}_scorethreshold_{}_for_round{}_training.csv'\
										.format(nms_threshold, score_threshold, (round+1))
with open(result_csv, 'w') as f:
	f.write(result_str)