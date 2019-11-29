from retinanet.dataset import Ring_Cell_all_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import model_all_dataset as model
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import cv2
import shutil
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.gpu_nms import gpu_nms
from metric import detection_metric, calculate_metric_final
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

retinanet = model.resnet101(num_classes=2, pretrained=True)
retinanet = torch.nn.DataParallel(retinanet).cuda()

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	return gpu_nms(dets, thresh)


# return whole image once
test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/test_0.txt')

# description = 'resnet152_no_neg'
model_path = 'ckpt/latest_resnet101_test_fold_0_all_dataset_round_1;nms_0.4;scores_threshold_0.2.pth'
retinanet.module.load_state_dict(torch.load(model_path))

retinanet.eval()

image_size = 1024
stride_num = 3
score_threshold = 0.2

vis_dir = './vis_1024_new'
#
# if os.path.isdir(vis_dir):
# 	shutil.rmtree(vis_dir)
# os.mkdir(vis_dir)

pred_boxes_total = []
pred_scores_total = []
gt_boxes_total = []

font = cv2.FONT_HERSHEY_SIMPLEX

for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
	h, w = image.size()[1:]
	stride_h = (h - image_size) / (stride_num - 1)
	stride_w = (w - image_size) / (stride_num - 1)

	pred_boxes = []
	pred_scores = []

	for h_index in range(stride_num):
		for w_index in range(stride_num):
			image_patch = image[:, int(h_index * stride_h) : int(h_index * stride_h) + image_size,
						  int(w_index * stride_w): int(w_index * stride_w) + image_size]
			# predict
			scores_patch, labels_patch, boxes_patch = retinanet(image_patch.unsqueeze(0).cuda().float())
			scores_patch = scores_patch.cpu().detach().numpy()  # size -> [num_box]
			# labels_patch = la            bels_patch.cpu().detach().numpy()  # size -> [num_box]
			boxes_patch = boxes_patch.cpu().detach().numpy()  # size -> [num_box, 4]

			# change bbox coordinates

			if boxes_patch.shape[0] != 0:
				start_x = int(w_index * stride_w)
				start_y = int(h_index * stride_h)
				box_index = (boxes_patch[:, 0] > 5) & (boxes_patch[:, 1] > 5) & (boxes_patch[:, 2] < image_size - 6)\
							& (boxes_patch[:, 3] < image_size - 6) & (scores_patch > score_threshold)

				boxes_patch = boxes_patch[box_index]
				scores_patch = scores_patch[box_index]

				boxes_patch[:, 0] = boxes_patch[:, 0] + start_x
				boxes_patch[:, 1] = boxes_patch[:, 1] + start_y
				boxes_patch[:, 2] = boxes_patch[:, 2] + start_x
				boxes_patch[:, 3] = boxes_patch[:, 3] + start_y

				boxes_patch = boxes_patch.tolist()
				scores_patch = scores_patch.tolist()

				pred_boxes.extend(boxes_patch)
				pred_scores.extend(scores_patch)


	image = image_.permute(1, 2, 0).numpy()
	# for box in pred_boxes:
	#     image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

	# nms

	if len(pred_boxes) != 0:
		pred_boxes = torch.Tensor(pred_boxes).unsqueeze(0)  # size -> [1, num_box, 4]
		pred_scores = torch.Tensor(pred_scores).unsqueeze(0).unsqueeze(-1)  # size -> [1, num_box, 1]

		pred_boxes_w = pred_boxes[0, :, 2] - pred_boxes[0, :, 0]
		pred_boxes_h = pred_boxes[0, :, 3] - pred_boxes[0, :, 1]


		# wh_idx = (pred_boxes_w > 10) & (pred_boxes_h > 10)
		# pred_boxes = pred_boxes[:, wh_idx, :]
		# pred_scores = pred_scores[:, wh_idx, :]

		anchors_nms_idx = nms(torch.cat([pred_boxes, pred_scores], dim=2)[0, :, :], 0.4)

		pred_boxes = pred_boxes[0, anchors_nms_idx, :]
		pred_scores = pred_scores[0, anchors_nms_idx, 0]

		pred_boxes = pred_boxes.numpy().tolist()
		pred_scores = pred_scores.numpy().tolist()

		pred_boxes_total.append(pred_boxes)
		pred_scores_total.append(pred_scores)
		gt_boxes_total.append(bbox)

	else:
		pred_boxes_total.append([])
		pred_scores_total.append([])
		gt_boxes_total.append(bbox)


	for j, box in enumerate(pred_boxes):
		if float(pred_scores[j]) >=score_threshold:
			image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
			image = cv2.putText(image, str(float(pred_scores[j]))[:3], (int(box[0]) + 10, int(box[1]) + 20), font, 0.8, (0, 0, 0),
								2)

	for box in bbox:
		image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

	cv2.imwrite(os.path.join(vis_dir, 'test_{}_101_all_dataset_{}_latest_round1.jpg'.format(i, score_threshold)), image)


recall, precision, froc, FPs = calculate_metric_final(pred_boxes_total, gt_boxes_total, pred_scores_total, score_threshold=score_threshold)

print('froc: {}, recall: {}, precision: {}, FPs: {}'.format(froc, recall[-1], precision[-1], FPs))






