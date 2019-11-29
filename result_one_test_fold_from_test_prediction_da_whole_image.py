from retinanet.dataset import Ring_Cell_all_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import model_all_dataset_weight_loss as model
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import cv2
import shutil
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.gpu_nms import gpu_nms
from lib_new.nms.nums_py import py_cpu_nms, py_cpu_nms_contain
from imgaug import augmenters as iaa
from metric import detection_metric, calculate_metric_final
import imgaug as ia

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

retinanet = model.resnet18(num_classes=2, pretrained=True)
retinanet = torch.nn.DataParallel(retinanet).cuda()

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


# this .py is used to generate the result or prediction of the model on test fold
# and combine this test fold to make new training data

da_list = ['original', 'rotate90', 'rotate180', 'rotate270', 'fliplr', 'flipud']

model_path = 'ckpt_new/best_valid_recall_fold_2_all_dataset_weight_loss_1_resnet18_round4_using_best_valid_recall_0.4_ensemble(round0)_0.4_ensemble(round1)_0.4_ensemble(round2)_0.4_multi3(round3)_1e-4_no_pretrain_remove_some_da.pth'

test_fold = model_path.split('_fold_')[1][0]

test_dataset = Ring_Cell_all_dataset('/data/sqy/code/miccai2019/train_test_4/test_{}.txt'.format(test_fold))


retinanet.module.load_state_dict(torch.load(model_path))

retinanet.eval()

image_size = 1024
stride_num = 1
score_threshold = 0.05
nms_threshold = 0.4

for da in da_list:
	# da = 'rotate270'
	result_dict = {}
	result_csv = './test_result_da/retinanet_resnet18_round4_fold_{}_weight_loss_1_on_test_data_best_valid_recall_{}_0.4(round0)_0.4(round1)_0.4(round2)_0.4_multi3(round3)_whole_image.csv'.format(test_fold, da)
	print(result_csv)
	if os.path.exists(result_csv):
		continue

	# if os.path.isdir(vis_dir):
	# 	shutil.rmtree(vis_dir)
	# os.mkdir(vis_dir)

	pred_boxes_total = []
	pred_scores_total = []
	gt_boxes_total = []

	font = cv2.FONT_HERSHEY_SIMPLEX


	for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
		h, w = image.size()[1:]
		# stride_h = (h - image_size) / (stride_num - 1)
		# stride_w = (w - image_size) / (stride_num - 1)

		if da == 'rotate90':
			seq = iaa.Rot90(1)
			image = np.array(image).transpose((1, 2, 0))
			image = seq.augment_image(image).transpose((2, 0, 1))
			image = image - np.zeros_like(image)
			image = torch.Tensor(image)
		elif da == 'rotate180':
			seq = iaa.Rot90(2)
			image = np.array(image).transpose((1, 2, 0))
			image = seq.augment_image(image).transpose((2, 0, 1))
			image = image - np.zeros_like(image)
			image = torch.Tensor(image)
		elif da == 'rotate270':
			seq = iaa.Rot90(3)
			image = np.array(image).transpose((1, 2, 0))
			image = seq.augment_image(image).transpose((2, 0, 1))
			image = image - np.zeros_like(image)
			image = torch.Tensor(image)
		elif da == 'fliplr':
			seq = iaa.Fliplr(1)
			image = np.array(image).transpose((1, 2, 0))
			image = seq.augment_image(image).transpose((2, 0, 1))
			image = image - np.zeros_like(image)
			image = torch.Tensor(image)
		elif da == 'flipud':
			seq = iaa.Flipud(1)
			image = np.array(image).transpose((1, 2, 0))
			image = seq.augment_image(image).transpose((2, 0, 1))
			image = image - np.zeros_like(image)
			image = torch.Tensor(image)

		pred_boxes = []
		pred_scores = []

		result_dict[image_name] = []

		image_patch = image
		# predict
		scores_patch, labels_patch, boxes_patch = retinanet(image_patch.unsqueeze(0).cuda().float(), score_threshold=score_threshold)
		scores_patch = scores_patch.cpu().detach().numpy()  # size -> [num_box]
		# labels_patch = la            bels_patch.cpu().detach().numpy()  # size -> [num_box]
		boxes_patch = boxes_patch.cpu().detach().numpy()  # size -> [num_box, 4]

		# change bbox coordinates

		if boxes_patch.shape[0] != 0:
			# start_x = int(w_index * stride_w)
			# start_y = int(h_index * stride_h)
			# box_index = (boxes_patch[:, 0] > 2) & (boxes_patch[:, 1] > 2) & (boxes_patch[:, 2] < image_size - 3)\
			# 			& (boxes_patch[:, 3] < image_size - 3) & (scores_patch > score_threshold)

			# boxes_patch = boxes_patch[box_index]
			# scores_patch = scores_patch[box_index]

			# boxes_patch[:, 0] = boxes_patch[:, 0] + start_x
			# boxes_patch[:, 1] = boxes_patch[:, 1] + start_y
			# boxes_patch[:, 2] = boxes_patch[:, 2] + start_x
			# boxes_patch[:, 3] = boxes_patch[:, 3] + start_y

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

			anchors_nms_idx = nms(torch.cat([pred_boxes, pred_scores], dim=2)[0, :, :], nms_threshold)

			pred_boxes = pred_boxes[:, anchors_nms_idx, :]
			pred_scores = pred_scores[:, anchors_nms_idx, :]

			# anchors_nms_idx = nms_contain(torch.cat([pred_boxes, pred_scores], dim=2)[0, :, :], 0.8)
			#
			# pred_boxes = pred_boxes[0, anchors_nms_idx, :]
			# pred_scores = pred_scores[0, anchors_nms_idx, 0]

			pred_boxes = pred_boxes[0, :, :]
			pred_scores = pred_scores[0, :, 0]

			pred_boxes = pred_boxes.numpy().tolist()
			pred_scores = pred_scores.numpy().tolist()

			bbs = []
			for box in pred_boxes:
				bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
			bbs = ia.BoundingBoxesOnImage(bbs, shape=(h, w, 3))

			if da == 'original':
				pred_boxes = []
				for box in bbs.bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])
			elif da == 'rotate90':
				seq = iaa.Rot90(3)
				bbs_ = seq.augment_bounding_boxes([bbs])
				pred_boxes = []
				for box in bbs_[0].bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])

			elif da == 'rotate180':
				seq = iaa.Rot90(2)
				bbs_ = seq.augment_bounding_boxes([bbs])
				pred_boxes = []
				for box in bbs_[0].bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])

			elif da == 'rotate270':
				seq = iaa.Rot90(1)
				bbs_ = seq.augment_bounding_boxes([bbs])
				pred_boxes = []
				for box in bbs_[0].bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])

			elif da == 'fliplr':
				seq = iaa.Fliplr(1)
				bbs_ = seq.augment_bounding_boxes([bbs])
				pred_boxes = []
				for box in bbs_[0].bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])

			elif da == 'flipud':
				seq = iaa.Flipud(1)
				bbs_ = seq.augment_bounding_boxes([bbs])
				pred_boxes = []
				for box in bbs_[0].bounding_boxes:
					pred_boxes.append([box.x1, box.y1, box.x2, box.y2])

			pred_boxes_total.append(pred_boxes)
			pred_scores_total.append(pred_scores)
			gt_boxes_total.append(bbox)

		else:
			pred_boxes_total.append([])
			pred_scores_total.append([])
			gt_boxes_total.append(bbox)


		for j, box in enumerate(pred_boxes):
			if float(pred_scores[j]) >=score_threshold:
				# image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
				# image = cv2.putText(image, str(float(pred_scores[j]))[:3], (int(box[0]) + 10, int(box[1]) + 20), font, 0.8, (0, 0, 0),
				# 					2)
				result_dict[image_name].append([box, pred_scores[j]])




		# for box in bbox:
		# 	image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

		# cv2.imwrite(os.path.join(vis_dir, 'train_{}_18_all_dataset_{}_latest.jpg'.format(i, score_threshold)), image)
	result_str = ''
	for image_name in result_dict:
		result_str += image_name
		result_str += ','
		results = result_dict[image_name]
		for result in results:
			box, score = result
			for element in box:
				result_str += str(element)
				result_str += ' '
			result_str += str(score)
			result_str += ';'
		result_str += '\n'

	with open(result_csv, 'w') as f:
		f.write(result_str)

# recall, precision, froc, FPs = calculate_metric_final(pred_boxes_total, gt_boxes_total, pred_scores_total, score_threshold=score_threshold)
#
# print('froc: {}, recall: {}, precision: {}, FPs: {}'.format(froc, recall[-1], precision[-1], FPs))






