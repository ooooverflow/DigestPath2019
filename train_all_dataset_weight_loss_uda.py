from dataset_uda import Ring_Cell_random_crop, collate_fn, Ring_Cell_all_dataset, Ring_Cell_random_crop_all
import torch
from torch.utils.data import Dataset, DataLoader
import model_all_dataset_uda as model
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from metric import detection_metric, calculate_metric_final, calculate_metric_final_new
from lib_new.nms.gpu_nms import gpu_nms
import random
from lib_new.nms.nums_py import py_cpu_nms

def seed_torch(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

# set seed for torch and numpy
seed_torch(2)

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	dets = dets.cpu().detach().numpy()
	# return gpu_nms(dets, thresh)
	return py_cpu_nms(dets, thresh)

def get_lr(optimizer):
	return optimizer.param_groups[0]['lr']

def main(params):
	print('fold {}'.format(params['test_fold']))
	description = 'resnet18_remove_some_da_uda_l1_loss_top2000'

	if params['writer'] == True:
		writer = SummaryWriter(comment='_4fold_{}all dataset(pos+neg)_weight_loss_{}_{}'.format(params['test_fold'], params['weight_loss'], description))

	retinanet = model.resnet18(num_classes=2, pretrained=True)
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	if os.path.exists(params['model_path']) and params['resume']:
		retinanet.module.load_state_dict(torch.load(params['model_path']))
		print('resume training from {}'.format(params['model_path']))

	optimizer = torch.optim.Adam(retinanet.parameters(), lr=params['learning_rate'])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.9)

	train_dataset = Ring_Cell_random_crop_all(params['train_txt'], mixup=False)
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=params['batch_size'],
		num_workers=4,
		collate_fn=collate_fn,
		shuffle=True
	)

	test_dataset = Ring_Cell_all_dataset(params['test_txt'])


	# train
	step = 1

	best_recall = 0
	best_recall_valid = 0
	best_precision = 0
	best_ap = 0
	best_froc = 0
	best_fps = 0

	for epoch in range(params['max_epoch']):

		retinanet.train()
		retinanet.module.freeze_bn()

		epoch_loss_train = []
		epoch_loss = []
		epoch_cls_loss_train = []
		epoch_reg_loss_train = []

		epoch_l1_cls_loss_train = []
		epoch_l1_reg_loss_train = []

		tq = tqdm(total=len(train_dataloader))
		lr = get_lr(optimizer)
		tq.set_description('epoch:{}, learning rate:{}'.format(epoch, lr))

		for index, (image, image_uda, bbox, bbox_uda, image_, image_uda_, reverse) in enumerate(train_dataloader):
			optimizer.zero_grad()
			(classification_loss, regression_loss), classification, regression = retinanet([image.cuda().float(), bbox])
			(classification_loss_uda, regression_loss_uda), classification_uda, regression_uda = retinanet([image_uda.cuda().float(), bbox_uda], reverse=reverse)

			classification_uda = classification_uda[:, :, 1].view(-1)
			classification = classification[:, :, 1].view(-1)

			cls_idx = (classification > 0.05) | (classification_uda > 0.05)

			classification_uda = classification_uda[cls_idx]
			classification = classification[cls_idx]

			if classification.size(0) == 0:
				l1_loss_cls = 0
			else:
				l1_loss_cls = torch.nn.SmoothL1Loss()(classification_uda, classification)
				# l1_loss_cls = torch.nn.SmoothL1Loss()(classification_uda, torch.zeros_like(classification))
			print(l1_loss_cls)
			# l1_loss_cls_batch = torch.nn.KLDivLoss(reduction='none')(torch.nn.LogSoftmax()(classification_uda), torch.nn.Softmax()(classification))
			# l1_loss_reg_batch = torch.nn.L1Loss(reduction='none')(regression_uda, regression).sum(-1)
			#
			# _, idx = torch.sort(-l1_loss_cls_batch)
			# idx = idx[:, :200]
			# l1_loss_cls = 0
			# for b in range(image.shape[0]):
			# 	l1_loss_cls += l1_loss_cls_batch[b][idx[b]].mean()
			#
			#
			# _, idx = torch.sort(-l1_loss_reg_batch)
			# idx = idx[:, :200]
			# l1_loss_reg = 0
			# for b in range(image.shape[0]):
			# 	l1_loss_reg += l1_loss_reg_batch[b][idx[b]].mean()


			epoch_l1_cls_loss_train.append(float(l1_loss_cls))
			# epoch_l1_reg_loss_train.append(float(l1_loss_reg))

			# l1_loss_reg = torch.nn.L1Loss()(regression_uda, regression)

			classification_loss = classification_loss.mean()
			regression_loss = regression_loss.mean()
			classification_loss_uda = classification_loss_uda.mean()
			regression_loss_uda = regression_loss_uda.mean()

			# print(l1_loss_cls, l1_loss_reg, classification_loss, classification_loss_uda, regression_loss, regression_loss_uda)
			loss = (classification_loss + regression_loss + classification_loss_uda + regression_loss_uda) / 2 + (l1_loss_cls)

			if bool(loss == 0):
				continue

			loss.backward()

			tq.update(1)

			torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

			optimizer.step()

			epoch_loss_train.append(float(loss))
			epoch_loss.append(float(loss))
			epoch_cls_loss_train.append(float(classification_loss))
			epoch_reg_loss_train.append(float(regression_loss))

			if index % 20 == 0 and params['writer'] == True:
				writer.add_scalar('loss for train', float(np.mean(epoch_loss)), step)
				# writer.add_scalar('l1_loss_cls', float)
				step += 1
				epoch_loss = []

		if params['save_model']:
			torch.save(retinanet.module.state_dict(), 'ckpt_new/latest_fold_{}_all_dataset_weight_loss_{}_{}.pth'.format(params['test_fold'], params['weight_loss'], description))

		# if epoch == 50:
		# 	torch.save(retinanet.module.state_dict(),
		# 			   'ckpt_new/50_epoch_fold_{}_all_dataset_{}.pth'.format(
		# 				   params['test_fold'], params['weight_loss'], description))

		tq.close()

		scheduler.step(np.mean(epoch_loss_train))


		if params['writer'] == True:
			writer.add_scalar('epoch/loss for train', float(np.mean(epoch_loss_train)), epoch)
			writer.add_scalar('epoch/cls loss for train', float(np.mean(epoch_cls_loss_train)), epoch)
			writer.add_scalar('epoch/reg loss for train', float(np.mean(epoch_reg_loss_train)), epoch)
			writer.add_scalar('epoch/l1 cls loss for train', float(np.mean(epoch_l1_cls_loss_train)), epoch)
			writer.add_scalar('epoch/l1 reg loss for train', float(np.mean(epoch_l1_reg_loss_train)), epoch)

		# test
		retinanet.eval()
		with torch.no_grad():
			test_image_size = params['test_image_size']
			stride_num = params['test_stride_num']

			pred_boxes_total = []
			pred_scores_total = []
			gt_boxes_total = []

			for i, (image, bbox, image_, image_name) in enumerate(tqdm(test_dataset)):
				h, w = image.size()[1:]
				stride_h = (h - test_image_size) / (stride_num - 1)
				stride_w = (w - test_image_size) / (stride_num - 1)

				pred_boxes = []
				pred_scores = []

				for h_index in range(stride_num):
					for w_index in range(stride_num):
						image_patch = image[:, int(h_index * stride_h): int(h_index * stride_h) + test_image_size,
									  int(w_index * stride_w): int(w_index * stride_w) + test_image_size]
						# predict
						scores_patch, labels_patch, boxes_patch = retinanet(image_patch.unsqueeze(0).cuda().float(), score_threshold=0.05)
						scores_patch = scores_patch.cpu().detach().numpy()  # size -> [num_box]
						# labels_patch = labels_patch.cpu().detach().numpy()  # size -> [num_box]
						boxes_patch = boxes_patch.cpu().detach().numpy()  # size -> [num_box, 4]

						# change bbox coordinates

						if boxes_patch.shape[0] != 0:
							start_x = int(w_index * stride_w)
							start_y = int(h_index * stride_h)
							box_index = (boxes_patch[:, 0] > 5) & (boxes_patch[:, 1] > 5) & (
										boxes_patch[:, 2] < test_image_size - 6) \
										& (boxes_patch[:, 3] < test_image_size - 6)

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

				# image = image_.permute(1, 2, 0).numpy()
				# for box in pred_boxes:
				#     image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

				# nms
				if len(pred_boxes) != 0:
					pred_boxes = torch.Tensor(pred_boxes).unsqueeze(0)  # size -> [1, num_box, 4]
					pred_scores = torch.Tensor(pred_scores).unsqueeze(0).unsqueeze(-1)  # size -> [1, num_box, 1]

					# pred_boxes_w = pred_boxes[0, :, 2] - pred_boxes[0, :, 0]
					# pred_boxes_h = pred_boxes[0, :, 3] - pred_boxes[0, :, 1]

					# wh_idx = (pred_boxes_w > 15) & (pred_boxes_h > 15)
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

			recall, precision, froc, FPs, recall_record, precision_record, froc_record, FPs_record, score_record = \
				calculate_metric_final_new(pred_boxes_total, gt_boxes_total, pred_scores_total, score_threshold=0.2)

			if params['writer'] == True:
				writer.add_scalar('epoch/average froc', float(froc), epoch)
				writer.add_scalar('epoch/recall', float(recall[-1]), epoch)
				writer.add_scalar('epoch/precision', float(precision[-1]), epoch)
				writer.add_scalar('epoch/fps', float(FPs), epoch)
				writer.add_scalar('epoch/max valid recall', float(recall_record[-1]), epoch)
				writer.add_scalar('epoch/precision when max valid recall', float(precision_record[-1]), epoch)
				writer.add_scalar('epoch/score when max valid recall', float(score_record[-1]), epoch)
				writer.add_scalar('epoch/FPs when max valid recall', float(FPs_record), epoch)

			# if float(froc) > best_froc:
			# 	best_froc = float(froc)
			# 	torch.save(retinanet.module.state_dict(), 'ckpt/best_froc_resnet18_fold_{}_all_dataset.pth'.format(params['test_fold']))

			if float(recall_record[-1]) > best_recall_valid and params['save_model']:
				best_recall_valid = float(recall_record[-1])
				torch.save(retinanet.module.state_dict(),
						   'ckpt_new/best_valid_recall_fold_{}_all_dataset_weight_loss_{}_{}.pth'.format(params['test_fold'], params['weight_loss'], description))

			if float(recall[-1]) > best_recall and params['save_model']:
				best_recall = float(recall[-1])
				torch.save(retinanet.module.state_dict(),
						   'ckpt_new/best_recall_fold_{}_all_dataset_weight_loss_{}_{}.pth'.format(params['test_fold'], params['weight_loss'], description))

			# if float(precision[-1]) > best_precision:
			# 	best_precision = float(precision[-1])
			# 	torch.save(retinanet.module.state_dict(), 'ckpt/best_precision_resnet18_fold_{}_all_dataset.pth'.format(params['test_fold']))
			#
			# if float(FPs) > best_fps:
			# 	best_fps = float(FPs)
			# 	torch.save(retinanet.module.state_dict(), 'ckpt/best_fps_resnet18_fold_{}_all_dataset.pth'.format(params['test_fold']))



			print('froc: {}, recall: {}, precision: {}, fps: {}, best valid recall: {}'.format(froc, recall[-1], precision[-1], FPs, recall_record[-1]))

	input()

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	test_fold = 0
	params = {
		'learning_rate': 1e-4,
		'optim': 'adam',
		'max_epoch': 300,
		'test_fold': test_fold,
		'train_txt': '../train_test_4/train_{}.txt'.format(test_fold),
		'test_txt': '../train_test_4/test_{}.txt'.format(test_fold),
		'batch_size': 16,
		'writer': False,
		'model_path': 'ckpt/best_precision_resnet101.pth',
		'resume': False,
		'test_image_size': 1024,
		'test_stride_num': 3,
		'weight_loss': 1,
		'save_model': False
	}


	main(params)