from dataset_csv_with_confidence_score import collate_fn, Ring_Cell_random_crop_all
from dataset import Ring_Cell_all_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import model_all_dataset_weight_loss as model
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from metric import detection_metric, calculate_metric_final, calculate_metric_final_new
# from lib.nms.pth_nms import pth_nms
from lib_new.nms.nums_py import py_cpu_nms
import random

def seed_torch(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

# set seed for torch and numpy
seed_torch(0)

def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	# Accept dets as tensor"""
	# return pth_nms(dets, thresh)
	dets = dets.cpu().detach().numpy()
	return py_cpu_nms(dets, thresh)

def get_lr(optimizer):
	return optimizer.param_groups[0]['lr']

def main(params):

	if params['writer'] == True:
		writer = SummaryWriter(comment='_resnet18 4fold_{} baseline all dataset(pos+neg) training with confidence'.format(params['test_fold']))

	retinanet = model.resnet18(num_classes=2, pretrained=True)
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	if os.path.exists(params['model_path']) and params['resume']:
		retinanet.module.load_state_dict(torch.load(params['model_path']))
		print('resume training from {}'.format(params['model_path']))

	optimizer = torch.optim.Adam(retinanet.parameters(), lr=params['learning_rate'])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.9)

	train_dataset = Ring_Cell_random_crop_all(params['train_txt'],
				confidence_csv_path='./bbox/retinanet_resnet18_training_data_with_confidence_score_using_test_data_prediction.csv',
											  pm_dir='probability_map/from_round0_test_prediction_new')

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=params['batch_size'],
		num_workers=8,
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

		tq = tqdm(total=len(train_dataloader))
		lr = get_lr(optimizer)
		tq.set_description('epoch:{}, learning rate:{}'.format(epoch, lr))

		for index, (data, label, _) in enumerate(train_dataloader):
			optimizer.zero_grad()
			classification_loss, regression_loss = retinanet([data.cuda().float(), label])

			classification_loss = classification_loss.mean()
			regression_loss = regression_loss.mean()

			loss = classification_loss + regression_loss

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
				step += 1
				epoch_loss = []

		torch.save(retinanet.module.state_dict(), 'ckpt_new/latest_resnet18_fold_{}_all_dataset_weight_loss_{}_confidence.pth'.format(params['test_fold'], params['weight_loss']))

		tq.close()

		scheduler.step(np.mean(epoch_loss_train))

		if params['writer'] == True:
			writer.add_scalar('epoch/loss for train', float(np.mean(epoch_loss_train)), epoch)
			writer.add_scalar('epoch/cls loss for train', float(np.mean(epoch_cls_loss_train)), epoch)
			writer.add_scalar('epoch/reg loss for train', float(np.mean(epoch_reg_loss_train)), epoch)


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
						scores_patch, labels_patch, boxes_patch = retinanet(image_patch.unsqueeze(0).cuda().float())
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


			if float(recall_record[-1]) > best_recall_valid:
				best_recall_valid = float(recall_record[-1])
				torch.save(retinanet.module.state_dict(),
						   'ckpt_new/best_valid_recall_resnet18_fold_{}_all_dataset_weight_loss_{}_confidence.pth'.format(
							   params['test_fold'], params['weight_loss']))

			if float(recall[-1]) > best_recall:
				best_recall = float(recall[-1])
				torch.save(retinanet.module.state_dict(),
						   'ckpt_new/best_recall_resnet18_fold_{}_all_dataset_weight_loss_{}_confidence.pth'.format(
							   params['test_fold'], params['weight_loss']))



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
		'batch_size': 32,
		'writer': True,
		'model_path': 'ckpt/best_precision_resnet101.pth',
		'resume': False,
		'test_image_size': 1024,
		'test_stride_num': 3,
		'weight_loss': 1
	}


	main(params)