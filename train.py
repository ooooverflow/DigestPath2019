from dataset import Ring_Cell_random_crop, collate_fn, Ring_Cell_all
import torch
from torch.utils.data import Dataset, DataLoader
import model as model
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from metric import detection_metric
from lib.nms.pth_nms import pth_nms


def nms(dets, thresh):
	"Dispatch to either CPU or GPU NMS implementations.\
	Accept dets as tensor"""
	return pth_nms(dets, thresh)


def get_lr(optimizer):
	return optimizer.param_groups[0]['lr']

def main(params):

	if params['writer'] == True:
		writer = SummaryWriter(comment='_resnet18 3fold 0 baseline')

	retinanet = model.resnet18(num_classes=2, pretrained=True)
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	if os.path.exists(params['model_path']) and params['resume']:
		retinanet.module.load_state_dict(torch.load(params['model_path']))
		print('resume training from {}'.format(params['model_path']))

	optimizer = torch.optim.Adam(retinanet.parameters(), lr=params['learning_rate'])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.9)

	train_dataset = Ring_Cell_random_crop(params['train_txt'])
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=params['batch_size'],
		num_workers=4,
		collate_fn=collate_fn,
		shuffle=True
	)

	test_dataset = Ring_Cell_all(params['test_txt'])


	# train
	step = 1

	best_recall = 0
	best_precision = 0
	best_ap = 0

	for epoch in range(params['max_epoch']):

		retinanet.train()
		retinanet.module.freeze_bn()

		epoch_loss_train = []
		epoch_loss = []

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

			if index % 20 == 0 and params['writer'] == True:
				writer.add_scalar('loss for train', float(np.mean(epoch_loss)), step)
				step += 1
				epoch_loss = []

		torch.save(retinanet.module.state_dict(), 'ckpt/latest_resnet18_fold_0.pth')

		tq.close()

		scheduler.step(np.mean(epoch_loss_train))


		if params['writer'] == True:
			writer.add_scalar('epoch/loss for train', float(np.mean(epoch_loss_train)), epoch)

		# test
		retinanet.eval()
		test_image_size = params['test_image_size']
		stride_num = params['test_stride_num']

		pred_boxes_total = []
		pred_scores_total = []
		gt_boxes_total = []

		for i, (image, bbox, image_) in enumerate(tqdm(test_dataset)):
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

				anchors_nms_idx = nms(torch.cat([pred_boxes, pred_scores], dim=2)[0, :, :], 0.5)

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

		average_precision, recall, precision = detection_metric(pred_boxes_total, gt_boxes_total, pred_scores_total)

		if params['writer'] == True:

			writer.add_scalar('epoch/average precision', float(average_precision), epoch)
			writer.add_scalar('epoch/recall', float(recall[-1]), epoch)
			writer.add_scalar('epoch/precision', float(precision[-1]), epoch)

		if float(average_precision) > best_ap:
			best_ap = float(average_precision)
			torch.save(retinanet.module.state_dict(), 'ckpt/best_ap_resnet18_fold_0.pth')

		if float(recall[-1]) > best_recall:
			best_recall = float(recall[-1])
			torch.save(retinanet.module.state_dict(), 'ckpt/best_recall_resnet18_fold_0.pth')

		if float(precision[-1]) > best_precision:
			best_precision = float(precision[-1])
			torch.save(retinanet.module.state_dict(), 'ckpt/best_precision_resnet18_fold_0.pth')



		print('ap: {}, recall: {}, precision: {}'.format(average_precision, recall[-1], precision[-1]))


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '7, 6'
	params = {
		'learning_rate': 1e-4,
		'optim': 'adam',
		'max_epoch': 600,
		'train_txt': '../train_test_3/train_0.txt',
		'test_txt': '../train_test_3/test_0.txt',
		'batch_size': 8,
		'writer': True,
		'model_path': 'ckpt/best_precision_resnet101.pth',
		'resume': False,
		'test_image_size': 1024,
		'test_stride_num': 3
	}


	main(params)