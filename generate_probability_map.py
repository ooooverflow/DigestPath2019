import numpy as np
import os
import cv2

def random_crop_with_confidence_score(image, bboxs=None, crop_w=512, crop_h=512):
	h, w = image.shape[:2]
	probability_map = np.ones([h, w]).astype(np.float)
	# generate the probability map of the center point of the bbox
	for i, bbox in enumerate(bboxs):
		score = bbox[4]
		if score == 0:
			weight = 20
		else:
			weight = 1 / score
			if weight >= 20:
				weight = 20

		x1 = int(bbox[2]) - 253
		y1 = int(bbox[3]) - 253
		x2 = int(bbox[0]) + 253
		y2 = int(bbox[1]) + 253

		x1 = np.clip(x1, a_min=256, a_max=w - 256)
		y1 = np.clip(y1, a_min=256, a_max=h - 256)
		x2 = np.clip(x2, a_min=256, a_max=w - 256)
		y2 = np.clip(y2, a_min=256, a_max=h - 256)
		probability_map[int(y1): int(y2) + 1, int(x1): int(x2) + 1] += weight

	# probability_map_view = probability_map.reshape(-1)
	# probability_map = probability_map.reshape(-1)
	# probability_map_view = probability_map_view / np.sum(probability_map_view)
	probability_map = probability_map / np.sum(probability_map)
	return probability_map



def get_label(csv_path):
	pred_dict_box = {}
	with open(csv_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line[:-1]
			line = line.split(',')
			image_name = line[0]
			pred_dict_box[image_name] = []
			if len(line[1]) != 0:
				preds = line[1].split(';')[:-1]
				for pred in preds:
					pred = pred.split(' ')
					box = []
					for elemet in pred[:-1]:
						box.append(float(elemet))
					pred_dict_box[image_name].append(box)

	return pred_dict_box


txt_path='../train_test_4/train_0.txt',
confidence_csv_path='./bbox/retinanet_resnet18_training_data_with_confidence_score_using_test_data_prediction.csv'

label_dict = get_label(confidence_csv_path)

probability_map_dir = os.path.join('probability_map', 'from_round0_test_prediction_new')
if not os.path.isdir(probability_map_dir):
	os.mkdir(probability_map_dir)

for image_name in label_dict:
	image_path = os.path.join('/data/sqy/challenge/MICCAI2019/Signet_ring_cell_dataset/sig-train-pos', image_name+'.jpeg')
	print(image_path)
	if not os.path.exists(image_path):
		continue
	image = cv2.imread(image_path)
	bboxs = label_dict[image_name]
	probability_map = random_crop_with_confidence_score(image, bboxs)
	np.save(os.path.join(probability_map_dir, image_name), probability_map)
