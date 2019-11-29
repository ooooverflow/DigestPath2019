import numpy as np

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_froc(true_positives, false_positives, recall, num_pic):
    for index in range(len(true_positives)):
        tp = true_positives[index]
        fp = false_positives[index]
        r = recall[index]


def detection_metric(pred_bboxes, gt_bboxes, pred_scores, iou_threshold=0.3, score_threshold=0.5):
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

    for i in range(len(pred_bboxes)):
        detections = pred_bboxes[i]
        annotations = np.array(gt_bboxes[i])
        num_annotations += len(annotations)
        detected_annotations = []

        for j, d in enumerate(detections):
            score = pred_scores[i][j]
            if score < score_threshold:
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

    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)

    return average_precision, recall, precision



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
				if score < score_threshold:
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
				if score < score_threshold:
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

	FPs = float(FPs / normal_regions)
	FPs = max(100 - FPs, 0)

	return recall, precision, froc, FPs


def calculate_metric_final_new(pred_bboxes, gt_bboxes, pred_scores, iou_threshold=0.3, score_threshold=0.5):
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
		recall = np.array([0])
	precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
	if len(precision) == 0:
		precision = np.array([0])

	# index where precision greater equal 0.2
	index_record = np.where(precision >= 0.2)[0]
	if index_record.shape[0] != 0:
		index_record = index_record[-1]
		# recall, precision, FPs when precision is 0.2
		recall_record = recall[:index_record+1]
		precision_record = precision[:index_record+1]
		score_record = scores[:index_record+1]
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

	if recall.shape[0] == 0:
		recall = np.append(recall, 0)
	if precision.shape[0] == 0:
		precision = np.append(precision, 0)
	if scores.shape[0] == 0:
		scores = np.append(scores, 0)


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