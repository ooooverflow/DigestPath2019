
with open('/data/sqy/code/miccai2019/train_test_4/train_0.txt', 'r') as f:
	lines = f.readlines()
	pos_list = []
	neg_list = []

	for line in lines:
		label = line.split('/')[-2]
		if label == 'sig-train-neg':
			neg_list.append(line)
		else:
			pos_list.append(line)

	half_pos_num = int(len(pos_list) / 2)
	half_neg_num = int(len(neg_list) / 2)

	train1_list = pos_list[:half_pos_num]
	train1_list.extend(neg_list[:half_neg_num])

	train2_list = pos_list[half_pos_num:]
	train2_list.extend(neg_list[half_neg_num:])

	with open('/data/sqy/code/miccai2019/train_test_4/train_0_0.txt', 'w') as f:
		input_str = ''
		for i in train1_list:
			input_str += i
		f.write(input_str)

	with open('/data/sqy/code/miccai2019/train_test_4/train_0_1.txt', 'w') as f:
		input_str = ''
		for i in train2_list:
			input_str += i
		f.write(input_str)
