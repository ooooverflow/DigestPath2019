# with open('./bbox/retinanet_resnet18_training_data_with_confidence_score_of_whole_image.csv', 'r') as f:
# 	lines = f.readlines()
#
# cnt = 0
# for line in lines:
# 	if line[-1] == '\n':
# 		line = line[:-1]
# 	line = line.split(',')
# 	image_name = line[0]
# 	info = line[1]
# 	info = info.split(';')
# 	if info[0] == '-1':
# 		continue
# 	cnt +=1
# 	print(image_name, info[-1])

