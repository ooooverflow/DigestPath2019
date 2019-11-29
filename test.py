import torch
from dataset import NIH
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from torch.utils.data import DataLoader
from model import resnet50
import torch.optim as optim
import numpy as np
import tqdm

def cal_iou(a, b):
    if type(a) == torch.Tensor:
        a = a.squeeze()
    if type(b) == torch.Tensor:
        b = b.squeeze()
    ax1, ay1, az1, ax2, ay2, az2 = a[0], a[1], a[2], a[3], a[4], a[5]
    bx1, by1, bz1, bx2, by2, bz2 = b[0], b[1], b[2], b[3], b[4], b[5]
    x1 = max(ax1, bx1)
    x2 = min(ax2, bx2)
    y1 = max(ay1, by1)
    y2 = min(ay2, by2)
    z1 = max(az1, bz1)
    z2 = min(az2, bz2)
    interS = (x2 - x1) * (y2 - y1) * (z2 - z1)
    if interS < 0:
        interS = 0
    if interS < 0:
        interS = 0
    union = (az2 - az1) * (ay2 - ay1) * (ax2 - ax1) + (bz2 - bz1) * (by2 - by1) * (bx2 - bx1) - interS
    return (float(interS) / float(union))


model = resnet50(2)
model = torch.nn.DataParallel(model).cuda()

model_path = './checkpoints_128x128x64_new/77_epoch.pth'
model.module.load_state_dict(torch.load(model_path))

test_path = '/home/sqy/NIH_data_128x128x64_16x16x8/test'
dataset = NIH(test_path)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4
)


model.eval()
iou_rec = []
tq = tqdm.tqdm(total=len(dataloader))
tq.set_description('test')
for step, (data, label) in enumerate(dataloader):
    data = data.cuda().float()
    label = label[0][0][:-1]
    scores, classification, predict_bbox = model(data)

    if len(predict_bbox) == 0:
        iou_rec.append(0)
    else:
        predict_bbox = predict_bbox[0]
        x1, y1, z1, x2, y2, z2 = int(predict_bbox[0]), int(predict_bbox[1]), int(predict_bbox[2]), int(predict_bbox[3]), int(predict_bbox[4]), int(predict_bbox[5])
        predict = [x1, y1, z1, x2, y2, z2]
        iou = cal_iou(label, predict)
        iou_rec.append(iou)
    tq.update(1)
tq.close()
iou_average = np.mean(iou_rec)

print('average iou for test : %f' % iou_average)