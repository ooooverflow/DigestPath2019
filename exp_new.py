from retinanet.dataset import Ring_Cell_all_dataset
from tqdm import tqdm
import torch

# x = torch.Tensor([0.01,0.05,0.05,0.05,0.05])
# y = torch.Tensor([0.05,0.05,0.05,0.05,0.05])
# x = torch.nn.LogSoftmax(dim=-1)(x)
# y = torch.nn.Softmax(dim=-1)(y)
# # x = torch.log(x)
# print(x)
# print(y)
# z = torch.nn.KLDivLoss(reduction='none')(x, y)
# print(z)

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
import glob
import os
import shutil

ct_dir = 'test_result_da_ct'

ct_path_list = glob.glob(os.path.join(ct_dir, 'retinanet_resnet18_round1_train_on_fold_0_1_result_on_fold_0_0_weight_loss_1_latest_*.csv'))

for ct_path in ct_path_list:
    new_ct_path = ct_path.replace('.csv', '_0.5(round0).csv')
    shutil.move(ct_path, new_ct_path)

