from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import torch
import cv2
from utils import get_box
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from torchvision import transforms
from PIL import Image
import random

def collate_fn(batch):
    '''
    In order to return bbox in batch, we add zero tensor if there is not enough bbox
    Example: if batch size if 4, and bbox size is [1, 6], [1, 6], [1, 6], [2, 6] seperately, then the returned bbox size
    will be [4(batch size), 2(max num bbox), 6(coordinate, class, instance)]
    :param batch:
    :return:
    '''
    image, image_uda, bbox, bbs_uda, image_, image_uda_, seq_reverse = zip(*batch)
    max_len = 0
    for box in bbox:
        if len(box) > max_len:
            max_len = len(box)

    if max_len > 0:
        bbox_padded = np.ones((len(bbox), max_len, 5)) * -1
        bbox_padded_uda = np.ones((len(bbox), max_len, 5)) * -1

        for index, box in enumerate(bbox):
            if box.shape[0] > 0:
                bbox_padded[index, :len(box), :4] = box
                bbox_padded[index, : len(box), 4] = 1
                bbox_padded_uda[index, :len(box), :4] = bbs_uda[index]
                bbox_padded_uda[index, :len(box), 4] = 1

        bbox_padded = torch.Tensor(bbox_padded)
        bbox_padded_uda = torch.Tensor(bbox_padded_uda)
    else:
        bbox_padded = torch.ones((len(bbox), 1, 5)) * -1
        bbox_padded_uda = torch.ones((len(bbox), 1, 5)) * -1

    image = torch.stack(image, 0)
    image_ = torch.stack(image_, 0)
    image_uda = torch.stack(image_uda, 0)
    image_uda_ = torch.stack(image_uda_, 0)

    return image, image_uda, bbox_padded, bbox_padded_uda, image_, image_uda_, list(seq_reverse)

def random_crop(image, bbox=None, crop_w=512, crop_h=512):
    '''
    :param image: numpy.ndarry -> [H, W, 3]
    :param bbox:  list -> [[x1, y1, x2, y2], ...]
    :return:
    '''
    h, w = image.shape[:2]
    xmin = np.random.randint(0, w - 1 - crop_w)
    ymin = np.random.randint(0, h - 1 - crop_h)
    xmax = xmin + crop_w
    ymax = ymin + crop_h
    image = image[ymin: ymax, xmin: xmax]
    if bbox is not None:
        bbox = np.array(bbox)
        # exclude bbox out of range
        bbox = bbox[bbox[:, 0] < xmax - 8]
        bbox = bbox[bbox[:, 1] < ymax - 8]
        bbox = bbox[bbox[:, 2] > xmin + 7]
        bbox = bbox[bbox[:, 3] > ymin + 7]

        bbox[:, 0] = np.clip(bbox[:, 0] - xmin, 0, crop_w - 1)
        bbox[:, 1] = np.clip(bbox[:, 1] - ymin, 0, crop_h - 1)
        bbox[:, 2] = np.clip(bbox[:, 2] - xmin, 0, crop_w - 1)
        bbox[:, 3] = np.clip(bbox[:, 3] - ymin, 0, crop_h - 1)

        return image, bbox
    else:
        return image

seq = iaa.SomeOf((3, 7), [
    # iaa.Affine(scale=(0.8, 1.2),
    #            rotate=(-10, 10)),
    # iaa.GaussianBlur((0, 1.0)),
    # iaa.Add((-20, 30)),
    # iaa.GammaContrast((0.8, 1.2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((0,3))
], random_order=True)

fliplr = iaa.Fliplr(1)
flipud = iaa.Flipud(1)
rot90 = iaa.Rot90(1)
rot180 = iaa.Rot90(2)
rot270 = iaa.Rot90(3)

da_list = [fliplr, flipud, rot180]
reverse_list = ['fliplr', 'flipud', 'rot180']


class Ring_Cell_random_crop(Dataset):
    def __init__(self, txt_path, training=True):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # remove image with no bbox
        lines = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))]
        # random crop 25 times every epoch
        self.lines = lines * 25
        self.training = training
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        image_path = self.lines[index][:-1]
        image = cv2.imread(image_path)

        xml_path = image_path.replace('jpeg', 'xml')
        bbox = get_box(xml_path)

        image, bbox = random_crop(image, bbox)

        if self.training:
            # data augmentation
            # seq_det = seq.to_deterministic()
            seq_idx = random.choice(np.arange(3))
            seq_det = da_list[seq_idx]
            bbs = []
            for box in bbox:
                bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
            bbs = ia.BoundingBoxesOnImage(bbs, shape=image.shape)

            image_uda = seq_det.augment_image(image)
            bbs_uda_ = seq_det.augment_bounding_boxes([bbs])
            bbs_uda = []
            for box in bbs_uda_[0].bounding_boxes:
                bbs_uda.append([box.x1, box.y1, box.x2, box.y2])

            bbs_uda = np.clip(bbs_uda, a_min=0, a_max=511)

            image_uda = self.to_tensor(Image.fromarray(image_uda))

        bbox = np.clip(bbox, a_min=0, a_max=511)


        image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = self.to_tensor(Image.fromarray(image))

        return image, bbox, image_

        # vis
        # for box in bbox:
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imwrite('vis/{}.jpg'.format(index), image)

    def __len__(self):
        return len(self.lines)



class Ring_Cell_Stride(Dataset):
    def __init__(self):
        pass



class Ring_Cell_all(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # remove image with no bbox
        self.lines = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))]

        # random crop 25 times every epoch
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        image_path = self.lines[index][:-1]
        image = cv2.imread(image_path)

        xml_path = image_path.replace('jpeg', 'xml')
        bbox = get_box(xml_path)

        image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = self.to_tensor(Image.fromarray(image))

        return image, bbox, image_

    def __len__(self):
        return len(self.lines)

class Ring_Cell_random_crop_all(Dataset):
    # include pos and neg samples
    # pos:neg = 1:5
    def __init__(self, txt_path, training=True, mixup=False):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # balance the pos and neg samples
        lines_pos = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 25
        lines_neg = [x for x in lines if not os.path.exists(x[:-1].replace('jpeg', 'xml'))] * 5
        # random crop 25 times every epoch
        self.lines = lines_pos
        self.lines.extend(lines_neg)
        self.training = training
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mixup = mixup

    def __getitem__(self, index):
        image_path = self.lines[index][:-1]
        image = cv2.imread(image_path)

        xml_path = image_path.replace('jpeg', 'xml')
        if os.path.exists(xml_path):
            bbox = get_box(xml_path)
            image, bbox = random_crop(image, bbox)
            bbox = bbox.tolist()
        else:
            image = random_crop(image)
            bbox = []

        if self.mixup == True:
            index_mixup = np.random.randint(0, len(self.lines))
            image_path_mixup = self.lines[index_mixup][:-1]
            image_mixup = cv2.imread(image_path_mixup)
            xml_path_mixup = image_path_mixup.replace('jpeg', 'xml')

            if os.path.exists(xml_path_mixup):
                bbox_mixup = get_box(xml_path_mixup)
                image_mixup, bbox_mixup = random_crop(image_mixup, bbox_mixup)
                bbox_mixup = bbox_mixup.tolist()
            else:
                image_mixup = random_crop(image_mixup)
                bbox_mixup = []

            ratio_mixup = np.random.beta(a=1.5, b=1.5)
            # ratio_mixup = 0.5
            image = image * ratio_mixup + image_mixup * (1 - ratio_mixup)
            bbox.extend(bbox_mixup)

        if self.training:
            # data augmentation
            # seq_det = seq.to_deterministic()
            seq_idx = random.choice(np.arange(3))
            seq_det = da_list[seq_idx]
            image_uda = seq_det.augment_image(image)
            seq_reverse = reverse_list[seq_idx]

            if os.path.exists(xml_path):
                bbs = []
                for box in bbox:
                    bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
                bbs = ia.BoundingBoxesOnImage(bbs, shape=image.shape)
                # bbs_ = seq_det.augment_bounding_boxes([bbs])
                bbs_uda_ = seq_det.augment_bounding_boxes([bbs])
                bbs_uda = []
                for box in bbs_uda_[0].bounding_boxes:
                    bbs_uda.append([box.x1, box.y1, box.x2, box.y2])

                bbs_uda = np.clip(bbs_uda, a_min=0, a_max=511)

            image_uda_ = torch.Tensor(np.ascontiguousarray(image_uda.transpose(2, 0, 1)))
            image_uda = self.to_tensor(Image.fromarray(image_uda))

        if os.path.exists(xml_path):
            bbox = np.clip(bbox, a_min=0, a_max=511)


        image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = self.to_tensor(Image.fromarray(image.astype(np.uint8)))


        if self.training == True:
            if os.path.exists(xml_path):
                return image, image_uda, bbox, bbs_uda, image_, image_uda_, seq_reverse
            else:
                return image, image_uda, np.array([]), np.array([]), image_, image_uda_, seq_reverse

        else:
            if os.path.exists(xml_path):
                return image, bbox, image_
            else:
                return image, np.array([]), image_

        # vis
        # for box in bbox:
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imwrite('vis/{}.jpg'.format(index), image)

    def __len__(self):
        return len(self.lines)

class Ring_Cell_all_dataset(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # remove image with no bbox
        # self.lines = [x for x in lines if os.path.exists(x[:-1].replace('jpeg', 'xml'))]
        self.lines = lines
        # random crop 25 times every epoch
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        image_path = self.lines[index][:-1]
        image = cv2.imread(image_path)
        image_name = image_path.split('/')[-1].split('.')[0]

        xml_path = image_path.replace('jpeg', 'xml')
        if os.path.exists(xml_path):
            bbox = get_box(xml_path)

        image_ = torch.Tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = self.to_tensor(Image.fromarray(image))

        if os.path.exists(xml_path):
            return image, bbox, image_, image_name
        else:
            return image, [], image_, image_name

    def __len__(self):
        return len(self.lines)

if __name__ == '__main__':

    root_dir = '/data/sqy/challenge/MICCAI2019/Signet_ring_cell_dataset/sig-train-pos'
    xml_path_list = glob(os.path.join(root_dir, '*.xml'))
    # print(xml_path_list)

    # dataset = Ring_Cell_random_crop_all('../train_test_4/train_0.txt', training=False)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     num_workers=1,
    #     collate_fn=collate_fn,
    #     shuffle=True
    # )
    # for i, (image, bbox, image_) in enumerate(dataloader):
    #     print(bbox)
        # for index in range(image_.size(0)):
        #     img = image_[index]
        #     boxs = bbox[index]
        #     img = np.array(img).transpose(1, 2, 0)
        #     boxs = np.array(boxs)
        #
        #     for idx, box in enumerate(boxs):
        #         img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #     cv2.imwrite('vis/{}_{}.jpg'.format(i, index), img)
    import shutil

    if os.path.isdir('dataset_test'):
        shutil.rmtree('dataset_test')
    os.mkdir('dataset_test')

    dataset = Ring_Cell_random_crop_all('../train_test_4/test_0.txt', mixup=False)
    for i, (image, image_uda, bbox, bbox_uda, image_, image_uda_, reverse) in enumerate(dataset):

        image_ = np.array(image_).transpose((1, 2, 0))
        for box in bbox:
            image_ = cv2.rectangle(image_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.imwrite(os.path.join('dataset_test', '{}_original.jpg'.format(i)), image_)

        image_uda_flip = torch.flip(image_uda_, (2,))
        image_uda_flipud = torch.flip(image_uda_, (1,))
        image_uda_rotate = torch.flip(image_uda_, (1, 2))
        image_uda_ = np.array(image_uda_).transpose((1, 2, 0))

        for box in bbox_uda:
            image_uda_ = cv2.rectangle(image_uda_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.imwrite(os.path.join('dataset_test', '{}_uda.jpg'.format(i)), image_uda_)

        image_uda_flip = np.array(image_uda_flip).transpose((1, 2, 0))
        image_uda_flip = np.array(image_uda_flip)

        image_uda_flipud = np.array(image_uda_flipud).transpose((1, 2, 0))
        image_uda_flipud = np.array(image_uda_flipud)

        image_uda_rotate = np.array(image_uda_rotate).transpose((1, 2, 0))
        image_uda_rotate = np.array(image_uda_rotate)


        cv2.imwrite(os.path.join('dataset_test', '{}_uda_flip.jpg'.format(i)), image_uda_flip)
        cv2.imwrite(os.path.join('dataset_test', '{}_uda_flipud.jpg'.format(i)), image_uda_flipud)
        cv2.imwrite(os.path.join('dataset_test', '{}_uda_rotate.jpg'.format(i)), image_uda_rotate)