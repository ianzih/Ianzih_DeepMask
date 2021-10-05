import argparse
import models
import numpy as np
import time
import cv2
import os
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import torch
from tools.InferDeepMask import Infer
from utils.load_helper import load_pretrain
from mrcnnn import utils


model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch DeepMask/SharpMask evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMask)')
parser.add_argument('--resume', default='logs/checkpoint.pth.tar',
                    type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--img', default='TestImage/',
                    help='path/to/test/image')
#parser.add_argument('--img', default='data/testImage.jpg',help='path/to/test/image')
parser.add_argument('--nps', default=10, type=int,
                    help='number of proposals to save in test')
parser.add_argument('--si', default=-2.5, type=float, help='initial scale')
parser.add_argument('--sf', default=.5, type=float, help='final scale')
parser.add_argument('--ss', default=.5, type=float, help='scale step')


class ShapesDataset(utils.Dataset):
    # 得到該圖有多少物件
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # labelme中得到的yaml文件，從而得到mask每一層的標籤
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新寫draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新寫load_shapes，包含自己的類別
    # 在self.image_info中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        # Add classes
        self.add_class("shapes", 1, "tongue")

        for i in range(count):
            # 得圖片寬跟高
            #print(i)
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            #print(dataset_root_path + "labelme_json/" +filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path +
                                "labelme_json/" + filestr + "_json/img.png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重寫load_mask
    def load_mask(self, image_id):
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros(
            [info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))

        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []

        for i in range(len(labels)):
            if labels[i].find("tongue") != -1:
                labels_form.append("tongue")
            """
            elif labels[i].find("leg") != -1:
                 print "leg"
                labels_form.append("leg")
            """
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def load_image_gt(dataset, image_id):
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)

    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    return image, class_ids, mask 

def merged_mask(masks):
    """
    merge mask into one and return merged mask
    """
    n = masks.shape[2]

    if n != 0:
        merged_mask = np.zeros((masks.shape[0], masks.shape[1]))
        for i in range(n):
            merged_mask += masks[..., i]
        merged_mask = np.asarray(merged_mask, dtype=np.uint8)
        return merged_mask
    return masks[:, :, 0]

def compute_iou(predict_mask, gt_mask):
    """
    Computes Intersection over Union score for two binary masks.
    :param predict_mask: numpy array
    :param gt_mask: numpy array
    :type1 and type2 results are same
    :return iou score:
    """
    if predict_mask.shape[2] == 0:
        return 0
    mask1 = merged_mask(predict_mask)
    mask2 = merged_mask(gt_mask)

    #print(mask1)
    #print('_________')
    #print(mask2)
    # type 1
    """
    intersection = np.sum((mask1 + mask2) > 1)
    union = np.sum((mask1 + mask2) > 0)
    iou_score = intersection / float(union)
    print("Iou : ",iou_score)
    """
    # type2
    intersection = np.logical_and(mask1, mask2)  
    union = np.logical_or(mask1, mask2) 
    iou_score = np.sum(intersection) / np.sum(union)
    print("Iou : ", iou_score)
    
    return iou_score

def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)

def set_model():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup Model
    from collections import namedtuple

    if args.arch=='DeepMask':
        Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
        config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training (Deepmask)
    elif args.arch=='SharpMask':
        Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch', 'km', 'ks'])
        config = Config(iSz=160, oSz=56, gSz=160, batch=1, km=32, ks=32)

    model = (models.__dict__[args.arch](config))
    model = load_pretrain(model, args.resume)
    model = model.eval().to(device) #使模型為TEST階段

    return model , device

@torch.no_grad()
def main():
    # prepare Validation img dataset
    dataset_root_path = "./tongue_eval/"
    img_floder = dataset_root_path + "pic"
    mask_floder = dataset_root_path + "cv_mask"
    imglist = os.listdir(img_floder)
    count = len(imglist)

    dataset_val = ShapesDataset()
    dataset_val.load_shapes(184, img_floder, mask_floder,imglist, dataset_root_path)
    dataset_val.prepare()
    image_ids = dataset_val.image_ids
    IOUs = []

    for image_id in image_ids:
        
        model, device = set_model()
        scales = [2**i for i in range_end(args.si, args.sf, args.ss)]
        meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        infer = Infer(nps=args.nps, scales=scales, meanstd=meanstd, model=model, device=device)

        filestr = imglist[image_id].split(".")[0]
        print(filestr)
        # Load image and ground truth data
        image, gt_class_id, gt_mask  = load_image_gt(dataset_val,image_id)
        img_path = img_floder + '/' + imglist[image_id] 

        im = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        h, w = im.shape[:2]
        img = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
        img = torch.from_numpy(img / 255.).to(device)
        infer.forward(img)
        masks, scores = infer.getTopProps(.2, h, w)

        IOU = compute_iou(masks, gt_mask)
        IOUs.append(IOU)
        torch.cuda.empty_cache()

        
    print("mIOU: ", np.mean(IOUs))
    print("standard deviation : ", '{:.4f}'.format(np.std(IOUs, ddof=1)))   


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    main()
