import argparse
import models
import numpy as np
import time
import cv2
import os
from PIL import Image
import torch
from tools.InferSharpMask import Infer
from utils.load_helper import load_pretrain

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='PyTorch DeepMask/SharpMask evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SharpMask', choices=model_names,
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


def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)


def set_model():
        # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    from collections import namedtuple

    #############
    if args.arch == 'DeepMask':
        Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
        # default for training (Deepmask)
        config = Config(iSz=160, oSz=56, gSz=112, batch=1)
    elif args.arch == 'SharpMask':
        Config = namedtuple(
            'Config', ['iSz', 'oSz', 'gSz', 'batch', 'km', 'ks'])
        config = Config(iSz=160, oSz=56, gSz=160, batch=1, km=32, ks=32)

    model = (models.__dict__[args.arch](config))
    model = load_pretrain(model, args.resume)
    model = model.eval().to(device)  # 使模型為TEST階段

    return model, device

@torch.no_grad()
def main():
    directory_img = args.img

    for file_img in os.listdir(directory_img):
        model, device = set_model()
        scales = [2**i for i in range_end(args.si, args.sf, args.ss)]
        meanstd = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        infer = Infer(nps=args.nps, scales=scales,
                  meanstd=meanstd, model=model, device=device)

        im = np.array(Image.open(directory_img + "/" +
                                 file_img).convert('RGB'), dtype=np.float32)
        h, w = im.shape[:2]
        img = np.expand_dims(np.transpose(im, (2, 0, 1)),
                             axis=0).astype(np.float32)
        img = torch.from_numpy(img / 255.).to(device)
        infer.forward(img)
        masks, scores = infer.getTopProps(.2, h, w)

        for i in range(masks.shape[2]):
            res = im[:, :, ::-1].copy().astype(np.uint8)
            res[:, :, 2] = masks[:, :, i] * 255 + \
                (1 - masks[:, :, i]) * res[:, :, 2]

            print('Segment Proposal Score: {:.3f}'.format(scores[i]))

            # res = cv2.rectangle(res, (predict_box[0], predict_box[1]),
            #             (predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]), (0, 255, 0), 3)
            #res = cv2.polylines(res, [np.int0(rbox)], True, (0, 255, 255), 3)
            cv2.imwrite('result_img/' + file_img, res)

            #cv2.imshow('Proposal', res)
            # cv2.waitKey(0)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    main()
