# Ianzih_DeepMask

This code from [foolwood/deepmask-pytorch](https://github.com/foolwood/deepmask-pytorch)

I made some revise.

## Start
```bash
DEEPMASK=$PWD
export PYTHONPATH=$DEEPMASK:$PYTHONPATH
mkdir -p ./pretrained/deepmask   #put your pre-trained models
```

## Testing Visualize Mask
```bash
python tools/computeProposals.py --arch DeepMask --resume $DEEPMASK/pretrained/deepmask/DeepMask.pth.tar --img ./data/test.jpg
#--resume your model , --img test img
```

## training
1. Prepare 
    ```bash
    mkdir -p $DEEPMASK/data/coco ; cd $DEEPMASK/data/coco
    mkdir train ;  mkdir val ; #train dir. put your train_img , val dir. put your val_img
    mkdir annotations ; #put your train & val .json
    ```
2. traing
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --dataset coco -j 20 --freeze_bn
    ```
