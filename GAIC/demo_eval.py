from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset
import os
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import time
import numpy as np
import torch.multiprocessing

from config_cropping import cfg
from torchvision.transforms import ToTensor
from CACNet import CACNet
torch.multiprocessing.set_sharing_strategy('file_system')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Grid anchor based image cropping With Pytorch')
parser.add_argument('--input_dir', default='dataset/GAIC/images/test2',
                    help='root directory path of testing images')
parser.add_argument('--output_dir', default='dataset/test_result',
                    help='root directory path of testing images')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--net_path', default='weights/ablation/cropping/shufflenetv2/downsample4_multi_Aug1_Align9_Cdim8/80_0.595_0.570_0.525_0.487_0.750_0.720_0.698_0.684_0.776_0.799.pth',
                    help='Directory for saving checkpoint models')
# parser.add_argument('--net_path', default='weights/ablation/cropping/mobilenetv2/downsample4_multi_Aug1_Align9_Cdim8/73_0.515_0.480_0.463_0.446_0.670_0.655_0.648_0.631_0.734_0.759.pth',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--net_path', default='pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--net_path', default='weights/ablation/cropping/vgg16/downsample4_multi_Aug1_Align9_Cdim8/79_0.370_0.393_0.388_0.364_0.560_0.575_0.562_0.549_0.666_0.701.pth',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--net_path', default='weights/ablation/cropping/resnet50/downsample4_multi_Aug1_Align9_Cdim8/77_0.480_0.450_0.443_0.429_0.630_0.623_0.623_0.620_0.710_0.734.pth',
#                     help='Directory for saving checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset = setup_test_dataset(dataset_dir = args.input_dir)

def naive_collate(batch):
    return batch[0]

def output_file_name(input_path, idx):
    name = os.path.basename(input_path)
    segs = name.split('.')
    assert len(segs) >= 2
    return '%scrop_%d.%s'%('.'.join(segs[:-1]), idx, segs[-1])

def test():
    for epoch in range(0,1):

        net = build_crop_model(scale='multi',#scale='single',
                                alignsize=9, reddim=8, loadweight=False, model='shufflenetv2', downsample=4)

        # net = build_crop_model(scale='multi',#scale='single',
        #                        alignsize=9, reddim=8, loadweight=False, model='mobilenetv2',downsample=4)

        # net = build_crop_model(scale='multi',#scale='single',
        #                        alignsize=9, reddim=8, loadweight=False, model='vgg16', downsample=4)

        # net = build_crop_model(scale='multi',#scale='single',
        #                        alignsize=9, reddim=8, loadweight=False, model='resnet50', downsample=4)

        net.load_state_dict(torch.load(args.net_path))
        net.eval()

        if args.cuda:
            net = torch.nn.DataParallel(net,device_ids=[0])
            cudnn.benchmark = True
            net = net.cuda()

        data_loader = data.DataLoader(dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      collate_fn=naive_collate,
                                      shuffle=False)

        for id, sample in enumerate(data_loader):
            imgpath = sample['imgpath']
            image = sample['image']
            bboxes = sample['sourceboxes']
            resized_image = sample['resized_image']
            tbboxes = sample['tbboxes']

            if len(tbboxes['xmin'])==0:
                continue

            roi = []

            for idx in range(0,len(tbboxes['xmin'])):
                roi.append((0, tbboxes['xmin'][idx],tbboxes['ymin'][idx],tbboxes['xmax'][idx],tbboxes['ymax'][idx]))           

            resized_image = torch.unsqueeze(torch.as_tensor(resized_image), 0)
            if args.cuda:
                resized_image = Variable(resized_image.cuda())
                roi = Variable(torch.Tensor(roi))
            else:
                resized_image = Variable(resized_image)
                roi = Variable(torch.Tensor(roi))

            t0 = time.time()
            out = net(resized_image,roi)
            t1 = time.time()
            print('timer: %.4f sec.' % (t1 - t0))
            id_out = sorted(range(len(out)), key=lambda k: out[k], reverse = True)

            kcm_out=np.array([])
            kcm_resized=np.load('/home/jys/CACNet-Pytorch-main/211144.npy')
            for idx in range(0, len(tbboxes['xmin'])):

                best_crop = bboxes[idx]
                best_crop = [int(x) for x in best_crop]


                kcm_best_crop = kcm_resized[best_crop[0]: best_crop[2], best_crop[1]: best_crop[3]]

                crop_value = kcm_best_crop.sum()
                avg_kcm=crop_value/(best_crop[2]- best_crop[0])/(best_crop[3]- best_crop[1])
                kcm_out=np.append(kcm_out,avg_kcm)
            id_kcm = sorted(range(len(kcm_out)), key=lambda k: kcm_out[k], reverse=True)



            for id in range(0,1):
                top_box = bboxes[id_out[id]]
                top_crop = image[int(top_box[0]):int(top_box[2]),int(top_box[1]):int(top_box[3])]
                imgname = imgpath[0].split('/')[-1]
                cv2.imwrite(os.path.join(args.output_dir, 
                                         output_file_name(imgpath, id+1)),
                            top_crop[:,:,(2, 1, 0)])


if __name__ == '__main__':
    test()
