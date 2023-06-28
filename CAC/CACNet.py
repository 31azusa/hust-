import os

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
import einops
import numpy as np
from matplotlib import pyplot as plt
from torchvision.ops import roi_pool
from PIL import Image
from torchvision.transforms import ToTensor
import pickle

from config_cropping import cfg #引入了一个具体的class

class vgg_base(nn.Module):
    def __init__(self, loadweights=True):
        super(vgg_base, self).__init__()
        vgg = models.vgg16(pretrained=loadweights)
        self.feature1 = nn.Sequential(vgg.features[:6])      # /2
        self.feature2 = nn.Sequential(vgg.features[6:10])    # /4
        self.feature3 = nn.Sequential(vgg.features[10:17])   # /8
        self.feature4 = nn.Sequential(vgg.features[17:30])   # /16

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f2,f3,f4

class CompositionModel(nn.Module):
    def __init__(self):
        super(CompositionModel, self).__init__()
        self.comp_types = 9
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.ReLU(True)
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Linear(128, self.comp_types, bias=True)
        for m in self.modules():#为模型赋予合理的初始值
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f2, f3, f4):
        x = self.conv1(f4)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f3
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f2
        x = self.conv4(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        conf   = F.softmax(logits, dim=1)
        with torch.no_grad():
            B,C,H,W = x.shape
            w  = self.fc_layer.weight.data # cls_num, channels
            trans_w = einops.repeat(w, 'n c -> b n c', b=B)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x) # b n hw
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=H, w=W)
            kcm = torch.sum(conf[:,:,None,None] * cam, dim=1, keepdim=True)
            kcm = F.interpolate(kcm, scale_factor=4, mode='bilinear', align_corners=True)#双线性插值，放缩4倍
            return logits, kcm #预测构图规则类别和热力图

class CroppingModel(nn.Module):
    def __init__(self, anchor_stride):
        super(CroppingModel, self).__init__()
        self.anchor_stride = anchor_stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        out_channel = int((16 / anchor_stride)**2 * 4)
        self.output = nn.Conv2d(256, out_channel, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''
        :param x: b,512,H/16,W/16
        :return: b,4. anchor shifts of the best crop
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.output(x)
        return out

def generate_anchors(anchor_stride):
    assert anchor_stride <= 16, 'not implement for anchor_stride{} > 16'.format(anchor_stride)
    P_h = np.array([2+i*4 for i in range(16 // anchor_stride)])
    P_w = np.array([2+i*4 for i in range(16 // anchor_stride)])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1
    return anchors

def shift(shape, stride, anchors):
    shift_w = torch.arange(0, shape[0]) * stride
    shift_h = torch.arange(0, shape[1]) * stride
    shift_w, shift_h = torch.meshgrid([shift_w, shift_h])
    shifts  = torch.stack([shift_w, shift_h], dim=-1)  # h,w,2
    # add A anchors (A,2) to
    # shifts (h,w,2) to get
    # shift anchors (A,h,w,2)
    trans_anchors = einops.rearrange(anchors, 'a c -> a 1 1 c')
    trans_shifts  = einops.rearrange(shifts,  'h w c -> 1 h w c')
    all_anchors   = trans_anchors + trans_shifts
    return all_anchors

class PostProcess(nn.Module):
    def __init__(self, anchor_stride, image_size):
        super(PostProcess, self).__init__()
        self.num_anchors = (16 // anchor_stride) ** 2
        anchors = generate_anchors(anchor_stride)
        feat_shape  = (image_size[0] // 16, image_size[1] // 16)
        all_anchors = shift(feat_shape, 16, anchors)
        all_anchors = all_anchors.float().unsqueeze(0) # 1,num_anchors,h//16,w//16,2
        self.upscale_factor = self.num_anchors // 2
        anchors_x   = F.pixel_shuffle(all_anchors[...,0], upscale_factor=self.upscale_factor)
        anchors_y   = F.pixel_shuffle(all_anchors[...,1], upscale_factor=self.upscale_factor)
        # 1,h//s,w//s,2 where s=16//anchor_stride
        all_anchors = torch.stack([anchors_x, anchors_y], dim=-1).squeeze(1)
        self.register_buffer('all_anchors', all_anchors)
        # build grid for sampling the pixel from KCM
        grid_x = (all_anchors[...,0] - image_size[0]/2) / (image_size[0]/2)
        grid_y = (all_anchors[...,1] - image_size[1]/2) / (image_size[1]/2)
        # 1,h//s,w//s,2, on a range of [-1,1]
        grid   = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid', grid)

    def forward(self, offsets, kcm):
        '''
        :param offsets: b,num_anchors*4,h//16,w//16
        :param kcm: b,1,h,w
        :return: b,4
        '''
        offsets = einops.rearrange(offsets, 'b (n c) h w -> b n h w c',
                                   n=self.num_anchors, c=4)
        coords  = [F.pixel_shuffle(offsets[...,i], upscale_factor=self.upscale_factor) for i in range(4)]
        # b, h//s, w//s, 4, where s=16//anchor_stride
        offsets = torch.stack(coords, dim=-1).squeeze(1)
        regression = torch.zeros_like(offsets) # b,h,w,4
        regression[...,0::2] = offsets[..., 0::2] + self.all_anchors[...,0:1]
        regression[...,1::2] = offsets[..., 1::2] + self.all_anchors[...,1:2]

        trans_grid  = einops.repeat(self.grid, '1 h w c -> b h w c',
                                    b=offsets.shape[0])
        # b,1,h//s, w//s
        sample_kcm  = F.grid_sample(kcm, trans_grid, mode='bilinear', align_corners=True)
        reg_weight  = F.softmax(sample_kcm.flatten(1), dim=1).unsqueeze(-1)
        regression  = einops.rearrange(regression, 'b h w c -> b (h w) c')
        weighted_reg = torch.sum(reg_weight * regression, dim=1)
        return weighted_reg

class ComClassifier(nn.Module):#分类器
    def __init__(self, loadweights=True):
        super(ComClassifier, self).__init__()
        self.backbone   = vgg_base(loadweights=loadweights)
        self.composition_module = CompositionModel()

    def forward(self, x, only_classify=False):
        f2,f3,f4 = self.backbone(x)
        logits,kcm = self.composition_module(f2,f3,f4)
        return logits,kcm

class CACNet(nn.Module):
    def __init__(self, loadweights=True):
        super(CACNet, self).__init__()
        anchor_stride = 8
        image_size = cfg.image_size
        assert cfg.backbone == 'vgg16', cfg.backbone
        self.backbone  = vgg_base(loadweights=loadweights)
        self.composition_module = CompositionModel()
        self.cropping_module = CroppingModel(anchor_stride)
        self.post_process = PostProcess(anchor_stride, image_size)

    def forward(self, im, only_classify=False):
        f2,f3,f4 = self.backbone(im) # 1/4, 1/8, 1/16
        logits,kcm = self.composition_module(f2,f3,f4)
        if only_classify:
            return logits,kcm
        else:
            offsets = self.cropping_module(f4)
            box = self.post_process(offsets, kcm)
            return logits, kcm, box

def output_file_name(input_path, idx):
    name = os.path.basename(input_path)
    segs = name.split('.')
    assert len(segs) >= 2
    return '%scrop_%d.%s'%('.'.join(segs[:-1]), idx, segs[-1])

if __name__ == '__main__':
    device = torch.device('cuda:0')
    #x #= torch.randn(2,3, cfg.image_size[0],cfg.image_size[1])
    #x = Image.open('workspace/aesthetic_cropping/dataset/FCDB/data/2551085398_fbb77b329a_b.jpg')
    #y = Image.open('workspace/aesthetic_cropping/dataset/FCDB/data/76537045_5997edc2af_b.jpg')
    #x = Image.open('workspace/211144.jpg')
    #x = Image.open('workspace/aesthetic_cropping/dataset/FCDB/data/11020299_16c8146f6e_o.jpg').convert('RGB')
    #x = cv2.imread('workspace/211144.jpg')
    x = cv2.imread('dataset/211144.jpg')
    x = x[:, :, (2, 1, 0)]
    width,height = x.shape[1],x.shape[0]
    h = cfg.image_size[1]
    w = cfg.image_size[0]
    #resized_image = x.resize((w, h), Image.ANTIALIAS)  # 第二个参数用于放缩时抗锯齿处理，使得边缘更加平滑
    resized_image = cv2.resize(x,(w,h))
    tensor_image = ToTensor()(resized_image).unsqueeze(0)
    weight_file = "./pretrained_model/best-FLMS_iou.pth"
    model = CACNet(loadweights=True)
    model.load_state_dict(torch.load(weight_file, map_location=device))#加载预训练模型
    model = model.to(device).eval()
    model = model.to(device)
    #source_img=cv2.imread('workspace/aesthetic_cropping/dataset/FCDB/data/11020299_16c8146f6e_o.jpg')
    cls,kcm,crop = model(tensor_image.to(device))
    # 将 kcm 转换成 numpy 数组并归一化
    crop[:, 0::2] = crop[:, 0::2] / 224 * width  # 横向恢复
    crop[:, 1::2] = crop[:, 1::2] / 224 * height  # 纵向
    best_crop = crop[0].detach().cpu().numpy().tolist()
    best_crop = [int(x) for x in best_crop]

    croped_img = x[best_crop[1]: best_crop[3], best_crop[0]: best_crop[2]]
    croped_img = x[28: 484, 26: 996]
    croped_img = croped_img[:, :, (2, 1, 0)]
    cv2.imwrite('dataset/test_result/crop211144.jpg', croped_img)
    # kcm_arr = kcm.squeeze().detach().cpu().numpy()
    # #
    # kcm_arr /= np.max(kcm_arr)
    # #
    # kcm_resized = cv2.resize(kcm_arr, (width, height), interpolation=cv2.INTER_LINEAR)
    #
    # kcm_best_crop = kcm_resized[best_crop[0]: best_crop[2],best_crop[1]: best_crop[3]]
    # np.save('219695.npy',kcm_resized)
    # source_value = kcm_resized.sum()
    # crop_value = kcm_best_crop.sum()
    # print('box', crop)
    # # # 创建彩色图像对象并显示
    # h, w = kcm_arr.shape
    # fig1 = plt.figure()
    # plt.imshow(kcm_resized, cmap='hot')
    # plt.colorbar()
    #
    # fig2 = plt.figure()
    # plt.hist(kcm_arr.flatten(), bins=256, range=(0, 1), density=True, color='red', alpha=0.5)
    #
    # fig3 = plt.figure()
    #
    #
    # #plt.imshow('Image',croped_img)
    #
    # plt.show()
    # kcm_color = np.zeros((h, w, 3), dtype=np.uint8)
    # kcm_color[..., 0] = (kcm_arr * 255).astype(np.uint8)  # 红通道
    # kcm_color[..., 1] = (kcm_arr * 255).astype(np.uint8)  # 绿通道
    # kcm_color[..., 2] = (kcm_arr * 255).astype(np.uint8)  # 蓝通道
    # kcm_img = Image.fromarray(kcm_color)
    # kcm_img.show()
    # print(cls.shape, box.shape)
    # print('classification', cls)

    # model = ComClassifier()
    # print(model(x))
