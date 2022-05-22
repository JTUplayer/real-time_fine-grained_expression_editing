# coding: utf-8

import os.path as osp

import torch
import numpy as np
import cv2
from torchvision import transforms
from .utils.prior_box import PriorBox
from .utils.nms_wrapper import nms
from .utils.box_utils import decode
from .utils.timer import Timer
from .utils.functions import check_keys, remove_prefix, load_model
from .utils.config import cfg
from .models.faceboxes import FaceBoxesNet

# some global configs


def viz_bbox( img, dets, wfp='out.jpg'):
    # show
    vis_thres = 0.5
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite(wfp, img)
    print(f'Viz bbox to {wfp}')



class FaceBoxes:

    def __init__(self,cuda=False):
        self.confidence_threshold = 0.05
        self.top_k = 5000
        self.keep_top_k = 750
        self.nms_threshold = 0.3
        self.vis_thres = 0.5
        self.resize = 1
        self.scale_flag = True
        self.HEIGHT, self.WIDTH = 720, 1080
        make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
        self.pretrained_path = make_abs_path('weights/FaceBoxesProd.pth')
        self.cuda=cuda
        net = FaceBoxesNet(phase='test', size=None, num_classes=2)  # initialize detector
        self.net = load_model(net, pretrained_path=self.pretrained_path, load_to_cpu=True)
        if self.cuda:
            self.net.cuda()
        # print('Finished loading model!')
    def __call__(self, img_raw):
        # scaling to speed up
        if img_raw.max()<=10 and img_raw.min()>=0:
            img_raw=img_raw*255
        if img_raw.min()<0:
            img_raw=(img_raw+1)*127.5
        scale = 1
        if self.scale_flag:
            h, w = img_raw.shape[1:]
            if h > self.HEIGHT:
                scale = self.HEIGHT / h
            if w * scale > self.WIDTH:
                scale *= self.WIDTH / (w * scale)
            # print(scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                Resize = transforms.Resize([h_s,w_s])
                img_raw_scale = Resize(img_raw)

            img = img_raw_scale.float()
        else:
            img = np.float32(img_raw)

        # forward
        _,im_height, im_width= img.shape
        scale_bbox = torch.Tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]]).cuda()
        img[0] -= 104
        img[1]-=117
        img[2]-=123
        img = img.unsqueeze(0)
        loc, conf = self.net(img)  # forward pass
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data.cuda()
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        if self.scale_flag:
            boxes = boxes * scale_bbox / scale / self.resize
        else:
            boxes = boxes * scale_bbox / self.resize

        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        inds=scores.argmax()
        # ignore low scores
        b = list(boxes[inds])
        b.append(scores[inds])
        det_bboxes=[]
        xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
        bbox = [xmin, ymin, xmax, ymax, score]
        return bbox


def main():
    face_boxes = FaceBoxes(cuda=True)
    import os
    path='/home/shuzixi/ht/dataset/rafd-ori-front'
    file_names=os.listdir(path)
    for index,file in enumerate(file_names):
        if index%1000==0:
            print(index)
        file_path=os.path.join(path,file)
        img = cv2.imread(file_path)
        img_raw=torch.tensor(img).cuda()
        img_raw = img_raw.permute((2, 0, 1))
        det = face_boxes(img_raw)
        det= list(map(int, det))
        new_img=img[det[1]:det[3],det[0]:det[2],:]
        resize_img=cv2.resize(new_img,(128,128))
        cv2.imwrite(os.path.join("/home/shuzixi/ht/dataset/crop_img/rafd", file), resize_img)
    # mmi_path = '/home/shuzixi/ht/dataset/mmi'
    # mmi = np.load("/home/shuzixi/ht/pixel2style2pixel/data/mmi.npy", allow_pickle=True).item()
    # mmi_keys = list(mmi.keys())
    # new_mmi={}
    # for idx in range(len(mmi)):
    #     if idx%1000==0:
    #         print(idx)
    #     img=cv2.imread(os.path.join(mmi_path, mmi_keys[idx]))
    #     det = face_boxes(img)
    #     det= list(map(int, det))
    #     new_img=img[det[1]:det[3],det[0]:det[2],:]
    #     resize_img=cv2.resize(new_img,(128,128))
    #     new_name=str(idx)+'.jpg'
    #     cv2.imwrite(os.path.join("/home/shuzixi/ht/dataset/crop_img/mmi",new_name), resize_img)
    #     new_mmi[new_name]=mmi[mmi_keys[idx]]
    # np.save("new_mmi",new_mmi)
if __name__ == '__main__':
    main()
