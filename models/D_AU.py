import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import csv
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from Facecrop.FaceBoxes import FaceBoxes
class dataset(Dataset):
    def __init__(self,img_path,au_path):
        self.imgs=torch.from_numpy(np.load(img_path)).permute(0, 3, 1, 2)
        self.aus=torch.from_numpy(np.load(au_path))
    def __getitem__(self, idx):
        return self.imgs[idx],self.aus[idx]
    def __len__(self):
        return self.imgs.size(0)
class Traindataset(Dataset):
    def __init__(self,img_path,au_path):
        self.imgs1=torch.from_numpy(np.load(img_path)).permute(0, 3, 1, 2)
        self.aus1=torch.from_numpy(np.load(au_path))
        self.imgs2 = torch.from_numpy(np.load('../data/np-imgs/AU10-imgs.npy')).permute(0, 3, 1, 2)
        self.aus2 = torch.from_numpy(np.load('../data/np-imgs/AU10-au.npy'))
        self.l1=self.imgs1.size(0)
        self.l2=self.imgs2.size(0)
    def __getitem__(self, idx):
        if idx<self.l1:
            return self.imgs1[idx], self.aus1[idx][10]
        idx-=self.l1
        return self.imgs2[idx], self.aus2[idx][10]
    def __len__(self):
        return self.l1+self.l2
class Discriminator_AU(nn.Module):
    def __init__(self, input_nc=3, aus_nc=17, image_size=128, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        super(Discriminator_AU, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]
        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=nn.InstanceNorm2d),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)
        # patch discriminator top
        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
        # AUs classifier top
        k_size = int(image_size / (2 ** n_layers))
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)

        # from torchsummary import summary
        # summary(self.model.to("cuda"), (3, 128, 128))

    def forward(self, img):
        if img.shape[1]!=128 or img.shape[2]!=128:
            resize = transforms.Resize([128, 128])
            img=resize(img)
        if img.max()>10 and img.min()>=0:
            img=(img-127.5)/127.5
        if img.min()>=0 and img.max()<2:
            img=img*2-1
        embed_features = self.model(img)
        pred_aus = self.aus_top(embed_features)
        return pred_aus.squeeze()
def cal_weight(i,x):
    c=0.5
    x = x*20
    l=int(x)
    r=l+1
    if r==101:
        p=0.001
    else:
        p=max(au_frequence[i][r]*(x-l)+au_frequence[i][l]*(r-x),0.001)
    return c*(1/p+(1-c)*0.01)
def Cal_au_loss(y_hat,au,required_crop,D,face_box=None):
    if face_box==None:
        face_box = FaceBoxes(cuda=True)
    # y_hat=y_hat.cuda()
    # au=au.cuda()
    if not y_hat.is_cuda:
        y_hat=y_hat.cuda()
    if not au.is_cuda:
        au=au.cuda()
    if required_crop:
        y_crop = None
        resize = transforms.Resize([128, 128])
        for img in y_hat:
            det = face_box(img[[2, 1, 0]])
            det = list(map(int, det))
            img_crop = img[:, det[1]:det[3], det[0]:det[2]]
            img_crop = resize(img_crop)
            # save_image(img_crop.cpu().detach().float(), "results/%d.png" % random.randint(0,100000), nrow=1, normalize=True)
            img_crop = img_crop.unsqueeze(0)
            if y_crop == None:
                y_crop = img_crop
            else:
                y_crop = torch.cat((y_crop, img_crop), dim=0)
        pred_au = D(y_crop)
    else:
        pred_au = D(y_hat)
    loss = criterionMSE(au, pred_au).cpu().detach()
    return float(loss)
def Test(require_crop,D):
    if require_crop:
        face_boxes = FaceBoxes(cuda=True)
    total_loss = 0
    cnt = 0
    test_dataset = dataset("../data/img-256-test.npy", '../data/au-test.npy')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    for index, (img, au) in enumerate(test_loader):
        #save_image(img.float(), "../result/img/%d.png"%index, nrow=8, normalize=True)
        if require_crop:
            loss=Cal_au_loss(img,au,required_crop=True,face_box=face_boxes,D=D)
        else:
            loss = Cal_au_loss(img, au, required_crop=False, D=D)
        total_loss += loss
        cnt += 1
    return total_loss/cnt
if __name__ == '__main__':
    face_boxes = FaceBoxes(cuda=True)
    au_frequence = np.load("au-frequence.npy")
    img_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(128, scale=(0.8, 1)),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(contrast=0.5)
    ])
    criterionMSE = torch.nn.MSELoss()
    test=0
    if test==0:
        test_dataset = dataset("../data/img-256-test.npy",'../data/au-test.npy')
        train_dataset = Traindataset("../data/Final-128-img.npy","../data/Final-au.npy")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        D = Discriminator_AU()
        D.load_state_dict(torch.load("D-1dim-test.pth"))
        D = D.cuda()
        lr = 2e-5
        D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
        cnt = sum(p.numel() for p in D.parameters() if p.requires_grad)
        print("Total parameters:",cnt)
        print(Test(require_crop=True,D=D))
        for epoch in range(200):
            if epoch % 20 == 0 and epoch != 0 and lr > 2e-8:
                lr /= 2
                for param_group in D_optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
                    param_group["lr"] = lr
            for index, (img, au) in enumerate(train_loader):
                y_hat = img.float().cuda()/255.0
                au = au.float().cuda()
                y_hat=img_transform(y_hat)
                pred_au = D(y_hat)
                loss = criterionMSE(au, pred_au)
                D_optimizer.zero_grad()
                loss.backward()
                D_optimizer.step()
                if index % 100 == 0:
                    print("Epoch:%d, Index:%d, Loss:%f" % (epoch, index, loss))
            torch.save(D.state_dict(), "D_AU.pth")
            avg_loss=test_loss=Test(require_crop=True,D=D)
            print("testloss=",test_loss)
            with open("accuracy.txt", "a")as f:
                f.write("Epoch:%3d, Avg_loss%8.4f,Loss: %8.4f\n"%(epoch,avg_loss, test_loss))
    if test==1:
        test_dataset = dataset("../data/img-256-test.npy", "../data/au-test.npy")
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        D = Discriminator_AU()
        D.load_state_dict(torch.load("D-AU3.pth"))
        D = D.cuda()
        loss=Test(require_crop=True,D=D)
        print("Test_loss%f"%(loss))
