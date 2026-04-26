print("starting script...")

import torch
import torch.nn as nn  #alshan elmodels
import torch.nn.functional as F #alshan el activiation function
import pytorch_lightning as pl #alshan el training loop w optimization
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path


#encoder w decoder
#input -> conv _> features -> pool -> smaller image
class Encoder(nn.Module):
    def __init__(self,inchann,outchann):
        super(Encoder,self).__init__()
        #mn el top ll botom layer "seqquential"
        self.double_conv = nn.Sequential(
            nn.Conv2d(inchann,outchann,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(outchann,outchann,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)#keep max val w bt divde el hieght w el width ala 2 channel nfs el haga

    def forward(self,x: torch.Tensor):#extract features keeping spatial size
        x=self.double_conv(x)#skip connection
        xpool=self.pool(x)#goes deeper
        print(x.shape)
        print(xpool.shape)
        return xpool,x#alshan el model yrember el features elly extractated w ystkhdmha fe el decoder


# small image ->   upsample ->  cominefeature saved "mn el skip" ->  inpainting   
class Decoder(nn.Module):
    def __init__(self,inchann,outchann):
        super(Decoder,self).__init__()
        self.up=nn.ConvTranspose2d(inchann,outchann,kernel_size=2,stride=2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(inchann,outchann,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(outchann,outchann,kernel_size=3,padding=1),
            nn.ReLU()
        )

    def forward(self,x:torch.tensor,skip:torch.tensor):
        x=self.up(x)
        x=torch.cat((x,skip),dim=1)
        x=self.double_conv(x)
        print(x.shape)
        print(skip.shape)
        return x


class UNET(nn.Module):
    def __init__(self,inchann=3,outchann=3,blocks=4,startchann=8):
        super().__init__()
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()

        self.encoders.append(Encoder(inchann,startchann))
        chann=startchann

        for i in range(blocks-1):
            self.encoders.append(Encoder(chann,chann*2))
            chann*=2

        self.bottlneck=nn.Sequential(
            nn.Conv2d(chann,chann*2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(chann*2,chann*2,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        chann*=2

        for i in range(blocks-1):
            self.decoders.append(Decoder(chann,chann//2))
            chann//=2

        self.output=nn.Conv2d(chann,outchann,kernel_size=1)

    def forward(self,x:torch.Tensor):
        skips=[]

        for encoder in self.encoders:
            x,skip=encoder(x)
            skips.append(skip)

        x=self.bottlneck(x)

        for decoder in self.decoders:
            skip=skips.pop()
            x=decoder(x,skip)

        x=self.output(x)
        return x


# -------------------------------
# DATASET + MASK LINKED HERE
# -------------------------------
class CelebADataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images = list(self.root.rglob("*.jpg"))

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)

        mask = torch.ones_like(img)
        h,w = img.shape[1], img.shape[2]

        mh, mw = h//4, w//4
        y = torch.randint(0,h-mh,(1,)).item()
        x = torch.randint(0,w-mw,(1,)).item()

        mask[:,y:y+mh,x:x+mw] = 0

        x_in = img * mask

        return x_in, img


# -------------------------------
# LIGHTNING MODEL
# -------------------------------
class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet = UNET()
        self.loss = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self,x):
        return self.unet(x)

    def compute_psnr(self, mse):
        return 10 * torch.log10(1.0 / (mse + 1e-8))

    def training_step(self,batch, batchid):
        x,y=batch
        ypred=self(x)
        lossfun=self.loss(ypred,y)
        self.log("train_loss",lossfun,prog_bar=True)
        return lossfun

    def validation_step(self,batch,batchid):

        x,y=batch
        ypred=self(x)

        lossfun=self.loss(ypred,y)
        mse = self.mse(ypred, y)
        psnr = self.compute_psnr(mse)

        self.log("val_l1", lossfun, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)

        print("validatio loss",lossfun,"PSNR",psnr)

        ypred_img = ypred[0].detach().cpu().numpy().transpose(1,2,0)
        y_img = y[0].detach().cpu().numpy().transpose(1,2,0)

        ypred_img = (ypred_img + 1)/2
        y_img = (y_img + 1)/2

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(ypred_img)
        axs[0].set_title("Predicted")
        axs[1].imshow(y_img)
        axs[1].set_title("Ground Truth")
        plt.suptitle(f"Epoch {self.current_epoch}")
        plt.show()

        return lossfun

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# -------------------------------
# TRAIN ENTRY (LINK DATA HERE)
# -------------------------------
if __name__ == "__main__":

    DATA_PATH = r"C:\Users\toaa ramadan\Desktop\comp vision project\u-net\data\processed"

    dataset = CelebADataset(DATA_PATH)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_ds,batch_size=8,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=8)

    net = Model()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto"
    )

    trainer.fit(net,train_loader,val_loader)