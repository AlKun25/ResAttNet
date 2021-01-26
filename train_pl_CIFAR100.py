from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.datamodule import LightningDataModule
import torch 
import torch.nn as nn 
from torch.nn import functional as F
import pytorch_lightning as pl
import time
import os
from torchvision.datasets import CIFAR100
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from .attention_module import *

class CIFAR100DataModule(pl.LightningDataModule):

    def prepare_data(self):
        # prepare transforms standard to CIFAR-100
        CIFAR100(root='./data/', train=True, download=True)
        CIFAR100(root='./data/', train=False, download=True)

    def train_dataloader(self):
        train_transform = transforms.Compose([  transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop((32, 32), padding=4), #left, top, right, bottom, 
                                                # transforms.Scale(224),
                                                transforms.ToTensor()
                                            ])
        cifar100_train = CIFAR100(root='./data/', train=True, download=False, transform=train_transform)
        cifar100_train = DataLoader( dataset=cifar100_train, 
                                    batch_size=16, 
                                    shuffle=True, 
                                    num_workers=4
                                )
        return cifar100_train

    def test_dataloader(self):
        test_transform = transforms.Compose([transforms.ToTensor()])
        cifar100_test = CIFAR100(root='./data/', train=False,download=False,transform=test_transform)
        cifar100_test = DataLoader(  dataset=cifar100_test, 
                                    batch_size=20, 
                                    shuffle=False
                                )
        return cifar100_test
        
class ResidualAttentionModel(pl.LightningModule):
    # for input size 32
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024,100)
        # self.model = model

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        logits = self.forward(images)
        loss = self.cross_entropy_loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    def test_step(self,test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = self.cross_entropy_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0001)
        return optimizer
        
class LitProgressBar(pl.callbacks.ProgressBar):

    def init_train_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running training ...')
        return bar

bar = LitProgressBar()
data_module = CIFAR100DataModule()
data_module.prepare_data()
test_data = data_module.test_dataloader()
train_data = data_module.train_dataloader()

model = ResidualAttentionModel()
trainer = pl.Trainer(max_epochs=10, gpus=-1, callbacks=[bar], accelerator='ddp')

trainer.fit(model, train_data)
trainer.test(test_dataloaders=test_data, verbose=True)
