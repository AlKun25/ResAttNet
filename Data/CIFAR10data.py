from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.datamodule import LightningDataModule
import torch 
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

class CIFAR10DataModule(pl.LightningDataModule):

    def prepare_data(self):
        # prepare transforms standard to CIFAR-10
        CIFAR10(root='Data', train=True, download=True)
        CIFAR10(root='Data', train=False, download=True)

    def train_dataloader(self):
        train_transform = transforms.Compose([  transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop((32, 32), padding=4), #left, top, right, bottom, 
                                                # transforms.Scale(224),
                                                transforms.ToTensor()
                                            ])
        cifar10_train = CIFAR10(root='Data', train=True, download=False, transform=train_transform)
        cifar10_train = DataLoader( dataset=cifar10_train, 
                                    batch_size=16, 
                                    shuffle=True, 
                                    num_workers=4
                                )
        return cifar10_train

    def test_dataloader(self):
        test_transform = transforms.Compose([transforms.ToTensor()])
        cifar10_test = CIFAR10(root='Data', train=False,download=False,transform=test_transform)
        cifar10_test = DataLoader(  dataset=cifar10_test, 
                                    batch_size=20, 
                                    shuffle=False
                                )
        return cifar10_test