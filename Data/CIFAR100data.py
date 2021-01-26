from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.datamodule import LightningDataModule
import torch 
import pytorch_lightning as pl
from torchvision.datasets import CIFAR100
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

class CIFAR100DataModule(pl.LightningDataModule):

    def prepare_data(self):
        # prepare transforms standard to CIFAR-100
        CIFAR100(root='data', train=True, download=True)
        CIFAR100(root='data', train=False, download=True)

    def train_dataloader(self):
        train_transform = transforms.Compose([  transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop((32, 32), padding=4), #left, top, right, bottom, 
                                                # transforms.Scale(224),
                                                transforms.ToTensor()
                                            ])
        cifar100_train = CIFAR100(root='data', train=True, download=False, transform=train_transform)
        cifar100_train = DataLoader( dataset=cifar100_train, 
                                    batch_size=16, 
                                    shuffle=True, 
                                    num_workers=4
                                )
        return cifar100_train

    def test_dataloader(self):
        test_transform = transforms.Compose([transforms.ToTensor()])
        cifar100_test = CIFAR100(root='data', train=False,download=False,transform=test_transform)
        cifar100_test = DataLoader(  dataset=cifar100_test, 
                                    batch_size=20, 
                                    shuffle=False
                                )
        return cifar100_test
