from utilities import *
import pytorch_lightning as pl

import sys
sys.path.append('.')

from data import *
from layers import *
from model import *

'''
n_classes and image_size are important variable that need to be changed to train different models
Set n_classes = 10 for CIFAR10 and 100 for CIFAR100
Set image_size = 224 for RAN_56_224 or RAN_92_224 and image_size = 32 for RAN_92_32
'''
n_classes = 10 # number of classes for classification
image_size = 32 # size of the input image in pixels 

bar = LitProgressBar()
data_module = CIFAR10DataModule(image_size)
data_module.prepare_data()
test_data = data_module.test_dataloader()
train_data = data_module.train_dataloader()

model = ResidualAttentionModel_92_32(n_classes)
trainer = pl.Trainer(max_epochs=10, gpus=-1, callbacks=[bar], accelerator='ddp')

trainer.fit(model, train_data_10)
trainer.test(test_dataloaders=test_data_10, verbose=True)