import sys
sys.path.append('.')

from Data.CIFAR10data import *
from layers.residual_attention_network import *
from modelling.ResNet32 import *

n_layers = 10

bar = LitProgressBar()
data_module_10 = CIFAR10DataModule()
data_module_10.prepare_data()
test_data_10 = data_module_10.test_dataloader()
train_data_10 = data_module_10.train_dataloader()

model = ResidualAttentionModel(n_layers)
trainer = pl.Trainer(max_epochs=10, gpus=-1, callbacks=[bar], accelerator='ddp')

trainer.fit(model, train_data_10)
trainer.test(test_dataloaders=test_data_10, verbose=True)