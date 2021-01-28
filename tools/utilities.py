import pytorch_lightning as pl

class LitProgressBar(pl.callbacks.ProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('running training ...')
        bar.leave = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.set_description('running testing ...')
        bar.leave = True
        return bar