import json
import pytorch_lightning as pl
import pands as pd
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import F1

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import label_ranking_average_precision_score


from box import Box
config = Box.from_json("config.json")

from models.audio_dataset import WaveformDataset, BirdDataset
from utils.metrics import BCEFocal2WayLoss, BCEFocalLoss

TRAIN_DF = pd.read_csv("train.df")

sfk = StratifiedKFold(
    n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    
class BirdsDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            BirdDataset(self._train_df)
            if train
            else BirdDataset(self._val_df, validation=True)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


class BirdsLightModel(pl.LightningModule):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.criterion = BCEFocal2WayLoss()
        self.f1 = F1(num_classes=config.num_classes, average='macro')
        self.save_hyperparameters(cfg)
        
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.forward(x)
        f1_score = self.f1(loss["logit"], y.int())
        self.log("val_loss", loss["logit"], on_epoch=True, prog_bar=True)
        self.log("val_f1_score", f1_score, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_f1_score': f1_score}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def run():
    for fold, (train_idx, val_idx) in enumerate(skf.split(TRAIN_DF["filename"], TRAIN_DF["primary_label"])):
        train_df = TRAIN_DF.loc[train_idx].reset_index(drop=True)
        val_df = TRAIN_DF.loc[val_idx].reset_index(drop=True)
        datamodule = BirdsDataModule(train_df, val_df, config)
        model = BirdsLightModel(config, BirdsSED())
        earystopping = EarlyStopping(monitor="val_f1_score")
        swa = callbacks.StochasticWeightAveraging()
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_f1_score",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(config.model.name)
        
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            gradient_clip_val=0.5,
            callbacks=[lr_monitor, loss_checkpoint, earystopping, swa],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)
        torch.cuda.empty_cache()