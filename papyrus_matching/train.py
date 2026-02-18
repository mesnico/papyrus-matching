import argparse
from pathlib import Path
import random
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchmetrics.functional as tm

from .loader.datamodule import BalancedMatchDataModule


# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')


class LitPapyrusTR(pl.LightningModule):
    def __init__(self, lr=1e-3, pretrained=True, frozen_backbone=True):
        super().__init__()

        # create model, change the first layer to accept RGBA, change last layer to output a single value
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.maxvit_t(weights=weights)
        self.model.stem[0][0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[-1] = torch.nn.Linear(512, 1, bias=False)

        if frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True
            for param in self.model.stem[0][0].parameters():
                param.requires_grad = True
            for param in self.model.blocks[-1].layers[-1].parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Trainable parameters: {trainable_params}/{total_params} ({100 * trainable_params / total_params:.2f}%)")

        self.lr = lr
        self.temperature = 1

        self._step_outputs = defaultdict(list)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, stage, batch, batch_idx):
        img, y_true = batch

        logits = self.forward(img).squeeze()  # (B,)
        loss = F.binary_cross_entropy_with_logits(logits, y_true.float())
        y_scores = torch.sigmoid(logits)

        # Metrics
        accuracy = tm.accuracy(y_scores, y_true, task='binary')
        auroc = tm.auroc(y_scores, y_true, task='binary')

        # Log stuff
        if stage == 'train':
            self.log(f'{stage}/loss', loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log(f'{stage}/accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=False)
            self.log(f'{stage}/auroc', auroc, prog_bar=True, on_step=True, on_epoch=False)

        out = {
            'loss': loss,
            'y_scores': y_scores,
            'y_true': y_true
        }

        self._step_outputs[stage].append(out)
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._common_step('test', batch, batch_idx)

    def _common_epoch_end(self, stage):
        step_outputs = self._step_outputs[stage]
        keys = list(step_outputs[0].keys())
        metrics = {key: [i[key].detach().cpu() for i in step_outputs] for key in keys}

        y_scores = torch.cat(metrics['y_scores'])
        y_true = torch.cat(metrics['y_true'])

        loss = torch.mean(torch.stack(metrics['loss']))
        accuracy = tm.accuracy(y_scores, y_true, task='binary')
        auroc = tm.auroc(y_scores, y_true, task='binary')

        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/accuracy', accuracy, prog_bar=True)
        self.log(f'{stage}/auroc', auroc, prog_bar=True)

        figure = sns.histplot(x=y_scores, hue=y_true).get_figure()
        self.logger.experiment.add_figure(f'{stage}/scores', figure, self.current_epoch)

        self._step_outputs[stage].clear()

    def on_train_epoch_end(self):
        self._common_epoch_end('train')

    def on_validation_epoch_end(self):
        self._common_epoch_end('val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


def main(args):
    seed_everything(42, workers=True)

    run_dir = 'runs/'

    dm = BalancedMatchDataModule(
        data_root='data/unified',
        batch_size=args['batch_size'],
        num_workers=8,
    )
    scorer = LitPapyrusTR(lr=1e-4)

    resume = None
    if args.get('resume', False):
        ckpts = Path(run_dir).glob('lightning_logs/version_*/checkpoints/*.ckpt')
        ckpts = sorted(ckpts, reverse=True, key=lambda p: p.stat().st_mtime)
        resume = ckpts[0] if ckpts else None

    trainer = Trainer(
        default_root_dir=run_dir,
        max_epochs=args['epochs'],
        accelerator='gpu',
        # deterministic=True,
        callbacks=[
            ModelCheckpoint(monitor="val/auroc", mode='max', save_last=True),
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(scorer, dm, ckpt_path=resume)
    trainer.test(scorer, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Papyrus Match Scorer')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    args = parser.parse_args()
    args = vars(args)
    main(args)