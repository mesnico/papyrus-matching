import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models, transforms, utils
import torchmetrics.functional as tm

from dataloader import PapyrMatchesDataset


def collate_fn(batch):

    def _common(batch, direction):
        dir_batch = [i for i in batch if i[2].direction == direction]
        A = [i[0] for i in dir_batch]
        B = [i[1] for i in dir_batch]
        matches = [i[2] for i in dir_batch]

        A = torch.stack(A) if A else torch.tensor([])
        B = torch.stack(B) if B else torch.tensor([])
        return A, B, matches

    hA, hB, hmatches = _common(batch, 'horizontal')
    vA, vB, vmatches = _common(batch, 'vertical')
    return hA, hB, hmatches, vA, vB, vmatches


class PapyriMatchesDataModule(pl.LightningDataModule):
    def __init__(self, root='data/fragments/', batch_size=8):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # RGBA
        ])

    def setup(self, stage=None):
        # find papyri files
        image_paths = sorted(self.root.rglob('*.png'))

        # split image list
        n_images = len(image_paths)
        n_train = round(n_images * 0.50)
        n_valid = round(n_images * 0.25)
        n_test  = n_images - n_train - n_valid

        train_image_paths = image_paths[:n_train]
        valid_image_paths = image_paths[n_train:n_train + n_valid]
        test_image_paths  = image_paths[-n_test:]

        common = dict(
            patch_size=112,
            aspect=2,
            transform=self.transform,
        )

        self.train_dataset = ConcatDataset([PapyrMatchesDataset(i, **common) for i in train_image_paths])
        self.valid_dataset = ConcatDataset([PapyrMatchesDataset(i, **common) for i in valid_image_paths])
        self.test_dataset  = ConcatDataset([PapyrMatchesDataset(i, **common) for i in test_image_paths ])

        print("train_dataset len:", len(self.train_dataset))
        print("valid_dataset len:", len(self.valid_dataset))
        print( "test_dataset len:", len(self.test_dataset ))

    def _common_dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, collate_fn=collate_fn, batch_size=self.batch_size, pin_memory=True, num_workers=4, **kwargs)

    def train_dataloader(self):
        return self._common_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._common_dataloader(self.valid_dataset)

    def test_dataloader(self):
        return self._common_dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self._common_dataloader(self.test_dataset)

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass


class LitPapyrusTR(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        # create model, change the first layer to accept RGBA, change last layer to output a single value
        self.model = models.maxvit_t()
        self.model.stem[0][0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[-1] = torch.nn.Linear(512, 1, bias=False)
        
        self.lr = lr
        self.temperature = 1

        self._step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, stage, batch, batch_idx):
        hA, hB, hmatches, vA, vB, vmatches = batch
        device = hA.device if hA.numel() > 0 else vA.device

        def _combine_all(A, B, direction):
            axis = 2 if direction == 'horizontal' else 1
            matches = [torch.cat((Ai, Bj), dim=axis) for Ai in A for Bj in B]
            matches = torch.stack(matches)
            return matches

        def _forward_and_loss(A, B, direction):
            n = len(A)
            if n == 0:
                logits = torch.tensor([], device=device, dtype=torch.float32)
                loss = torch.tensor(0, device=device, dtype=torch.float32)
                return logits, loss
    
            AB = _combine_all(A, B, direction).to(device)  # (n x n, 4, H, W)

            if batch_idx == 0 and n > 1:
                gridA  = 0.5 + utils.make_grid(A [:, :3], nrow=n) / 2
                gridB  = 0.5 + utils.make_grid(B [:, :3], nrow=n) / 2
                gridAB = 0.5 + utils.make_grid(AB[:, :3], nrow=n) / 2

                self.logger.experiment.add_images(f'{stage}/A' , gridA , self.current_epoch, dataformats='CHW')
                self.logger.experiment.add_images(f'{stage}/B' , gridB , self.current_epoch, dataformats='CHW')
                self.logger.experiment.add_images(f'{stage}/AB', gridAB, self.current_epoch, dataformats='CHW')

            logits = self.forward(AB).view(n, n)  # (n, n)
            targets = torch.arange(n).to(device)  # (n)
            logits = logits / self.temperature
            lossAB = F.cross_entropy(logits, targets, reduction='sum')
            lossBA = F.cross_entropy(logits.T, targets, reduction='sum')
            loss = lossAB + lossBA
            return logits, loss

        h_logits, h_loss = _forward_and_loss(hA, hB, 'horizontal')
        v_logits, v_loss = _forward_and_loss(vA, vB, 'vertical')

        nh, nv = len(hA), len(vA)
        loss = (h_loss + v_loss) / 2 * (nh + nv)

        # Metrics
        y_scores = torch.cat((h_logits.ravel(), v_logits.ravel()))
        y_true = torch.cat((torch.eye(nh, dtype=torch.int).flatten(), torch.eye(nv, dtype=torch.int).flatten())).to(device)

        accuracy = tm.accuracy(y_scores, y_true, task='binary')
        auroc = tm.auroc(y_scores, y_true, task='binary')

        # Log stuff
        if stage == 'train':
            self.log(f'{stage}/loss', loss, prog_bar=True)
            self.log(f'{stage}/accuracy', accuracy, prog_bar=True)
            self.log(f'{stage}/auroc', auroc, prog_bar=True)

        out = {
            'loss': loss,
            'y_scores': y_scores,
            'y_true': y_true
        }

        self._step_outputs.append(out)
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._common_step('test', batch, batch_idx)

    def _common_epoch_end(self, stage):
        step_outputs = self._step_outputs
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

        self._step_outputs.clear()

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

    dm = PapyriMatchesDataModule(batch_size=4)
    scorer = LitPapyrusTR(lr=1e-3)

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
            ModelCheckpoint(monitor="val/auroc", save_last=True),
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(scorer, dm, ckpt_path=resume)
    trainer.test(scorer, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Papyrus Match Scorer')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    args = parser.parse_args()
    args = vars(args)
    main(args)