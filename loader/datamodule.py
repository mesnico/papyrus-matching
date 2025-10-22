import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, Subset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T2
from typing import List, Iterator
import numpy as np

from loader.pos_real import PositiveRealMatchDataset
from loader.pos_synth import PositiveSyntheticMatchDataset
from loader.neg import NegativeMatchDataset
from loader import utils


# =============================================================================
# 2. Helper Components
# =============================================================================

class LabelWrapperDataset(Dataset):
    """
    A wrapper to add a fixed label to all samples from an existing dataset.
    """
    def __init__(self, dataset: Dataset, label: int):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        return image, self.label


class BalancedBatchSampler(Sampler[List[int]]):
    """
    A custom batch sampler to create balanced batches of positive and negative samples.
    
    This sampler ensures that each batch contains an equal number of samples from
    two distinct groups of indices (e.g., positive and negative classes).
    """
    def __init__(self, pos_indices: List[int], neg_indices: List[int], batch_size: int):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size

        # Ensure the batch size is even for a 50/50 split
        if self.batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for a 50/50 split, but got {self.batch_size}.")

        self.pos_per_batch = self.batch_size // 2
        self.neg_per_batch = self.batch_size // 2

        # Determine the number of batches based on the smaller class
        self.num_pos_batches = len(self.pos_indices) // self.pos_per_batch
        self.num_neg_batches = len(self.neg_indices) // self.neg_per_batch
        self.num_batches = min(self.num_pos_batches, self.num_neg_batches)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices at the start of each epoch
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

        for i in range(self.num_batches):
            # Get indices for the current batch
            start_pos = i * self.pos_per_batch
            end_pos = start_pos + self.pos_per_batch
            batch_pos_indices = self.pos_indices[start_pos:end_pos]

            start_neg = i * self.neg_per_batch
            end_neg = start_neg + self.neg_per_batch
            batch_neg_indices = self.neg_indices[start_neg:end_neg]

            # Combine and shuffle the batch indices
            batch = batch_pos_indices + batch_neg_indices
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


# =============================================================================
# 3. The PyTorch Lightning DataModule
# =============================================================================

class BalancedMatchDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that combines positive and negative datasets
    and uses a balanced batch sampler for training.
    """
    def __init__(
        self,
        train_root: str = "data/organized",
        val_root: str = "data/organized_test",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.train_pos_indices = None
        self.train_neg_indices = None

        self.train_transforms = T2.Compose([
            T2.ToImage(),
            T2.Resize((224, 224)),
            utils.ApplyToRGB(
                T2.JPEG(quality=(20, 100)),
            ),
            utils.ApplyToRGB(
                T2.RandomPosterize(bits=4, p=0.2),
            ),
            utils.ApplyToRGB(
                T2.RandomGrayscale(p=0.2)
            ),
            utils.ApplyToRGB(
                T2.ColorJitter(0.5, 0.5, 0.3, 0.1)
            ),
            T2.ToDtype(torch.float32, scale=True),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.masks_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def setup(self, stage: str = None):
        """
        This method is called by Lightning to prepare the data.
        - It creates the combined datasets for training and validation.
        - It identifies positive and negative indices for the training sampler.
        """
        if stage == "fit" or stage is None:
            # --- Setup for Training Data ---
            pos_real_train = PositiveRealMatchDataset(self.train_root, transform=self.train_transforms, mask_transform=self.masks_transforms, erosion_size=(1, 10))
            pos_synth_train = PositiveSyntheticMatchDataset(self.train_root, transform=self.train_transforms, mask_transform=self.masks_transforms, max_shift=30, pad=3)
            neg_train = NegativeMatchDataset(self.train_root, transform=self.train_transforms, mask_transform=self.masks_transforms, max_shift=30, pad=3)

            # Combine positive datasets
            pos_train = ConcatDataset([pos_real_train, pos_synth_train])
            
            # Wrap datasets with labels
            labeled_pos_train = LabelWrapperDataset(pos_train, label=1)
            labeled_neg_train = LabelWrapperDataset(neg_train, label=0)
            
            # Create the final training dataset
            self.train_dataset = ConcatDataset([labeled_pos_train, labeled_neg_train])

            # Get indices for the balanced sampler
            len_pos = len(labeled_pos_train)
            len_neg = len(labeled_neg_train)
            self.train_pos_indices = list(range(len_pos))
            self.train_neg_indices = list(range(len_pos, len_pos + len_neg))
            
            print(f"Training setup complete. Positive samples: {len_pos}, Negative samples: {len_neg}")

            # --- Setup for Validation Data ---
            pos_real_val = PositiveRealMatchDataset(self.val_root, transform=self.val_transforms, mask_transform=self.masks_transforms, erosion_size=4)
            pos_synth_val = PositiveSyntheticMatchDataset(self.val_root, transform=self.val_transforms, mask_transform=self.masks_transforms)
            neg_val = NegativeMatchDataset(self.val_root, transform=self.val_transforms, mask_transform=self.masks_transforms)
            
            # Combine and wrap validation datasets
            pos_val = ConcatDataset([pos_real_val, pos_synth_val])
            pos_val_to_neg_val_ratio = len(pos_val) // len(neg_val)
            assert pos_val_to_neg_val_ratio >= 1, "We assume for now that val set has far more positive samples than negative samples."
            pos_val = Subset(pos_val, range(0, len(neg_val) * pos_val_to_neg_val_ratio, pos_val_to_neg_val_ratio))
            labeled_pos_val = LabelWrapperDataset(pos_val, label=1)
            labeled_neg_val = LabelWrapperDataset(neg_val, label=0)

            self.val_dataset = ConcatDataset([labeled_pos_val, labeled_neg_val])
            print(f"Validation setup complete. Positive samples: {len(labeled_pos_val)}, Negative samples: {len(labeled_neg_val)}")


    def train_dataloader(self) -> DataLoader:
        """
        Creates the training DataLoader with the balanced batch sampler.
        """
        if self.train_dataset is None:
            raise RuntimeError("The 'setup' method must be called before creating the dataloader.")

        sampler = BalancedBatchSampler(
            pos_indices=self.train_pos_indices,
            neg_indices=self.train_neg_indices,
            batch_size=self.batch_size
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler, # Use batch_sampler instead of sampler
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates the validation DataLoader. Uses a standard sampler to evaluate
        on the true data distribution.
        """
        if self.val_dataset is None:
            raise RuntimeError("The 'setup' method must be called before creating the dataloader.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )