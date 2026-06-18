import os
from typing import Any, List, Dict, Optional

import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset, get_dataset_config_names
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from nase.data.components.xmind_dataframe import xMINDDataFrame
from nase.data.components.datasets import (
        DenoisingAutoEncoderDataset,
        DatasetCollate,
        RecDatasetCollate,
        RecommendationDatasetTest
        )
from nase.data.components.samplers import MultinomialSampler
from nase.data.components import file_utils
from nase.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DataModule(LightningDataModule):
    def __init__(
            self,
            polynews_dataset_name: str,
            parallel_data: bool,
            combined_objectives: bool,
            training_languages: Optional[List[str]],
            min_examples: Optional[int],
            behavior_sample_fraction: Optional[float],
            local_train_data_dir: Optional[str],
            local_train_file: str,
            local_train_fraction: Optional[float],
            local_train_max_rows: Optional[int],
            tokenizer: DictConfig,
            rec_tokenizer: DictConfig,
            xmind_dataset_size: str,
            mind_dataset_url: str,
            xmind_dataset_url: Dict[str, Dict[str, str]],
            data_dir: str,
            tgt_languages: List[str],
            dataset_attributes: List[str],
            valid_time_split: str,
            concatenate_inputs: bool,
            max_history_len: int,
            train_batch_size: int,
            val_batch_size: int,
            num_workers: int,
            pin_memory: bool,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.lg2dataset = {}
        self.tokenizer = hydra.utils.instantiate(self.hparams.tokenizer)
        self.rec_tokenizer = hydra.utils.instantiate(self.hparams.rec_tokenizer)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        if self.hparams.local_train_data_dir:
            local_train_path = os.path.join(self.hparams.local_train_data_dir, self.hparams.local_train_file)
            if not file_utils.check_integrity(local_train_path):
                raise FileNotFoundError(f"Local training dataset not found: {local_train_path}")

            log.info(f"Loading local training dataset from {local_train_path}.")
            dataset = load_dataset(
                    "csv",
                    data_files=local_train_path,
                    delimiter="\t",
                    split="train",
                    )

            if self.hparams.local_train_max_rows is not None:
                max_rows = min(self.hparams.local_train_max_rows, len(dataset))
                dataset = dataset.select(range(max_rows))

            if self.hparams.local_train_fraction is not None:
                fraction = self.hparams.local_train_fraction
                if not 0 < fraction <= 1:
                    raise ValueError("local_train_fraction must be in the interval (0, 1].")
                sample_size = max(1, int(len(dataset) * fraction))
                dataset = dataset.shuffle(seed=42).select(range(sample_size))

            dataset_name = os.path.basename(os.path.normpath(self.hparams.local_train_data_dir)) or "local"
            self.lg2dataset = {dataset_name: dataset}
            log.info(f"Using local training dataset '{dataset_name}' with {len(dataset)} rows for fine-tuning.\n")
        else:
            configs = get_dataset_config_names(self.hparams.polynews_dataset_name)
            if not self.hparams.parallel_data:
                languages = self.hparams.training_languages if self.hparams.training_languages else configs 
            else:
                 if self.hparams.training_languages:
                     languages = [lang for lang in configs if lang.split('-')[0] in self.hparams.training_languages or lang.split('-')[1] in self.hparams.training_languages]
                 else:
                     languages = configs
            
            log.info(f"Loading training datasets from {self.hparams.polynews_dataset_name}.")
            for lang in tqdm(languages):
                dataset = load_dataset(
                        self.hparams.polynews_dataset_name,
                        lang,
                        split='train'
                        )
                if self.hparams.min_examples:
                    if len(dataset) > self.hparams.min_examples:
                        self.lg2dataset[lang] = dataset

            languages_omitted = [lang for lang in languages if lang not in self.lg2dataset]
            log.info(f"Using {len(self.lg2dataset)} languages from PolyNews for pretraining. Omitted {len(languages_omitted)} languages with fewer than {self.hparams.min_examples} examples: {' '.join(languages_omitted)}\n")
        
        log.info("Loading validation datasets.")
        for tgt_lang in tqdm(self.hparams.tgt_languages):
            xMINDDataFrame(
                    dataset_size=self.hparams.xmind_dataset_size,
                    mind_dataset_url=self.hparams.mind_dataset_url,
                    xmind_dataset_url=self.hparams.xmind_dataset_url,
                    data_dir=self.hparams.data_dir,
                    tgt_lang=tgt_lang,
                    eval_tgt_lang=True if tgt_lang != 'eng' else False,
                    dataset_attributes=self.hparams.dataset_attributes,
                    valid_time_split=self.hparams.valid_time_split,
                    behavior_sample_fraction=self.hparams.behavior_sample_fraction,
                    train=True,
                    validation=True,
                    download_mind=True,
                    download_xmind=True if tgt_lang!='eng' else False
                    )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # training
        self.data_train = DenoisingAutoEncoderDataset(
                parallel_data=self.hparams.parallel_data,
                combined_objectives=self.hparams.combined_objectives,
                datasets=self.lg2dataset)

        # validation
        self.data_val = {}
        for tgt_lang in self.hparams.tgt_languages:
            validset = xMINDDataFrame(
                    dataset_size=self.hparams.xmind_dataset_size,
                    mind_dataset_url=self.hparams.mind_dataset_url,
                    xmind_dataset_url=self.hparams.xmind_dataset_url,
                    data_dir=self.hparams.data_dir,
                    tgt_lang=tgt_lang,
                    eval_tgt_lang=True if tgt_lang != 'eng' else False,
                    dataset_attributes=self.hparams.dataset_attributes,
                    valid_time_split=self.hparams.valid_time_split,
                    behavior_sample_fraction=self.hparams.behavior_sample_fraction,
                    train=True,
                    validation=True,
                    download_mind=False,
                    download_xmind=False
                    )
            self.data_val[tgt_lang] = RecommendationDatasetTest(
                    news=validset.news,
                    behaviors=validset.behaviors,
                    max_history_len=self.hparams.max_history_len
                    )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                dataset=self.data_train,
                sampler=MultinomialSampler(self.lg2dataset), 
                collate_fn=DatasetCollate(self.tokenizer),
                batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True
                )
    
    def val_dataloader(self) -> CombinedLoader:
        loaders = {
                tgt_lang: DataLoader(
                    dataset=self.data_val[tgt_lang],
                    batch_size=self.hparams.val_batch_size,
                    collate_fn=RecDatasetCollate(
                        dataset_attributes=self.hparams.dataset_attributes,
                        tokenizer=self.rec_tokenizer,
                        concatenate_inputs=self.hparams.concatenate_inputs
                        ),
                    shuffle=False,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    )
                for tgt_lang in self.data_val
                }
        return CombinedLoader(loaders, mode='sequential')
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
