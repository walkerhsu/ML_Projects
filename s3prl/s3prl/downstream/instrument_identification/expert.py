import os
import math
from pathlib import Path
import pandas as pd
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import InstrumentDataset, collate_fn


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs
    ):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame

            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.

                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.meta_data_root = self.datarc["meta_data_root"]

        self.fold = self.datarc["fold"]
        if self.fold is None:
            self.fold = "fold1"

        print(
            f'[Expert] - using the testing fold: "{self.fold}". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.'
        )

        train_path = os.path.join(
            self.meta_data_root, self.fold.replace("fold", "Session"), "Training.json"
        )
        print(f"[Expert] - Training path: {train_path}")

        test_path = os.path.join(
            self.meta_data_root, self.fold.replace("fold", "Session"), "Testing.json"
        )
        print(f"[Expert] - Testing path: {test_path}")

        dataset = InstrumentDataset(
            train_path, "train", self.datarc["train_batch_size"], **self.datarc
        )
        trainlen = int((1 - self.datarc["valid_ratio"]) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]

        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = InstrumentDataset(
            test_path, "test", self.datarc["eval_batch_size"], **self.datarc
        )

        self.connector = nn.Linear(upstream_dim, self.modelrc["input_dim"])
        model_cls = eval(self.modelrc["select"])
        model_conf = self.modelrc.get(self.modelrc["select"], {})
        self.model = model_cls(
            input_dim=self.modelrc["input_dim"],
            output_dim=dataset.instruments_num,
            **model_conf,
        )

        self.objective = nn.BCEWithLogitsLoss()
        self.register_buffer("best_score", torch.zeros(1))
        self.expdir = expdir

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            split: string
                'train'
                    will always be called before the training loop

                'dev', 'test', or more
                    defined by the 'eval_dataloaders' field in your downstream config
                    these will be called before the evaluation loops during the training loop

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if split == "train":
            return self._get_train_dataloader(self.train_dataset, epoch)
        elif split == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler

        sampler = get_ddp_sampler(dataset, epoch)
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=collate_fn,
        )

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        """
        Args:
            split: string
                'train'
                    when the forward is inside the training loop

                'dev', 'test' or more
                    when the forward is inside the evaluation loop

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records (also customized by you)

                Note1. downstream/runner.py will call self.log_records
                    1. every `log_step` during training
                    2. once after evalute the whole dev/test dataloader

                Note2. `log_step` is defined in your downstream config
                eg. downstream/example/config.yaml

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """

        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(
            device=device
        )

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        predicted, _ = self.model(features, features_len)
        label_truth = []
        for label in labels:
            label_truth.append([0.0 for _ in range(self.test_dataset.instruments_num)])
            for l in label:
                label_truth[-1][l] = 1.0

        label_truth = torch.Tensor(label_truth).to(features.device)
        loss = self.objective(predicted, label_truth)
        predicted_classid = []
        accs = []
        for idx, predicted_one in enumerate(predicted.cpu().tolist()):
            predicted_classid.append([])
            accs.append(0)
            # if more than one that is larger than 0.5, select them
            predicted_classid[-1] = [
                i for i, x in enumerate(predicted_one) if x >= 0.75
            ]
            # if none : select the largest one
            if len(predicted_classid[-1]) == 0:
                predicted_classid[-1] = [predicted_one.index(max(predicted_one))]

            records["predict"].append(predicted_classid[-1])
            records["truth"].append(labels[idx])
            for label in labels[idx]:
                if label in predicted_classid[-1]:
                    accs[-1] += 1.0
            accs[-1] /= len(labels[idx]) + len(predicted_classid[-1]) - accs[-1]

        records["acc"] += accs
        records["loss"].append(loss.item())

        records["filename"] += filenames

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev', 'test' or more:
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f"instrument/{mode}-{key}", average, global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", "a") as f:
                if key == "acc" or key == "loss":
                    print(f"{mode} {key}: {average}")
                    f.write(f"{mode} at step {global_step}: {average}\n")
                    if key == "acc" and mode == "dev" and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(
                            f"New best on {mode} at step {global_step}: {average}\n"
                        )
                        save_names.append(f"{mode}-best.ckpt")

        if mode in ["dev", "test"]:
            with open(
                Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w"
            ) as file:
                assert len(records["filename"]) == len(records["predict"])
                line = [
                    f"{f} {e}\n"
                    for f, e in zip(records["filename"], records["predict"])
                ]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                assert len(records["filename"]) == len(records["truth"])
                line = [
                    f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])
                ]
                file.writelines(line)

        return save_names
