from typing import List, Dict
import os

import pandas as pd

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader, Dataset

import torchaudio
import slue_toolkit.text_ner.ner_deberta_modules as ndm

import hydra
from hydra.utils import get_original_cwd


class VoxPopuliDataset(Dataset):
    def __init__(
        self,
        manifest_dir: str = "../slue-toolkit/manifest/slue-voxpopuli/nlp_ner",
        split: str = "fine-tune",
        data_dir: str = "../slue-toolkit/data/slue-voxpopuli/",
        model_type: str = "deberta-base",
        label_type: str = "raw",
    ) -> None:
        assert label_type in ["raw", "combined"], "Invalid Label Type"

        self.data_dir = data_dir
        self.split = split

        data_obj = ndm.DataSetup(
            os.path.join(
                get_original_cwd(),
                manifest_dir,
            ),
            model_type,
        )
        _, _, _, _, self.dataset = data_obj.prep_data(
            split,
            label_type,
        )

        self.df = pd.read_csv(
            os.path.join(
                get_original_cwd(),
                data_dir + f"slue-voxpopuli_{split}.tsv",
            ),
            sep="\t",
        )

    def __getitem__(self, index) -> Dict:
        waveform = self.load_audio(index)
        normalized_text = self.df.iloc[index]["normalized_text"]
        item = self.dataset.__getitem__(index)
        input_ids = item["input_ids"]
        token_type_ids = item["token_type_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        return {
            "waveform": waveform,
            "normalized_text": normalized_text,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def load_audio(self, index):
        df_row = self.df.iloc[index]
        filename = os.path.join(
            get_original_cwd(),
            self.data_dir,
            f"{self.split}/{df_row['id']}.ogg",
        )
        waveform, sample_rate = torchaudio.load(
            filename,
            channels_first=True,
            format="ogg",
        )
        return waveform.squeeze()

    def __len__(self):
        return len(self.df)


def voxpopuli_collate_fn(
    inputs: List,
) -> Dict:
    waveforms = [data["waveform"] for data in inputs]
    normalized_text = [data["normalized_text"] for data in inputs]
    input_ids = torch.stack([data["input_ids"] for data in inputs])
    token_type_ids = torch.stack([data["token_type_ids"] for data in inputs])
    attention_mask = torch.stack([data["attention_mask"] for data in inputs])
    labels = torch.stack([data["labels"] for data in inputs])

    padded_waveforms = rnn.pad_sequence(waveforms, batch_first=True)
    return {
        "waveform": padded_waveforms,
        "normalized_text": normalized_text,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_voxpopuli_dataloader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    manifest_dir: str,
    split: str,
    data_dir: str,
    model_type: str,
    label_type: str,
):
    assert split in ["fine-tune", "dev", "test"], "Invalid Split"

    vp_dataset = VoxPopuliDataset(
        manifest_dir=manifest_dir,
        split=split,
        data_dir=data_dir,
        model_type=model_type,
        label_type=label_type,
    )

    return DataLoader(
        dataset=vp_dataset,
        batch_size=batch_size,
        collate_fn=voxpopuli_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True if split == "fine-tune" else False,
    )


@hydra.main(config_path=None)
def test_dataset(cfg) -> None:
    dataset = VoxPopuliDataset(
        data_dir="../../../../slue-toolkit/data/slue-voxpopuli/",
        manifest_dir="../../../../slue-toolkit/manifest/slue-voxpopuli/nlp_ner",
        split="fine-tune",
    )
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    test_dataset()
