from typing import List, Dict
import os
import json
import re

import pandas as pd

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader, Dataset

import torchaudio
import slue_toolkit.text_ner.ner_deberta_modules as ndm

import hydra
from hydra.utils import get_original_cwd

from src.utils.text_preprocess import remove_special_characters, extract_all_chars


# todo: set up matrix for subword tokens
class VoxPopuliDataset(Dataset):
    def __init__(
        self,
        manifest_dir: str = "../slue-toolkit/manifest/slue-voxpopuli/nlp_ner",
        e2e_manifest_dir: str = "../slue-toolkit/manifest/slue-voxpopuli/e2e_ner",
        vocab_path: str = "./data/voxpopuli_vocab.json",
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

        with open(os.path.join(get_original_cwd(), e2e_manifest_dir,f"{split}.combined.ltr")) as f:
            e2e_ltr = f.read()
            self.e2e_ltr_list = e2e_ltr.split("\n")

        self.df["normalized_text"] = self.df["normalized_text"].apply(lambda s: remove_special_characters(s))

        if not os.path.exists(os.path.join(get_original_cwd(), vocab_path)):
            self.vocab = extract_all_chars(
                self.df["normalized_text"]
            )
            self.vocab.sort()
            vocab_dict = {v: k for k, v in enumerate(self.vocab)}
            vocab_dict["|"] = vocab_dict[" "]
            del vocab_dict[" "]
            vocab_dict["[UNK]"] = len(vocab_dict)
            vocab_dict["[PAD]"] = len(vocab_dict)

            with open(os.path.join(get_original_cwd(), vocab_path), "w") as vocab_file:
                json.dump(vocab_dict, vocab_file)

    def __getitem__(self, index) -> Dict:
        waveform = self.load_audio(index)
        normalized_text = self.df.iloc[index]["normalized_text"]
        e2e_ner_text = self.e2e_ltr_list[index].replace(" ", "")
        item = self.dataset.__getitem__(index)
        input_ids = item["input_ids"]
        token_type_ids = item["token_type_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        return {
            "waveform": waveform,
            "text": normalized_text,
            "e2e_text": e2e_ner_text,
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
    normalized_text = [remove_special_characters(data["text"]) for data in inputs]
    e2e_text = [data["e2e_text"] for data in inputs]
    input_ids = torch.stack([data["input_ids"] for data in inputs])
    token_type_ids = torch.stack([data["token_type_ids"] for data in inputs])
    attention_mask = torch.stack([data["attention_mask"] for data in inputs])
    labels = torch.stack([data["labels"] for data in inputs])
    padded_waveforms = rnn.pad_sequence(waveforms, batch_first=True)
    return {
        "waveform": padded_waveforms,
        "text": normalized_text,
        "e2e_text": e2e_text,
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
    e2e_manifest_dir: str,
    vocab_path: str,
    split: str,
    data_dir: str,
    model_type: str,
    label_type: str,
):
    assert split in ["fine-tune", "dev", "test"], "Invalid Split"

    vp_dataset = VoxPopuliDataset(
        manifest_dir=manifest_dir,
        e2e_manifest_dir=e2e_manifest_dir,
        vocab_path=vocab_path,
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
def dataset_debug(cfg) -> None:
    dataset = VoxPopuliDataset(
        data_dir="../../../../slue-toolkit/data/slue-voxpopuli/",
        manifest_dir="../../../../slue-toolkit/manifest/slue-voxpopuli/nlp_ner",
        e2e_manifest_dir="../../../../slue-toolkit/manifest/slue-voxpopuli/e2e_ner",
        vocab_path="../../../data/voxpopuli_vocab.json",
        split="fine-tune",
    )
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    dataset_debug()
