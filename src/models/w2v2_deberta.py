from typing import Any, List
import os
import torch

import torch.nn as nn
import torch.nn.functional as f
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, WordErrorRate, CharErrorRate

from src.models.wav2vec_ft_hug import Wav2Vec2FTModule

from transformers import Wav2Vec2CTCTokenizer

from hydra.utils import get_original_cwd


class W2V2DebertaModule(LightningModule):
    def __init__(
        self,
        audio_ckpt_path: str = "facebook/hubert-larget-ls960-ft",
        vocab_path: str = "./data/e2e_voxpopuli_vocab.json",
        lm_pretrain_model: str = "microsoft/deberta-base",
        optimizer: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        w2v2_output_size: int = 768,
    ):
        super().__init__()
        # todo: may change to pytorch flash:
        self.save_hyperparameters(logger=False)

        self.tokenizer = Wav2Vec2CTCTokenizer(
            os.path.join(
                get_original_cwd(),
                self.hparams.vocab_path,
            ),
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_lower_case=True,
        )

        self.wav2vec2 = Wav2Vec2FTModule.load_from_checkpoint(
            os.path.join(
                get_original_cwd(),
                audio_ckpt_path,
            )
        )
        # self.deberta = DebertaModel.from_pretrained(lm_pretrain_model)

        self.lm_head = nn.Linear(
            self.hparams.w2v2_output_size,
            self.tokenizer.vocab_size,
        )

        self.train_cer = CharErrorRate()
        self.train_wer = WordErrorRate()
        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()
        # not yet used
        # self.test_cer = CharErrorRate()
        # self.test_wer = WordErrorRate()

        # for logging best so far validation accuracy
        self.val_cer_best = MinMetric()
        self.val_wer_best = MinMetric()

    def forward(self, inputs):
        output = self.wav2vec2(inputs)
        output = self.lm_head(output.hidden_states[-1])
        logits = f.log_softmax(output, dim=-1).transpose(0, 1)

        labels = inputs["labels"]
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        attention_mask = (
            inputs["attention_mask"] if "attention_mask" in inputs.keys() else torch.ones_like(inputs["input_values"])
        )
        input_lengths = self.wav2vec2.model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        with torch.backends.cudnn.flags(enabled=False):
            loss = f.ctc_loss(
                logits,
                flattened_targets,
                input_lengths,
                target_lengths,
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    def step(self, batch: Any):
        inputs = dict()
        inputs["input_values"] = batch["waveform"]
        inputs["output_hidden_states"] = True

        inputs["labels"] = self.tokenizer(
            batch["e2e_text"],
            return_tensors="pt",
            padding=True,
        ).input_ids

        output = self.forward(inputs)

        logits = output["logits"]
        loss = output["loss"]

        preds = torch.argmax(logits, dim=-1)
        pred = self.tokenizer.batch_decode(preds)

        return loss, pred, [text.lower() for text in batch["text"]]

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred, gt = self.step(batch)

        # print(pred, gt)

        cer = self.train_cer(pred, gt)
        wer = self.train_wer(pred, gt)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cer", cer, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/wer", wer, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, gt = self.step(batch)

        print(pred, gt)

        # log val metrics
        cer = self.val_cer(pred, gt)
        wer = self.val_wer(pred, gt)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/cer", cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", wer, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        cer = self.val_cer.compute()
        wer = self.val_wer.compute()
        self.val_cer_best.update(cer)
        self.val_wer_best.update(wer)
        self.log(
            "val/cer_best", self.val_cer_best.compute(), on_epoch=True, prog_bar=True
        )
        self.log(
            "val/wer_best", self.val_wer_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_wer.reset()
        self.train_cer.reset()
        self.val_cer.reset()
        self.val_wer.reset()

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optimizer)(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
