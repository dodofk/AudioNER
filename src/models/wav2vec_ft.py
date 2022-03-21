from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, WordErrorRate, CharErrorRate

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2FTModule(LightningModule):
    def __init__(
        self,
        pretrain_name: str = "facebook/hubert-larget-ls960-ft",
        optimizer: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = Wav2Vec2ForCTC.from_pretrained(self.hparams.pretrain_name)
        self.processor = Wav2Vec2Processor.from_pretrained(self.hparams.pretrain_name)

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
        return self.model(**inputs)

    def step(self, batch: Any):
        inputs = dict()
        inputs["input_values"] = batch["waveform"]

        with self.processor.as_target_processor():
            inputs["labels"] = self.processor(
                batch["normalized_text"],
                return_tensors="pt",
                padding=True,
            ).input_ids

        output = self.forward(inputs)

        logits = output.logits
        loss = output.loss

        preds = torch.argmax(logits, dim=-1)
        pred = self.processor.batch_decode(preds)

        return loss, pred, [text.upper() for text in batch["normalized_text"]]

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred, gt = self.step(batch)

        cer = self.train_cer(pred, gt)
        wer = self.train_wer(pred, gt)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cer", cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/wer", wer, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, gt = self.step(batch)

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
