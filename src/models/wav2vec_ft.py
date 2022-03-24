from typing import Any, List

import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, WordErrorRate, CharErrorRate

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

from hydra.utils import get_original_cwd


class Wav2Vec2FTModule(LightningModule):
    def __init__(
        self,
        pretrain_name: str = "facebook/hubert-larget-ls960-ft",
        vocab_path: str = "./data/voxpopuli_vocab.json",
        optimizer: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()
        # todo: may change to pytorch flash:
        # https://devblog.pytorchlightning.ai/fine-tuning-wav2vec-for-speech-recognition-with-lightning-flash-bf4b75cad99a
        self.save_hyperparameters(logger=False)

        tokenizer = Wav2Vec2CTCTokenizer(
            os.path.join(
                get_original_cwd(),
                self.hparams.vocab_path,
            ),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.hparams.pretrain_name,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=self.processor.tokenizer.vocab_size,
        )
        self.model.config.ctc_zero_infinity = True
        self.model.freeze_feature_extractor()

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
                batch["text"],
                return_tensors="pt",
                padding=True,
            ).input_ids

        output = self.forward(inputs)

        logits = output.logits
        loss = output.loss

        preds = torch.argmax(logits, dim=-1)
        pred = self.processor.batch_decode(preds)

        # print("Debugging: \n Labels: ", inputs["labels"], "\nPreds: ", preds, "\nBatch text: ", batch["text"])
        # gt = self.processor.batch_decode(inputs["labels"])

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
