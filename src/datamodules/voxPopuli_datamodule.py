from pytorch_lightning import LightningDataModule
from src.datamodules.components.voxPopuli_dataset import build_voxpopuli_dataloader


class VoxPopuliDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        model_type: str = "deberta-base",
        manifest_dir: str = "../slue-toolkit/manifest/slue-voxpopuli/nlp_ner",
        label_type: str = "raw",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return build_voxpopuli_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            manifest_dir=self.hparams.manifest_dir,
            data_dir=self.hparams.data_dir,
            model_type=self.hparams.model_type,
            label_type=self.hparams.label_type,
            split="fine-tune",
        )

    def val_dataloader(self):
        return build_voxpopuli_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            manifest_dir=self.hparams.manifest_dir,
            data_dir=self.hparams.data_dir,
            model_type=self.hparams.model_type,
            label_type=self.hparams.label_type,
            split="dev",
        )

    def test_dataloader(self):
        return build_voxpopuli_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            manifest_dir=self.hparams.manifest_dir,
            data_dir=self.hparams.data_dir,
            model_type=self.hparams.model_type,
            label_type=self.hparams.label_type,
            split="dev",
        )

    def predict_dataloader(self):
        raise NotImplementedError
