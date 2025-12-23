import hydra
import pytorch_lightning as pl
from model import LSTMClassifierLightning
from omegaconf import DictConfig
from transformers import BertTokenizer

from data import YelpPolarityDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    tokenizer = BertTokenizer.from_pretrained(config.module.tokenizer_name)
    data_module = YelpPolarityDataModule(
        train_path=config.infer.train_path,
        test_path=config.infer.test_path,
        batch_size=config.infer.batch_size,
        num_workers=config.infer.num_workers,
        seed=config.random.seed,
        tokenizer=tokenizer,
        max_length=config.infer.max_length,
        val_split=config.infer.val_split,
    )

    model = LSTMClassifierLightning.load_from_checkpoint(
        config["infer"]["ckpt_path"], weights_only=False
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto")

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
