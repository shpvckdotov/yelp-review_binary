import hydra
import numpy as np
import pytorch_lightning as pl
from model import LSTMClassifierLightning
from omegaconf import DictConfig

from data import YelpPolarityDataModule


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: DictConfig):
    dm = YelpPolarityDataModule(config)

    data_module = YelpPolarityDataModule(
        train_path=config.infer.train_path,
        test_path=config.infer.test_path,
        batch_size=config.infer.batch_size,
        num_workers=config.infer.num_workers,
        seed = config.random.seed,
        tokenizer_name=config.infer.tokenizer_name,
        max_length=config.infer.max_length,
        val_split=config.infer.val_split,
    )

    model = LSTMClassifierLightning.load_from_checkpoint(config["inference"]["ckpt_path"])
    trainer = pl.Trainer(accelerator="auto", devices="auto")

    accs = trainer.predict(model, datamodule=dm)
    print(f"Test accuracy: {np.mean(accs):.2f}")


if __name__ == "__main__":
    main()