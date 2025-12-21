import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from lstm_classifier import LSTMClassifier
from model import LSTMClassifierLightning
from data import YelpPolarityDataModule
from transformers import BertTokenizer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(config.random.seed)
    tokenizer = BertTokenizer.from_pretrained(config.module.tokenizer_name)
    data_module = YelpPolarityDataModule(
        train_path=config.module.train_path,
        test_path=config.module.test_path,
        batch_size=config.module.batch_size,
        num_workers=config.module.num_workers,
        seed = config.random.seed,
        tokenizer=tokenizer,
        max_length=config.module.max_length,
        val_split=config.module.val_split,
    )
    
    model_torch = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.training.embedding_dim,
        hidden_dim=config.training.hidden_dim,
        seed=config.random.seed,
        output_dim=config.training.output_dim,
        n_layers=config.training.n_layers,
        dropout=config.training.dropout,
        pad_idx=tokenizer.pad_token_id,
    )

    model = LSTMClassifierLightning(
        model=model_torch,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,

    )

    loggers = [
        pl.loggers.MLFlowLogger(
             experiment_name=config.logging.experiment_name,
             run_name=config.logging.run_name,
             save_dir=".",
             tracking_uri=config.logging.tracking_uri,
        )
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1, 
        accelerator=config["training"]["device"],
        devices=config["training"]["num_devices"],
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)



if __name__ == "__main__":
    main()