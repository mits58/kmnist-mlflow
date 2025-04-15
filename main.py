import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms


class KuzushijiMNIST(pl.LightningDataModule):
    def __init__(self, batch_size: int, is_standardized: bool, root="data/"):
        super(KuzushijiMNIST, self).__init__()
        self.root_dir = root
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.batch_size = batch_size
        self.is_standardized = is_standardized

        self.transform = [transforms.ToTensor()]
        if self.is_standardized:
            self.transform.append(
                transforms.Normalize(mean=0.1918, std=0.3483)
            )
        self.transform = transforms.Compose(self.transform)

    def setup(self, stage=None):
        self.train_data = datasets.KMNIST(
            root=self.root_dir, download=True, train=True,
            transform=self.transform
        )
        self.train_data, self.val_data = random_split(self.train_data,
                                                      [55000, 5000])
        self.test_data = datasets.KMNIST(
            root=self.root_dir, download=True, train=False,
            transform=self.transform
        )

    def create_data_loader(self, data):
        return DataLoader(data, batch_size=self.batch_size, num_workers=8)

    def train_dataloader(self):
        return self.create_data_loader(self.train_data)

    def val_dataloader(self):
        return self.create_data_loader(self.val_data)

    def test_dataloader(self):
        return self.create_data_loader(self.test_data)


class KMNISTClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int, learning_rate: float,
                 activation: str, optimizer: str, **kwargs):
        super(KMNISTClassifier, self).__init__()
        self.scheduler = None
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.activation = getattr(torch.nn.functional, activation)
        # relu, hardtanh, hardswish, relu6, elu, tanh, selu, celu,
        # leaky_relu, prelu, rrelu, glu, gelu, sigmoid, mish
        self.optimizer = getattr(torch.optim, optimizer)
        # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, LBFGS
        # NAdam, RAdam, RMSprop, Rprop, SGD

        self.layer_1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.layer_2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_3 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        # 層数を変えられるようにしたいかも
        self.args = kwargs

        self.training_step_outputs = []
        self.validation_step_outsputs = []
        self.test_step_outputs = []

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        _, y_hat = torch.max(logits, dim=1)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=10)
        self.training_step_outputs.append({"loss": loss, "acc": acc})
        return {"loss": loss}

    def on_train_epoch_end(self):
        loss = torch.stack(
            [elem["loss"] for elem in self.training_step_outputs]
        ).mean()
        acc = torch.stack(
            [elem["acc"] for elem in self.training_step_outputs]
        ).mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.training_step_outputs.clear()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        _, y_hat = torch.max(logits, dim=1)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=10)
        self.validation_step_outsputs.append({"loss": loss, "acc": acc})
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        loss = torch.stack(
            [elem["loss"] for elem in self.validation_step_outsputs]
        ).mean()
        acc = torch.stack(
            [elem["acc"] for elem in self.validation_step_outsputs]
        ).mean()
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        self.validation_step_outsputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat, y, task="multiclass", num_classes=10)
        self.test_step_outputs.append(test_acc)
        return {"test_acc": test_acc}

    def on_test_epoch_end(self):
        avg_test_acc = torch.stack(self.test_step_outputs).mean()
        self.log("test_acc", avg_test_acc)

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.parameters(),
                                        lr=self.learning_rate)
        return self.optimizer


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    mlf_logger = MLFlowLogger(experiment_name="KuzushijiMNIST")

    dm = KuzushijiMNIST(**cfg.dataset)
    model = KMNISTClassifier(**cfg.model)
    mlf_logger.log_hyperparams(cfg.dataset)
    mlf_logger.log_hyperparams(cfg.model)
    dm.prepare_data()
    dm.setup()

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)
    trainer = Trainer(callbacks=[early_stop_callback], max_epochs=1000,
                      logger=mlf_logger)
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
