##############################################################################
############################ Prepare Dependencies ############################
import shutil

from git import Repo
from pathlib import Path

TOOLBOX_PROPS = {
    "name": "toolbox",
    "repo_name": "doi-ml-toolbox",
    "remote_repo_url": "https://github.com/NareshPS/doi-ml-toolbox.git",
    "repo_relative_path": "torch/toolbox",
}


def prepare_toolbox():
    toolbox_path = Path(TOOLBOX_PROPS["name"])

    # 1. Skip toolbox download if it already exists
    if toolbox_path.exists():
        print(f"[INFO] Toolbox exists locally. Skipping download!")
    else:
        # 2. Clone toolbox's container repository
        print(
            f'[INFO] Cloning toolbox container repository: {TOOLBOX_PROPS["remote_repo_url"]}'
        )
        Repo.clone_from(TOOLBOX_PROPS["remote_repo_url"], TOOLBOX_PROPS["repo_name"])

        # 3. Construct path to the toolbox inside the repository
        repo_path = Path(TOOLBOX_PROPS["repo_name"])
        toolbox_in_repo_path = repo_path / TOOLBOX_PROPS["repo_relative_path"]

        if toolbox_in_repo_path.exists():
            # 4. Copy toolbox to the current directory
            print(f"[INFO] Copying the toolbox from {toolbox_in_repo_path}")
            shutil.copytree(toolbox_in_repo_path, toolbox_path)
        else:
            print(f"[ERROR] Path: {toolbox_in_repo_path} does not exist.")

        # 5. Perform Cleanup.
        shutil.rmtree(repo_path)


##############################################################################

##############################################################################
############################## Download Dataset ##############################

DATASET_PROPS = {
    "name": "pizza_steak_sushi",
    "remote_store_url": "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
}


def download_data():
    from toolbox import data_download

    """Downloads pizza_sushi_steak dataset from GitHub

    Returns:
        train_path: Path to the train set
        test_path: Path to the test set
    """
    # 1. Download pizza, steak, sushi images from GitHub
    image_path = data_download.download_data(
        source=DATASET_PROPS["remote_store_url"],
        destination=DATASET_PROPS["name"],
    )

    # 2. Setup train and test paths
    train_path = image_path / "train"
    test_path = image_path / "test"

    print(f"Data Path: {image_path}")
    print(f"Train Path: {train_path}")
    print(f"Test Path: {test_path}")

    return train_path, test_path


##############################################################################


##############################################################################
############################# Parse Commandline Arguments #############################
import argparse


def create_args_parser():
    parser = argparse.ArgumentParser(description="Pizza Steak Sushi Classifier")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Directory to place training logs",
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="Transformer embeddings dimensions",
    )
    parser.add_argument(
        "--num_heads", type=int, required=True, help="The number of transformer heads"
    )
    parser.add_argument(
        "--num_encoders",
        type=int,
        required=True,
        help="The number of transformer encoders",
    )
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        required=False,
        help="The size of image patch",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="The training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        required=False,
        default=None,
        help="The number of samples to use to train and validate.",
    )
    # return parser.parse_args()
    return parser


##############################################################################

##############################################################################
################################ Define Model ################################
import torch
import os

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torcheval.metrics.functional import multiclass_accuracy


# 1. Create PizzaSteakSushiClassifier as LightningModule.
class PizzaSteakSushiClassifier(LightningModule):

    """An implementation of Vision Transformer"""

    # 2. Initialize model hyperparameters.
    def __init__(
        self,
        args,
        img_size: int = 224,
        embedding_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ):
        from toolbox.models.vit import PatchEmbedding, TransformerEncoderBlock, ViT

        super().__init__()

        # 2.1. Tunable Hyperparameters
        self.embedding_dim = args.embedding_dim
        self.num_heads = args.num_heads
        self.num_encoders = args.num_encoders
        self.learning_rate = args.learning_rate
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size

        # 2.2 Fixed Hyperparameters
        self.in_channels = 3
        self.num_classes = 3
        self.mlp_dim = self.embedding_dim * 4
        self.img_size = img_size

        # 2.3. Other Settings
        self.data_workers = os.cpu_count()
        self.subset_size = args.subset_size

        # 3. Add assert to ensure that image is divisible into patches.
        assert (
            img_size % self.patch_size == 0
        ), f"img_size: {img_size} % patch_size: {self.patch_size} != 0"

        # 4. Compute the number of patches.
        self.num_patches = (img_size // self.patch_size) ** 2

        # 5. Create a learnable class embedding token.
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim))

        # 6. Create a learnable position embedding token.
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, self.embedding_dim)
        )

        # 7. Create a embedding dropout layer.
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create a patch embedding layer using PatchEmbedding module.
        self.patch_embedding = PatchEmbedding(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            patch_size=self.patch_size,
        )

        # 9. Initialize a Sequential with a series of TransformerEncoderBlocks.
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(self.num_encoders)
            ]
        )

        # 10. Create a classifier head with a LayerNorm and a Linear layer.
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=self.num_classes),
        )

    # 11. Create a forward method.
    def forward(self, x):
        # 12. Get the batch size of the input.
        batch_size = x.shape[0]

        # 13. Create class token embedding for each element in the batch.
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # 14. Create patch embedding.
        x = self.patch_embedding(x)

        # 15. Attach class token embedding to the patch embedding.
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to the patch and class token embedding.
        x = self.position_embedding + x

        # 17. Pass the patch and position embedding through the dropout layer (Step 7).
        x = self.embedding_dropout(x)

        # 18. Pass the patch and position embeddings from step 16 through the stack of transformer encoders.
        x = self.encoder(x)

        # 19. Pass index 0 of the output of the stack of transformer encoders through the classifier head.
        x = self.classifier(x[:, 0])

        # 20. That's ViT for you!
        return x

    # 21. Create training step
    def training_step(self, batch, batch_idx):
        return self.compute_loss_and_accuracy(batch, "train_loss", "train_acc")

    # 22. Create validation step
    def validation_step(self, batch, batch_idx):
        return self.compute_loss_and_accuracy(batch, "val_loss", "val_acc")

    def compute_loss_and_accuracy(self, batch, loss_metric_name, acc_metric_name):
        # 1. Separate samples and labels
        X, y = batch

        # 2. Apply the model function to the inputs
        logits = self(X)

        # 3. Compute loss between logits and true labels
        loss = F.cross_entropy(logits, y)

        # 4. Conpute model prediction accuracy
        acc = multiclass_accuracy(logits, y)

        # 5. Log loss and accuracy metrics
        self.log(
            loss_metric_name,
            loss,
            on_epoch=True,
        )
        self.log(
            acc_metric_name,
            acc,
            on_epoch=True,
        )

        return loss

    # 23. Configure Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # 24. Download Dataset
    def prepare_data(self):
        # 24.1. Get download location
        train_path, val_path = download_data()

        # 24.2. Save dataset locations
        self.train_path = train_path
        self.val_path = val_path

    # 25. Create a function to setup datasets
    def setup(self, stage=None):
        # 25.1. Create a transform to apply over train and validation set.
        transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )

        # 25.2. Create train dataset
        self.train_data = datasets.ImageFolder(
            root=self.train_path, transform=transform
        )

        # 25.3. Create test dataset
        self.val_data = datasets.ImageFolder(root=self.val_path, transform=transform)

        # 25.4. Get class names
        self.class_names = self.train_data.classes

    # 26. Create Train DataLoader
    def train_dataloader(self):
        if self.subset_size is None:
            dataset = self.train_data
        else:
            print(f"[WARN] Using Subset Size: {self.subset_size}")
            dataset = torch.utils.data.Subset(
                self.train_data, list(range(self.subset_size))
            )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.data_workers,
            shuffle=True,
            pin_memory=True,
        )

    # 27. Create Validation DataLoader
    def val_dataloader(self):
        if self.subset_size is None:
            dataset = self.val_data
        else:
            print(f"[WARN] Using Subset Size: {self.subset_size}")
            dataset = torch.utils.data.Subset(
                self.val_data, list(range(self.subset_size))
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.data_workers,
            shuffle=False,
            pin_memory=True,
        )


##############################################################################

##############################################################################
#################################### Main ####################################

import time

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, Callback

IMG_SIZE = 224

if __name__ == "__main__":
    # 1. Prepare the environment
    prepare_toolbox()

    # 2. Download Dataset
    train_path, test_path = download_data()

    # 3. Parse commandline arguments
    parser = create_args_parser()
    args = parser.parse_args()

    # 4. Initialize Logger
    print(f"[INFO] Logging to path: {args.log_path}")
    logger = pl_loggers.TensorBoardLogger(args.log_path)

    # 5. Define Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        enable_progress_bar=False,
        deterministic=True,
        default_root_dir=args.log_path,
    )

    # 6. Create Model
    model = PizzaSteakSushiClassifier(
        args,
        img_size=IMG_SIZE,
        embedding_dropout=0.1,
        attn_dropout=0.0,
        mlp_dropout=0.1,
    )

    # 7. Train
    start = time.time()
    trainer.fit(model=model)
    end = time.time()

    # 8.1. Log Training Time
    train_time = end - start
    logger.log_metrics({"train_time": end - start})

    # 8.2. Compute Validation Accuracy
    # with io.capture_output() as captured:
    val_accuracy = trainer.validate()[0]["val_acc"]

    # 8.3. Log Validation Accuracy
    logger.log_metrics({"val_acc": val_accuracy})

    # 8.4. Log the number of model parameters
    num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.log_metrics({"num_params": num_params})

    # 9. Save the logs
    logger.save()
    logger.finalize("success")

    # 10. Print Run Summary
    print(
        f"train time: {train_time}, val acc: {val_accuracy}, num_params: {num_params}"
    )
##############################################################################
