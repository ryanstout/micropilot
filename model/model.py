import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import timm
from torchmetrics import MeanSquaredError
from pytorch_lightning.loggers import WandbLogger
import wandb

class SteeringDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [img for img in os.listdir(directory) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')
        parts = img_name.split('_')
        steering_angle = float(parts[2])# / 20.0

        if self.transform:
            image, steering_angle = self.transform(image, steering_angle)

        # Convert steering_angle to tensor and ensure it is float32
        steering_angle = torch.tensor(steering_angle, dtype=torch.float32)
        
        return image, steering_angle

class AugmentAndNormalize(object):
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img, steering_angle):
        img = self.augment(img)
        if torch.rand(1) < 0.5:
            img = transforms.functional.hflip(img)
            steering_angle = -steering_angle
        return img, steering_angle
    

class SteeringAnglePredictor(nn.Module):
    def __init__(self, feature_dim, num_patches):
        super(SteeringAnglePredictor, self).__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        # Using nn.Sequential to define the model
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),  # Flatten patches and features
            nn.Linear(num_patches * feature_dim, 512),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x[:, 1:, :]  # Skipping the [CLS] token
        x = self.model(x)
        # Scale output to range [-1, 1]
        x = 2 * x - 1
        return x


class SteeringDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/8', batch_size=16, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers  # Allow number of workers to be configurable

    def setup(self, stage=None):
        dataset = SteeringDataset(directory=self.data_dir, transform=AugmentAndNormalize())
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)#, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)#, persistent_workers=True)


class SteeringModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vit = False

        if self.vit:
            self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.steering_angle_predictor = SteeringAnglePredictor(feature_dim=768, num_patches=196)

            # self.feature_extractor.head = nn.Identity()  # Remove classification head
            # self.regressor = nn.Sequential(
            #     nn.Linear(768, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 1)
            # )

        else:
            self.feature_extractor = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
            self.regressor = nn.Sequential(
                nn.Conv2d(in_channels=960, out_channels=480, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=480, out_channels=240, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(240, 1),
                nn.Tanh()
            )


        self.criterion = torch.nn.MSELoss()
        self.train_acc = MeanSquaredError()
        self.val_acc = MeanSquaredError()

    def forward(self, x):
        if self.vit:
            features = self.feature_extractor.forward_features(x)
            return self.steering_angle_predictor(features)

        else:
            features = self.feature_extractor(x)
        return self.regressor(features[-1])

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, reduce_fx=torch.mean)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # print(images.shape, labels.shape, images.min(), images.max(), labels.min(), labels.max())
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels)
        # self.val_acc.update(outputs.squeeze(), labels)

        if batch_idx == 0:
            # Log a few images to wandb, the correct steering and the predicted
            images = images[:4]
            labels = labels[:4].tolist()
            preds = outputs[:4].tolist()

            for i in range(4):
                image = images[i]
                label = labels[i]
                pred = preds[i]

                # Log it as a table row
                diff = abs(label - pred[0])
                self.logger.experiment.log({
                    'image': wandb.Image(image, caption=f"Label: {round(label, 3)}, Pred: {round(pred[0], 3)}, Diff: {round(diff, 3)}")
                })



        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, reduce_fx=torch.mean)
        return loss

    def configure_optimizers(self):
        if self.vit:
            return torch.optim.Adam(self.parameters(), lr=0.00001)
        else:
            return torch.optim.Adam(self.parameters(), lr=0.0001)

if __name__ == "__main__":
    data_module = SteeringDataModule()
    model = SteeringModel()

    # Setting precision based on user input or some logic
    use_fp16 = False  # Set this to False to use float32
    precision = 16 if use_fp16 else 32

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(project='SteeringModel', log_model=True)

    accelerator = 'mps'
    # accelerator = 'cpu'
    # Initialize the Trainer with the desired precision
    trainer = pl.Trainer(max_epochs=100, precision=precision, logger=wandb_logger, accelerator=accelerator, log_every_n_steps=10, callbacks=[ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', save_top_k=5)])

    trainer.fit(model, datamodule=data_module)