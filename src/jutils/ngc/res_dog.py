import os
from torch import optim, nn, utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop, ToTensor, Normalize, Compose, Resize
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import AE


def main():
    # init the autoencoder
    autoencoder = AE(224)

    # setup data
    transform = Compose([
        Resize(256),
        RandomResizedCrop(224, ),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder('/ho3d/HO3D/train/', transform)
    train_loader = utils.data.DataLoader(dataset)


    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        default_root_dir='/raid/dog/',
        gpus=[0],
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    

if __name__ == '__main__':
    main()