from lightning import Trainer
# from monai.networks.nets import ViT
from dataset.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from model.VIT444 import VITModel


train_dataset = CustomDataset("/home/10jay/project-medical/Vit-program/data","train")
val_dataset = CustomDataset("/home/10jay/project-medical/Vit-program/data","val")
train_dataloader = DataLoader(train_dataset, batch_size=16,drop_last=False, shuffle=True)
val_datalaoder = DataLoader(val_dataset, batch_size=16,drop_last=False, shuffle=False)
model = VITModel()
trainer = Trainer(devices="auto", max_epochs=1000)
trainer.fit(model, train_dataloader, val_datalaoder)
