import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc as calculate_auc
from Scheduler.LinearDecayScheduler import LinearDecayScheduler
from model.backbone import ViT
import os

class VITModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ViT(image_size=64,image_patch_size=8,frames=64,frame_patch_size=8,num_classes=2,channels=1,depth=6,dim=1024,heads=8,mlp_dim=2048,emb_dropout=0.1).to("cuda:1")

        self.loss = nn.CrossEntropyLoss()
        self.prediction = []
        self.gt = []
        self.sum_loss = []
        self.train_prediction = []
        self.train_gt = []        


    def forward(self,data):
        outputs = self.model(data)
        return outputs
        # 修改 training_step 方法以收集预测结果：

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        target = target.squeeze(-1).to(self.device)  # 确保目标张量在正确的设备上
        output = self(inputs.to(torch.float32).to(self.device))
        loss = self.loss(output, target)
        self.sum_loss.append(loss)
        self.log('train/loss', loss, sync_dist=True, prog_bar=True)

        # 收集预测结果
        preds = torch.argmax(output, dim=1).cpu()  # 
        self.train_prediction.extend(preds.cpu().numpy()) 
        self.train_gt.extend(target.cpu().numpy())  #
        # print('train_prediction',self.train_prediction)
        # print('train_target',self.train_gt)
        return loss
    
    def on_train_epoch_end(self):
        accuracy = accuracy_score(self.train_gt,self.train_prediction)
        precision = precision_score(self.train_gt,self.train_prediction,zero_division=1)
        recall = recall_score(self.train_gt,self.train_prediction)
        f1 = f1_score(self.train_gt,self.train_prediction)
        auc = roc_auc_score(self.train_gt,self.train_prediction)
        self.log("epoch_loss",torch.tensor(self.sum_loss).mean().item(),on_epoch=True)
        self.sum_loss.clear()
                # 计算混淆矩阵
        cm = confusion_matrix(self.train_gt, self.train_prediction)
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(self.train_gt, self.train_prediction)
        roc_auc = calculate_auc(fpr, tpr)  # 注意：这里使用 sklearn.metrics.auc 函数
        
        # 检查 AUC 值是否等于 ROC 曲线下的面积
        if auc != roc_auc:
            raise ValueError("AUC value is not equal to the area under the ROC curve. Stopping the program.")
        # 清空预测和真实标签列表
        self.train_prediction.clear()
        self.train_gt.clear()

        # 将图像保存到指定目录
        output_dir = "/home/10jay/Vit-program/output/train1"
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练集的混淆矩阵图像
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 12, "color": "black"}, cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        title = "Confusion Matrix\n"
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                title += f"{cm[i][j]} "
            title += "\n"
        plt.title(title.strip())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"train_confusion_matrix_epoch_{self.current_epoch}.png"))
        plt.clf()

        # 保存训练集的ROC曲线图像
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"train_ROC_curve_epoch_{self.current_epoch}.png"))
        plt.clf()

        # 记录其他指标
        self.log("train/acc", accuracy, sync_dist=True)
        self.log("train/pre", precision, sync_dist=True)
        self.log("train/f1", f1, sync_dist=True)
        self.log("train/auc", auc, sync_dist=True)
        self.log("train/recall", recall, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        target = target.squeeze(-1)
        output = self(inputs.to(torch.float32))
        loss = self.loss(output, target)
        self.prediction.extend(torch.argmax(output,dim=1).cpu())
        self.gt.extend(target.cpu())
        # print('prediction',self.prediction)
        # print('target',self.gt)
        self.log('val/loss', loss,sync_dist=True)
        return loss  
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = LinearDecayScheduler(optimizer, start_epoch=30, end_epoch=400, start_lr=1e-2, end_lr=1e-3)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def on_validation_epoch_end(self):
        val_targets = torch.tensor(self.gt).view(-1).numpy()
        val_predictions = torch.tensor(self.prediction).numpy()
        # print('val_targets',val_targets)
        # print('val_predictions',val_predictions)
        # 计算其他指标
        accuracy = accuracy_score(val_targets, val_predictions)
        precision = precision_score(val_targets, val_predictions, zero_division=1)
        recall = recall_score(val_targets, val_predictions)
        f1 = f1_score(val_targets, val_predictions)
        auc = roc_auc_score(val_targets, val_predictions)
        
        # 计算混淆矩阵
        cm = confusion_matrix(val_targets, val_predictions)
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(val_targets, val_predictions)
        roc_auc = calculate_auc(fpr, tpr)  # 注意：这里使用 sklearn.metrics.auc 函数
        # 检查 AUC 值是否等于 ROC 曲线下的面积
        if auc != roc_auc:
            raise ValueError("AUC value is not equal to the area under the ROC curve. Stopping the program.")
        

        # 清空预测和真实标签列表
        self.prediction.clear()
        self.gt.clear()


        output_dir = "/home/10jay/Vit-program/output/val1"
        os.makedirs(output_dir, exist_ok=True)

        # 保存验证集的混淆矩阵图像
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 12, "color": "black"}, cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        title = "Confusion Matrix\n"
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                title += f"{cm[i][j]} "
            title += "\n"
        plt.title(title.strip())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"val_confusion_matrix_epoch_{self.current_epoch}.png"))
        plt.clf()

        # 保存验证集的ROC曲线图像
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"val_ROC_curve_epoch_{self.current_epoch}.png"))
        plt.clf()

        # 记录其他指标
        self.log("val/acc", accuracy, sync_dist=True)
        self.log("val/pre", precision, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.log("val/auc", auc, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
