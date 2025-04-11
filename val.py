import os
import torch
import numpy as np
from PIL import Image

def calculate_metrics(seg_folder, gt_folder):
    smooth = 1e-5
    seg_files = os.listdir(seg_folder)
    gt_files = os.listdir(gt_folder)

    dice_scores = []
    iou_scores = []
    mae_scores = []
    acc_scores = []

    for seg_file in seg_files:
        if seg_file.endswith(".png"):
            base_name, _ = os.path.splitext(seg_file)
            gt_file = f"{base_name}.png"  # 假设真实标签文件名与分割结果文件名对应

            if gt_file in gt_files:
                seg_path = os.path.join(seg_folder, seg_file)
                gt_path = os.path.join(gt_folder, gt_file)

                # 读取分割结果和GT图像
                seg_image = Image.open(seg_path).convert("L")
                gt_image = Image.open(gt_path).convert("L")

                # 将图像转换为numpy数组
                seg_np = np.array(seg_image) / 255.0
                gt_np = np.array(gt_image) / 255.0

                # 将概率输出变为与标签相匹配的矩阵
                seg_np[seg_np > 0.5] = 1
                seg_np[seg_np <= 0.5] = 0

                seg_tensor = torch.Tensor(seg_np).unsqueeze(0)
                gt_tensor = torch.Tensor(gt_np).unsqueeze(0)

                # 计算交集和并集
                intersection = (seg_tensor * gt_tensor).sum()
                union = (seg_tensor + gt_tensor).sum() - intersection

                # 计算Dice和IoU
                dice_score = (2. * intersection + smooth) / (seg_tensor.sum() + gt_tensor.sum() + smooth)
                iou_score = (intersection + smooth) / (union + smooth)

                # 计算MAE和Accuracy
                mae_score = torch.abs(seg_tensor - gt_tensor).mean()
                acc_score = (seg_tensor == gt_tensor).sum().item() / seg_tensor.numel()

                dice_scores.append(dice_score.item())
                iou_scores.append(iou_score.item())
                mae_scores.append(mae_score.item())
                acc_scores.append(acc_score)

    return {
        'average_dice': sum(dice_scores) / len(dice_scores),
        'average_iou': sum(iou_scores) / len(iou_scores),
        'average_mae': sum(mae_scores) / len(mae_scores),
        'average_acc': sum(acc_scores) / len(acc_scores)
    }

# 示例用法
seg_folder = "/opt/data/private/jyp/CamoDiffusion-main/out_net4_5/ETIS-LaribPolypDB/CAMO"  # 替换为您的分割结果文件夹路径
gt_folder = "/opt/data/private/jyp/data/TestDataset/ETIS-LaribPolypDB/masks"  # 替换为对应的GT文件夹路径

metrics = calculate_metrics(seg_folder, gt_folder)
print(f"平均Dice系数: {metrics['average_dice']}")
print(f"平均IoU: {metrics['average_iou']}")
print(f"平均MAE: {metrics['average_mae']}")
print(f"平均准确度: {metrics['average_acc']}")
