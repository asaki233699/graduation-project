#!/usr/bin/env python3
# ============================================================
# Phase 3-2: SigLIP 语义蒸馏 Loss 模块
# 核心: 以 SigLIP 为"视觉语义教师", 引导 SVTR 学习光照无关特征
#
# 论文公式: L_total = L_rec + lambda * L_distill
#   L_distill = 1 - cosine_sim(SVTR_feat, SigLIP_feat)
#
# 用法 (伪代码):
#   from siglip_distill_loss import SigLIPDistillationLoss
#   distill_loss_fn = SigLIPDistillationLoss(siglip_model, temp=0.07)
#   ...
#   loss_rec = rec_head(student_feat, labels)
#   loss_distill = distill_loss_fn(student_feat, images)
#   loss = loss_rec + lambda_distill * loss_distill
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPDistillationLoss(nn.Module):
    """
    SigLIP 语义蒸馏 Loss
    
    参数:
        siglip_model: 预训练的 SigLIP 模型 (frozen, no grad)
        temperature: 蒸馏温度 (越小越"硬", 默认 0.07 参考 CLIP)
        pool_method: 学生特征池化方式 ('mean' 或 'attention')
    """
    
    def __init__(self, siglip_model, temperature=0.07, pool_method='mean'):
        super().__init__()
        self.siglip = siglip_model
        self.temperature = temperature
        self.pool_method = pool_method
        
        # 冻结 SigLIP, 不参与梯度更新
        for p in self.siglip.parameters():
            p.requires_grad = False
        self.siglip.eval()
        
        # 如果使用 attention pooling, 加一个可学习的池化层
        if pool_method == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(768, 256),  # 假设学生特征 dim=768
                nn.Tanh(),
                nn.Linear(256, 1)
            )
    
    def _pool_student_feat(self, student_feat):
        """
        将学生特征 [B, N, D] 池化为 [B, D]
        N = patch 数量 (如 SVTR 切成 8x32=256 个 patch)
        """
        if self.pool_method == 'mean':
            return student_feat.mean(dim=1)  # [B, D]
        elif self.pool_method == 'attention':
            attn_weights = self.attn_pool(student_feat).squeeze(-1)  # [B, N]
            attn_weights = F.softmax(attn_weights, dim=-1)           # [B, N]
            return (student_feat * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
    
    def forward(self, student_feat, images):
        """
        参数:
            student_feat: SVTR backbone 输出特征, shape [B, N, D]
            images: 原始图片, shape [B, 3, H, W] (用于 SigLIP 提取特征)
        
        返回:
            loss: 标量, 蒸馏损失
            metrics: dict, 额外的统计信息 (可选, 用于 logging)
        """
        # 1. 学生特征池化
        feat_s = self._pool_student_feat(student_feat)  # [B, D]
        feat_s = F.normalize(feat_s, p=2, dim=-1)
        
        # 2. 教师特征 (SigLIP, no grad)
        with torch.no_grad():
            # SigLIP 的 get_image_features 返回 [B, D]
            feat_t = self.siglip.get_image_features(images).pooler_output
            feat_t = F.normalize(feat_t, p=2, dim=-1)
        
        # 3. 余弦相似度 → 蒸馏损失
        # cosine_sim = feat_s · feat_t^T  的对角线 (自身对应)
        cosine_sim = (feat_s * feat_t).sum(dim=-1)  # [B]
        
        # L_distill = (1/|B|) * sum(1 - cos_sim)
        # 当学生和教师特征完全一致: cos_sim=1 → loss=0
        # 当学生和教师特征正交/相反: cos_sim→-1 → loss→2
        loss = (1.0 - cosine_sim).mean()
        
        # 额外统计
        metrics = {
            'cosine_sim_mean': cosine_sim.mean().item(),
            'cosine_sim_std': cosine_sim.std().item(),
        }
        
        return loss, metrics


class ContrastiveDistillLoss(nn.Module):
    """
    对比蒸馏 Loss (增强版)
    
    相比直接 cosine loss, 对比 loss 不仅拉近正样本对,
    还推开 batch 内的负样本对 (不同的图片-文本对)
    
    适合 batch size 较大的训练场景
    """
    
    def __init__(self, siglip_model, temperature=0.07):
        super().__init__()
        self.siglip = siglip_model
        self.temperature = temperature
        
        for p in self.siglip.parameters():
            p.requires_grad = False
        self.siglip.eval()
    
    def forward(self, student_feat, images):
        """
        参数同上 SigLIPDistillationLoss
        使用 InfoNCE 风格的对比损失
        """
        B = student_feat.shape[0]
        
        # 学生特征池化 + 归一化
        feat_s = student_feat.mean(dim=1)  # [B, D]
        feat_s = F.normalize(feat_s, p=2, dim=-1)
        
        # 教师特征
        with torch.no_grad():
            feat_t = self.siglip.get_image_features(images).pooler_output
            feat_t = F.normalize(feat_t, p=2, dim=-1)
        
        # InfoNCE: 正样本对角线, 负样本为 batch 内其他
        logits = feat_s @ feat_t.T / self.temperature  # [B, B]
        labels = torch.arange(B, device=logits.device)
        
        # 对称损失: S→T + T→S
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        loss = (loss_s2t + loss_t2s) / 2
        
        return loss, {'contrastive_loss': loss.item()}


# ============================================================
# 自检: 跑一下确认模块没问题
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SigLIP Distillation Loss 模块自检")
    print("=" * 60)
    
    from transformers import AutoModel
    
    # 加载 SigLIP (frozen)
    print("\n加载 SigLIP 模型...")
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    
    # 创建蒸馏 loss
    distill_loss = SigLIPDistillationLoss(siglip, temperature=0.07)
    print(f"蒸馏 Loss 创建成功!")
    print(f"  温度: {distill_loss.temperature}")
    print(f"  池化方式: {distill_loss.pool_method}")
    print(f"  SigLIP 参数冻结: {all(not p.requires_grad for p in siglip.parameters())}")
    
    # 模拟一批数据
    B, N, D = 4, 256, 768  # batch=4, 256 patches, dim=768
    student_feat = torch.randn(B, N, D)
    images = torch.randn(B, 3, 224, 224)
    
    # 禁止 grad (模拟推理)
    with torch.no_grad():
        loss, metrics = distill_loss(student_feat, images)
    
    print(f"\n模拟前向传播:")
    print(f"  学生特征: {student_feat.shape}")
    print(f"  图片: {images.shape}")
    print(f"  蒸馏 loss: {loss.item():.4f}")
    print(f"  cosine_sim: {metrics['cosine_sim_mean']:.4f}")
    
    print("\n✓ 模块自检通过!")
    print("=" * 60)
