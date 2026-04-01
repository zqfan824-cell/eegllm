"""
集成Vector Quantization和重建损失的EEGLLM模型
参考NeuroLM的VQ机制，增强域对抗学习效果

关键维度说明（避免再出维度bug）：
  - 数据加载器输出:  (B, seq_len, N)     N=通道数
  - Normalize层输入:  (B, seq_len, N)
  - PatchEmbedding输入: (B, N, seq_len)   permute后
  - PatchEmbedding输出: (B*N, patches_per_var, d_model)
  - 重编程层输出:     (B*N, patches_per_var, d_llm)
  - LLM输出:         (B*N, prompt_len + patches_per_var, llm_hidden)
  - 截断到d_ff后:    (B*N, total_len, d_ff)
  - reshape后:       (B, N, d_ff, patch_nums) → ClassificationHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EEGLLM import Model as EEGLLMBase, ReverseLayerF
from utils.reconstruction_losses import NormEMAVectorQuantizer, ReconstructionLosses
import math


class ModalAlignmentScheduler:
    """模态对齐权重调度器"""
    def __init__(self, max_alpha=1.0, schedule_type='sigmoid'):
        self.max_alpha = max_alpha
        self.schedule_type = schedule_type

    def get_alpha(self, current_step, total_steps):
        progress = current_step / max(total_steps, 1)
        if self.schedule_type == 'sigmoid':
            alpha = self.max_alpha * (2.0 / (1.0 + math.exp(-10 * (progress - 0.1))) - 1.0)
            alpha = max(0.01 * self.max_alpha, alpha)
        elif self.schedule_type == 'linear':
            alpha = self.max_alpha * progress
        elif self.schedule_type == 'cosine':
            alpha = self.max_alpha * (1 - math.cos(progress * math.pi)) / 2
        else:
            alpha = self.max_alpha * progress
        return max(0.0, min(alpha, self.max_alpha))


class ModalContrastiveLearning(nn.Module):
    """模态对比学习模块"""
    def __init__(self, eeg_dim, llm_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = min(eeg_dim, llm_dim) // 2

        self.eeg_projector = nn.Sequential(
            nn.Linear(eeg_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim // 2)
        )
        self.llm_projector = nn.Sequential(
            nn.Linear(llm_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim // 2)
        )

    def forward(self, eeg_features, llm_features):
        """
        Args:
            eeg_features: [B, L, eeg_dim]
            llm_features: [B, L, llm_dim]
        """
        eeg_proj = F.normalize(self.eeg_projector(eeg_features), dim=-1)
        llm_proj = F.normalize(self.llm_projector(llm_features), dim=-1)

        B, L, D = eeg_proj.shape
        eeg_flat = eeg_proj.view(-1, D)
        llm_flat = llm_proj.view(-1, D)

        similarity = torch.matmul(eeg_flat, llm_flat.T) / self.temperature
        labels = torch.arange(B * L, device=eeg_features.device)

        if similarity.requires_grad:
            return F.cross_entropy(similarity, labels)
        else:
            return torch.tensor(0.0, device=eeg_features.device, requires_grad=True)


class EEGLLM_VQ(EEGLLMBase):
    """
    集成Vector Quantization和重建损失的EEGLLM模型

    在基类EEGLLM的classification()流程中插入VQ步骤：
      patch_embedding → [VQ编码+量化] → reprogramming → LLM → 分类头
    """

    def __init__(self, configs):
        super().__init__(configs)

        # VQ相关参数
        self.vq_enabled = getattr(configs, 'enable_vq', True)
        self.reconstruction_enabled = getattr(configs, 'enable_reconstruction', False)

        if self.vq_enabled:
            self.vq_embed_dim = getattr(configs, 'vq_embed_dim', 128)
            self.vq_n_embed = getattr(configs, 'vq_n_embed', 8192)
            self.vq_beta = getattr(configs, 'vq_beta', 1.0)

            self.quantizer = NormEMAVectorQuantizer(
                n_embed=self.vq_n_embed,
                embedding_dim=self.vq_embed_dim,
                beta=self.vq_beta
            )

            # d_model → VQ空间 → d_model
            self.vq_encoder = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.Tanh(),
                nn.Linear(configs.d_model, self.vq_embed_dim)
            )
            self.vq_to_model = nn.Linear(self.vq_embed_dim, configs.d_model)

        if self.reconstruction_enabled and self.vq_enabled:
            # 重建解码器：从VQ特征重建频域和时域信号
            # 每个通道的patches取平均后得到 (B, N, vq_embed_dim)
            # 然后解码到 (B, N, freq_dim) 和 (B, N, seq_len)
            freq_dim = configs.seq_len // 2
            raw_dim = configs.seq_len

            self.freq_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, self.vq_embed_dim),
                nn.Tanh(),
                nn.Linear(self.vq_embed_dim, freq_dim)
            )
            self.raw_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, self.vq_embed_dim),
                nn.Tanh(),
                nn.Linear(self.vq_embed_dim, raw_dim)
            )

        # 增强域分类器
        if hasattr(self, 'reprogramming_layer') and hasattr(self.reprogramming_layer, 'domain_classifier'):
            self.reprogramming_layer.domain_classifier = nn.Sequential(
                nn.Linear(self.d_llm, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)
            )

            self.modal_alignment_scheduler = ModalAlignmentScheduler(
                max_alpha=getattr(configs, 'max_alpha', 1.0),
                schedule_type=getattr(configs, 'alpha_schedule', 'sigmoid')
            )
            self.modal_contrastive = ModalContrastiveLearning(
                eeg_dim=configs.d_model,
                llm_dim=self.d_llm,
                temperature=getattr(configs, 'contrastive_temp', 0.1)
            )
            self._init_enhanced_domain_classifier()

        print(f"[EEGLLM_VQ] VQ={self.vq_enabled}, Reconstruction={self.reconstruction_enabled}")
        if self.vq_enabled:
            print(f"  VQ embed_dim={self.vq_embed_dim}, codebook_size={self.vq_n_embed}")

    def _init_enhanced_domain_classifier(self):
        for m in self.reprogramming_layer.domain_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, alpha=0.0, return_reconstruction_loss=False):
        """
        统一前向传播接口

        Args:
            x_enc: (B, seq_len, N) 输入EEG数据 — 与数据加载器输出一致
            x_mark_enc: (B, seq_len, 4) 时间标记
            alpha: 域对抗学习权重
            return_reconstruction_loss: 是否返回辅助损失
        """
        if self.task_name == 'classification':
            return self.classification_vq(
                x_enc, x_mark_enc,
                alpha=alpha,
                return_reconstruction_loss=return_reconstruction_loss
            )
        else:
            # 非分类任务回退到基类（不使用VQ）
            return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def classification_vq(self, x_enc, x_mark_enc, alpha=0.0, return_reconstruction_loss=False):
        """
        带VQ的分类前向传播

        数据流:
          (B, seq_len, N)
          → normalize
          → permute → (B, N, seq_len)
          → reshape → (B*N, 1, seq_len)  → 统计信息 + prompt
          → restore → (B, N, seq_len) → permute → (B, N, seq_len)
          → patch_embedding → (B*N, patches_per_var, d_model)
          → [VQ] → (B*N, patches_per_var, d_model)
          → reprogramming → (B*N, patches_per_var, d_llm)
          → cat with prompt → LLM → truncate to d_ff
          → reshape → (B, N, d_ff, patch_nums) → ClassificationHead
        """
        # 保存原始输入用于重建损失
        x_enc_for_recon = x_enc.clone()  # (B, seq_len, N)

        # === Step 1: 标准化 ===
        x_enc = self.normalize_layers(x_enc, 'norm')  # (B, seq_len, N)

        B, T, N = x_enc.size()

        # === Step 2: 生成prompt（与基类classification()逻辑一致）===
        # 先reshape为 (B*N, T, 1) 来计算统计信息
        x_for_stats = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_for_stats, dim=1)[0]
        max_values = torch.max(x_for_stats, dim=1)[0]
        medians = torch.median(x_for_stats, dim=1).values
        trends = x_for_stats.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_for_stats.shape[0]):
            min_v = str(min_values[b].tolist()[0])
            max_v = str(max_values[b].tolist()[0])
            med_v = str(medians[b].tolist()[0])
            trend_str = 'increasing' if trends[b] > 0 else 'decreasing'

            if N == 32:  # DEAP
                prompt_ = (
                    f"<|start_prompt|>Dataset: DEAP 32-channel EEG for emotion recognition. "
                    f"Task: binary emotion classification (valence). "
                    f"Input stats: min {min_v}, max {max_v}, median {med_v}, trend {trend_str}. "
                    f"Analyze EEG patterns for emotional state.<|end_prompt|>"
                )
            elif N == 62:  # SEED
                prompt_ = (
                    f"<|start_prompt|>Dataset: SEED 62-channel EEG for emotion recognition. "
                    f"Task: three-class emotion classification (positive/neutral/negative). "
                    f"Input stats: min {min_v}, max {max_v}, median {med_v}, trend {trend_str}. "
                    f"Analyze EEG patterns for emotional state.<|end_prompt|>"
                )
            else:
                prompt_ = (
                    f"<|start_prompt|>Dataset: {N}-channel EEG for emotion classification. "
                    f"Input stats: min {min_v}, max {max_v}, median {med_v}, trend {trend_str}. "
                    f"Classify emotional state.<|end_prompt|>"
                )
            prompt.append(prompt_)

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_tokens.to(x_enc.device)
        )  # (B*N, prompt_len, d_llm)

        # === Step 3: 词嵌入映射 ===
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)  # (num_tokens, d_llm)

        # === Step 4: Patch Embedding ===
        x_enc_perm = x_enc.permute(0, 2, 1).contiguous()  # (B, N, T)
        enc_out, n_vars = self.patch_embedding(x_enc_perm.float())
        # enc_out: (B*N, patches_per_var, d_model)
        patches_per_var = enc_out.shape[1]

        # === Step 5: VQ编码和量化 ===
        vq_loss = torch.tensor(0.0, device=x_enc.device)
        if self.vq_enabled:
            vq_features = self.vq_encoder(enc_out)  # (B*N, patches_per_var, vq_embed_dim)
            quantized, vq_loss, _ = self.quantizer(vq_features)
            enc_out = self.vq_to_model(quantized)  # (B*N, patches_per_var, d_model)

        # === Step 6: 重建损失 ===
        reconstruction_loss = torch.tensor(0.0, device=x_enc.device)
        if self.reconstruction_enabled and return_reconstruction_loss and self.vq_enabled:
            reconstruction_loss = self._compute_reconstruction_loss(
                quantized, x_enc_for_recon, B, N
            )

        # === Step 7: 重编程 ===
        if return_reconstruction_loss:
            enc_out, domain_loss = self.reprogramming_layer(
                enc_out, source_embeddings, source_embeddings,
                alpha=alpha, return_domain_loss=True
            )
        else:
            enc_out = self.reprogramming_layer(
                enc_out, source_embeddings, source_embeddings,
                alpha=alpha, return_domain_loss=False
            )
            domain_loss = torch.tensor(0.0, device=x_enc.device)

        # === Step 8: 拼接prompt + LLM处理 ===
        llm_input = torch.cat([prompt_embeddings, enc_out], dim=1)
        # llm_input: (B*N, prompt_len + patches_per_var, d_llm)

        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out: (B*N, prompt_len + patches_per_var, d_ff)

        # === Step 9: 对比学习损失 ===
        contrastive_loss = torch.tensor(0.0, device=x_enc.device)
        if hasattr(self, 'modal_contrastive') and self.training and return_reconstruction_loss:
            # 从dec_out中取后patches_per_var个token作为LLM特征
            llm_feat = dec_out[:, -patches_per_var:, :]  # (B*N, patches_per_var, d_ff)
            # 将enc_out（重编程后）截断到d_ff维度做对比
            eeg_feat = enc_out[:, :, :self.d_ff]  # (B*N, patches_per_var, d_ff)
            # 取batch中一小部分来做对比学习（避免B*N太大导致OOM）
            sample_size = min(B, eeg_feat.shape[0])
            contrastive_loss = self.modal_contrastive(
                eeg_feat[:sample_size], llm_feat[:sample_size]
            )

        # === Step 10: reshape到分类头期望的格式 ===
        # dec_out: (B*N, prompt_len + patches_per_var, d_ff)
        dec_out = torch.reshape(
            dec_out, (B, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )  # (B, N, prompt_len + patches_per_var, d_ff)

        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # (B, N, d_ff, prompt_len + patches_per_var)

        dec_out = dec_out[:, :, :, -self.patch_nums:]
        # (B, N, d_ff, patch_nums)

        output = self.output_projection(dec_out)
        # output: (B, num_class)

        if return_reconstruction_loss:
            return output, {
                'vq_loss': vq_loss,
                'domain_loss': domain_loss,
                'contrastive_loss': contrastive_loss,
                'reconstruction_loss': reconstruction_loss,
            }
        else:
            return output

    def _compute_reconstruction_loss(self, quantized_features, x_original, B, N):
        """
        计算重建损失

        Args:
            quantized_features: (B*N, patches_per_var, vq_embed_dim) VQ量化后的特征
            x_original: (B, seq_len, N) 原始输入
            B: batch size
            N: 通道数
        """
        # 将VQ特征reshape为 (B, N, patches_per_var, vq_embed_dim) 后取平均
        patches_per_var = quantized_features.shape[1]
        vq_feat = quantized_features.view(B, N, patches_per_var, -1)
        vq_pooled = vq_feat.mean(dim=2)  # (B, N, vq_embed_dim)

        # 原始数据转为 (B, N, seq_len) 格式
        x_orig = x_original.permute(0, 2, 1).contiguous()  # (B, N, seq_len)

        # 频域目标
        x_fft = torch.fft.fft(x_orig, dim=2)
        freq_target = torch.abs(x_fft)[:, :, :x_orig.shape[2] // 2]  # (B, N, seq_len//2)

        # 重建预测
        freq_pred = self.freq_decoder(vq_pooled)  # (B, N, seq_len//2)
        raw_pred = self.raw_decoder(vq_pooled)     # (B, N, seq_len)

        freq_loss = F.mse_loss(freq_pred, freq_target)
        raw_loss = F.mse_loss(raw_pred, x_orig)

        return freq_loss + raw_loss
