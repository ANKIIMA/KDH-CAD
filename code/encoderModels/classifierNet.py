import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .qwen_model import QwenEncoder


class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, nhead=8, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dtype=dtype, device=device)
        self.norm = nn.LayerNorm(d_model, dtype=dtype, device=device)

    def forward(self, query, key, value, key_padding_mask=None):
        # query: [Batch, Q_Seq, Dim]
        # key/value: [Batch, KV_Seq, Dim]
        attn_output, _ = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        # 残差连接与 LayerNorm
        return self.norm(attn_output + query)


class PrototypeShiftClassifier(nn.Module):
    """KDH-CAD classifier: frozen VLM encoder + cross-attention + shift vector.

    The image token sequence attends to the frozen textbook prototypes
    (cross-attention, KV = base prototypes without shift); the learnable
    shift vector is applied to the prototypes only at the final similarity
    stage.
    """

    def __init__(self, args, class_names):
        super().__init__()
        self.args = args
        self.n_classes = len(class_names)
        self.device = args.device
        self.dtype = args.dtype

        # 1. 初始化 Encoder
        if args.model_type == "qwen":
            self.encoder = QwenEncoder(args.qwen_path, args.dtype, args.device)
            self.model_dim = self.encoder._get_hidden_dim()
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        # 2. 加载教材知识库（均值原型）
        self._init_knowledge_bank(args, class_names)

        # 3. 结构构建
        # Shift Vector
        if args.use_shift == "yes":
            self.shift_vector = nn.Parameter(torch.zeros(self.n_classes, self.model_dim, dtype=self.dtype, device=self.device))
            with torch.no_grad():
                proto_std = self.base_prototypes.std()
                nn.init.normal_(self.shift_vector, std=proto_std * 0.5)
            if getattr(args, "use_shift_scale", "no") == "yes":
                self.shift_scale = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=self.dtype))
            else:
                self.register_buffer("shift_scale", torch.tensor(1.0, device=self.device, dtype=self.dtype))
        else:
            self.register_buffer("shift_vector", torch.zeros(self.n_classes, self.model_dim, dtype=self.dtype, device=self.device))
            self.register_buffer("shift_scale", torch.tensor(1.0, device=self.device, dtype=self.dtype))

        self.cross_attn = CrossAttentionModule(self.model_dim, dtype=self.dtype, device=self.device)
        self.cross_logit_head = nn.Linear(self.model_dim, 1, bias=True).to(self.device, self.dtype)

    def _init_knowledge_bank(self, args, class_names):
        all_embs = []
        for cls in class_names:
            path_pt = os.path.join(args.embed_base_path, cls, f"{cls}.bundle.pt")
            try:
                data = torch.load(path_pt, map_location='cpu', weights_only=False)
                emb = data.get('embedding').to(self.dtype)  # 可能是 [Seq, Dim] 或 [1, Seq, Dim]
                if emb.ndim == 3:
                    emb = emb.squeeze(0)
                all_embs.append(emb)
            except Exception:
                # 兜底：如果读取失败，给一个随机向量
                print(f"err reading pt file {path_pt}, using Random.")
                all_embs.append(torch.randn(1, self.model_dim, dtype=self.dtype))

        means = [e.mean(dim=0) if e.ndim >= 2 else e for e in all_embs]
        self.register_buffer("base_prototypes", torch.stack(means).to(self.device))

    def pad_image_sequences(self, seq_list):
        max_len = max([s.shape[0] for s in seq_list])
        padded_seqs, masks = [], []
        for s in seq_list:
            cur_len = s.shape[0]
            pad_len = max_len - cur_len
            padded = F.pad(s, (0, 0, 0, pad_len))
            mask = torch.cat([torch.zeros(cur_len), torch.ones(pad_len)]).bool()
            padded_seqs.append(padded)
            masks.append(mask)
        return torch.stack(padded_seqs).to(self.device), torch.stack(masks).to(self.device)

    def forward(self, images_batch):
        # 1. 编码图像
        img_means_raw, img_lhs_list = self.encoder(images_batch)
        img_means_raw = img_means_raw.clone().to(self.dtype)
        batch_size = img_means_raw.shape[0]

        current_prototypes = (self.base_prototypes + self.shift_vector * self.shift_scale).to(self.dtype)

        def get_sim_comparison(feat):
            f_norm = F.normalize(feat, p=2, dim=-1)
            p_base_norm = F.normalize(self.base_prototypes, p=2, dim=-1)
            p_shift_norm = F.normalize(current_prototypes, p=2, dim=-1)
            sim_base = torch.mm(f_norm, p_base_norm.t())
            sim_shift = torch.mm(f_norm, p_shift_norm.t())
            return {"sim_base": sim_base.detach(), "sim_shift": sim_shift.detach()}

        # 2. 准备 Query: 图像序列 [Batch, Seq_Img, Dim]
        img_seq, img_mask = self.pad_image_sequences(img_lhs_list)
        img_seq = img_seq.to(self.dtype)

        # 3. 准备 KV: 使用固定的教材原型 (不带 Shift)
        proto_kv = self.base_prototypes.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. 执行交叉注意力: Image (Q) Attend to Textbook (KV)
        refined_img_seq = self.cross_attn(img_seq, proto_kv, proto_kv)

        # 5. 聚合图像特征 (Global Average Pooling)
        valid_mask = (~img_mask).float().unsqueeze(-1)  # [Batch, Seq_Img, 1]
        refined_img_means = ((refined_img_seq * valid_mask).sum(dim=1) /
                             valid_mask.sum(dim=1).clamp(min=1e-9)).to(self.dtype)

        # 6. 仅在计算相似度时引入 shift_vector
        dynamic_prototypes = self.base_prototypes + self.shift_vector * self.shift_scale

        img_norm = F.normalize(refined_img_means, p=2, dim=-1)
        proto_norm = F.normalize(dynamic_prototypes, p=2, dim=-1)

        similarity = torch.mm(img_norm, proto_norm.t())

        comp_data = get_sim_comparison(refined_img_means)

        return similarity * 10.0, refined_img_means, similarity, comp_data
