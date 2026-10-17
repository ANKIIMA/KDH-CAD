import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText

class QwenEncoder(nn.Module):
    def __init__(self, model_name, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        print(f"[*] Loading Qwen3-VL from: {model_name}")
        # 参考 encodeQwen3.py，使用 AutoModelForImageTextToText 加载
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            device_map=device, 
            trust_remote_code=True
        ).eval()

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images_batch):
        """
        Args:
            images_batch: List[List[PIL.Image]] (batch_size, num_views)
        优化：支持真正的 Batch 并行推理
        """
        # 1. 构建 Batch 级别的消息列表
        batch_messages = []
        for images_list in images_batch:
            # 单个样本的 content
            content = []
            for img in images_list:
                content.append({"type": "image", "image": img})
            
            content.append({
                "type": "text", 
                "text": "Observe the geometric primitives and topology of this mechanical part. Extract high-level features."
            })
            batch_messages.append([{"role": "user", "content": content}])

        # 2. 批量预处理 (Padding 是关键)
        # 注意：padding=True 会自动处理不同长度的序列
        inputs = self.processor.apply_chat_template(
            batch_messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt", 
            return_dict=True,
            padding=True 
        ).to(self.device)

        attention_mask = inputs['attention_mask']

        # 3. 并行前向传播
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # last_hidden_state: [Batch, Max_Seq_Len, Dim]
            last_hidden_state = outputs.hidden_states[-1]
            
            # 4. 向量化 Masked Mean Pooling
            # input_mask_expanded: [Batch, Max_Seq_Len, Dim]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 计算 Sum 和 Count
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) # [Batch, Dim]
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)          # [Batch, Dim]
            
            # 全局均值特征 [Batch, Dim]
            batch_means = (sum_embeddings / sum_mask).to(self.dtype)

            # 5. 处理全序列 (兼容性处理)
            # 因为做了 padding，需要把 padding 部分去掉，拆回 List 形式返回给下游
            # 否则下游的 pad_image_sequences 可能会再次 padding 导致混乱
            batch_full_lhs = []
            for i in range(last_hidden_state.size(0)):
                #只取 mask 为 1 的部分 (有效 token)
                valid_len = attention_mask[i].sum().int()
                # 截取有效长度: [Seq_Len_i, Dim]
                valid_seq = last_hidden_state[i, :valid_len, :].to(self.dtype) 
                batch_full_lhs.append(valid_seq)

        return batch_means, batch_full_lhs

    def _get_hidden_dim(self):
        # 自动获取隐藏层维度
        config = self.model.config
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        return config.text_config.hidden_size