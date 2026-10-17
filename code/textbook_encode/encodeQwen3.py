import torch
import os
import xml.etree.ElementTree as ET
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache
from PIL import Image
import re
import json
from loguru import logger

class Qwen3Encoder:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        logger.info(f"🚀 正在加载模型: {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def parse_xml_interleaved(self, xml_path, img_dir):
        """解析 XML 获取交织的文本和图片路径"""
        segments = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    segments.append({"type": "text", "content": elem.text.strip()})
                if elem.tag == 'graphic' and 'src' in elem.attrib:
                    fname = os.path.basename(elem.attrib['src'])
                    abs_path = os.path.join(img_dir, fname)
                    segments.append({"type": "image", "path": abs_path})
                if elem.tail and elem.tail.strip():
                    segments.append({"type": "text", "content": elem.tail.strip()})
        except Exception as e:
            logger.error(f"❌ XML 解析错误 {xml_path}: {e}")
        return segments

    def extract_ekt(self, class_name, video_paths, xml_paths, img_dir):
        """
        阶段 1：知识过滤与重组 (Pre-process)
        按照论文定义提取 Engineering Knowledge Triplet (EKT)
        """
        content = []
        
        # Prompt (P_guide)
        ekt_prompt = (
            f"You are an expert in mechanical engineering. Based on the provided videos and technical references, "
            f"extract the following information for the part class: 【{class_name}】.\n"
            f"Filter out irrelavent content and redundancies, then reorganize the information into the following structure:\n"
            f"- category: Canonical terminology.\n"
            f"- function: Mechanical roles and applications.\n"
            f"- visual: Typical shapes and visual identifiers.\n"
            f"- representative_file: Choose the [Image Path] that best represents this part.\n\n"
            f"Strictly output JSON with these 4 keys."
        )

        # 1. 组装输入 I (Interleaved Input)
        content.append({"type": "text", "text": "Input Assets:\n"})
        for v_path in video_paths:
            content.append({"type": "video", "video": v_path})
        
        for x_path in xml_paths:
            segments = self.parse_xml_interleaved(x_path, img_dir)
            for seg in segments:
                if seg["type"] == "text":
                    content.append({"type": "text", "text": seg["content"]})
                elif seg["type"] == "image" and os.path.exists(seg["path"]):
                    fname = os.path.basename(seg["path"])
                    content.append({"type": "text", "text": f"\n[Image Path: {fname}]\n"})
                    content.append({"type": "image", "image": Image.open(seg["path"]).convert("RGB")})

        content.append({"type": "text", "text": f"\n{ekt_prompt}"})
        
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)

        # 2. 调用 VLM 进行推理 (Phi_VLM)
        logger.info(f"[KDH-CAD] 正在为 {class_name} 提取 EKT 三元组...")
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048, # 结构化输出可能较长
                do_sample=False,     # 保持输出稳定性
            )
        
        # 3. 解析 JSON 输出
        raw_output = self.processor.batch_decode(
            gen_outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        # 清洗可能存在的 Markdown 代码块标记
        json_str = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_str:
            try:
                ekt_data = json.loads(json_str.group())
                ref_file = ekt_data.get('representative_file')
                if ref_file:
                    ekt_data['representative_file'] = os.path.join(img_dir, ref_file)
                return ekt_data
            except json.JSONDecodeError:
                logger.error("JSON 解析失败，返回原始文本")
        
        return {"raw_text": raw_output}

    def encode_knowledge(self, class_name, ekt_data, video_paths, xml_paths, img_dir):
        """
        阶段 1：编码多模态信息
        返回: (past_key_values, response, embedding)
        """
        content = []

        if video_paths:
            content.append({"type": "text", "text": "Visual Reference Videos:\n"})
            for v_path in video_paths:
                content.append({"type": "video", "video": v_path})

        content.append({"type": "text", "text": "\nDetailed Technical Images:\n"})
        for x_path in xml_paths:
            segments = self.parse_xml_interleaved(x_path, img_dir)
            for seg in segments:
                if seg["type"] == "image" and os.path.exists(seg["path"]):
                    try:
                        content.append({"type": "image", "image": Image.open(seg["path"]).convert("RGB")})
                    except Exception as e:
                        logger.warning(f"无法加载图片 {seg['path']}: {e}")

        # 基本数据块
        knowledge_text = (
            f"\n[Engineering Knowledge Summary for {class_name}]\n"
            f"Terminology: {ekt_data.get('category', class_name)}\n"
            f"Function description: {ekt_data.get('function', '')}\n"
            f"Visual Features: {ekt_data.get('visual', '')}\n"
            f"\nInstruct: Encode the above multi-modal evidence into a unified mechanical concept representation."
        )

        content.append({"type": "text", "text": knowledge_text})

        
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)

        input_len = inputs.input_ids.shape[1]
        past_key_values = DynamicCache()

        # 2. 文本生成 (阶段 1 Response)
        logger.info(f"正在为 {class_name} 生成Embedding 与 KV Cache...")
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                past_key_values=past_key_values,
                use_cache=True,
            )
        response = self.processor.batch_decode(gen_outputs[:, input_len:], skip_special_tokens=True)[0]

        # 3. 计算 Embedding (Last Hidden State 的平均值)
        # 注意：这里重新跑一次 forward 且开启 output_hidden_states
        with torch.no_grad():
            # 我们只需要原始 prompt 的 hidden states
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            # 获取最后一层隐藏状态: [batch, sequence_length, hidden_size]
            last_hidden_state = outputs.hidden_states[-1]
            # 计算序列维度的平均值 (Mean Pooling) -> [batch, hidden_size]
            embedding = last_hidden_state.mean(dim=1).squeeze(0).to(torch.bfloat16)

        return past_key_values, response, embedding

    @staticmethod
    def inference_with_cache(model, processor, past_key_values, query, device="cuda"):
        """阶段 2：基于 KV Cache 的增量推理"""
        past_length = past_key_values.get_seq_length()
        text_to_append = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs_turn_2 = processor.tokenizer(
            text_to_append, return_tensors="pt", add_special_tokens=False
        ).to(device)
        
        curr_length = inputs_turn_2.input_ids.shape[1]
        cache_position = torch.arange(past_length, past_length + curr_length, device=device)
        position_ids = cache_position.unsqueeze(0) 
        attention_mask = torch.ones((1, past_length + curr_length), device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs_turn_2.input_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=position_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                use_cache=True,
            )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]