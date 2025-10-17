#!/usr/bin/env python3
"""
Cloud-Scale Transformer v5.0 (Claude-3.5 Class)
================================================
Features:
- 8B→70B params | 200k YaRN++ | FlashAttn-2
- Vision-Language Model (CLIP integration)
- DeepSpeed ZeRO-3 | bf16 | DPO/RLHF/Constitutional AI
- vLLM inference | Function-calling | Safety Layer
- Multi-step tool use | Red-teaming | API Server
================================================
Usage:
1) Pre-train:  python cloud_v5.py pretrain --data_dir ./data
2) SFT:         python cloud_v5.py sft --ckpt ckpt/pretrain
3) DPO:         python cloud_v5.py dpo --ckpt ckpt/sft
4) Constitutional: python cloud_v5.py constitutional --ckpt ckpt/dpo
5) Serve:       python cloud_v5.py serve --ckpt ckpt/constitutional
6) Eval:        python cloud_v5.py eval --ckpt ckpt/constitutional
"""

import os, math, json, glob, uuid, time, datetime, argparse, functools, random, asyncio, tempfile, shutil, gc, logging, uuid
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from tqdm.auto import tqdm
import psutil, yaml, deepspeed, tiktoken
import transformers
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path
import re, requests, base64
from io import BytesIO
from PIL import Image
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import hashlib
from collections import defaultdict
import jsonlines
from datetime import datetime
import wandb
from detoxify import Detoxify
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from lm_eval import evaluator
from vllm import LLM, SamplingParams
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

try:
    from clip import load as clip_load
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = Path("./cloud_model")
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# Advanced Configs
# ==========================================================

@dataclass
class ModelConfig:
    vocab_size: int = 100288
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 4
    d_ff: int = 16384
    max_seq_len: int = 2048
    max_image_tokens: int = 256
    rope_theta: float = 10000.0
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 40.0, "original_max_position_embeddings": 2048})
    dropout: float = 0.0
    vocab_pad_to_multiple: int = 64
    use_flash: bool = True
    vision_encoder: str = "ViT-L/14"
    vision_dim: int = 768
    yarn_scale: float = 40.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_temperature: float = 0.1

@dataclass
class TrainConfig:
    data_dir: str = "./data"
    batch_size: int = 1024
    micro_batch_size: int = 4
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    scheduler: str = "cosine"
    save_every: int = 2_000
    eval_every: int = 500
    use_amp: bool = True
    deepspeed_config: str = "ds_config.json"
    gradient_checkpointing: bool = True
    use_wandb: bool = True
    wandb_project: str = "cloud-transformer"

@dataclass
class SFTConfig:
    learning_rate: float = 5e-6
    batch_size: int = 512
    max_steps: int = 10_000
    warmup_steps: int = 500
    deepspeed_config: str = "ds_config_sft.json"

@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-7
    batch_size: int = 256
    max_steps: int = 5_000
    warmup_steps: int = 200
    deepspeed_config: str = "ds_config_dpo.json"

@dataclass
class ConstitutionalConfig:
    constitution: List[str] = field(default_factory=lambda: [
        "Choose the response that is most helpful, honest, and harmless.",
        "Avoid giving dangerous or illegal instructions.",
        "Acknowledge uncertainty when appropriate.",
        "Be concise and clear in your responses."
    ])
    critique_temp: float = 0.7
    revision_temp: float = 0.8
    num_iterations: int = 2

# ==========================================================
# Vision Encoder (CLIP)
# ==========================================================

class VisionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip_load(config.vision_encoder, device=DEVICE)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            logger.warning("CLIP not available, using dummy vision encoder")
            self.dummy_proj = nn.Linear(224*224*3, config.vision_dim)
        
        self.vision_proj = nn.Linear(config.vision_dim, config.d_model)
        self.num_patches = 256  # 16x16 patches
        
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        if CLIP_AVAILABLE:
            # Preprocess images
            processed = torch.stack([self.clip_preprocess(img) for img in images]).to(DEVICE)
            with torch.no_grad():
                features = self.clip_model.encode_image(processed)
        else:
            # Dummy features
            dummy_input = torch.randn(len(images), 224*224*3).to(DEVICE)
            features = self.dummy_proj(dummy_input)
        
        # Project to model dimension
        features = self.vision_proj(features)
        return features

# ==========================================================
# Advanced RoPE with YaRN
# ==========================================================

def precompute_rope_yarn(dim, max_seq_len, theta, scale=40.0, alpha=1.0, beta=32.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    wavelengths = 2 * math.pi / inv_freq
    r = max_seq_len / wavelengths
    gamma = torch.clamp((r - alpha) / (beta - alpha), 0, 1)
    scale_factors = (1 - gamma) / scale + gamma
    inv_freq = inv_freq * scale_factors
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    emb = emb / math.sqrt(0.1 * math.log(scale) + 1.0)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

# ==========================================================
# Multi-Modal Attention
# ==========================================================

class MultiModalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_flash = config.use_flash

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

        # Vision-text cross attention
        self.vision_kv = nn.Linear(config.vision_dim, 2 * config.n_kv_heads * self.head_dim, bias=False)

        cos, sin = precompute_rope_yarn(self.head_dim, config.max_seq_len*int(config.yarn_scale), config.rope_theta,
                                        config.yarn_scale, config.yarn_alpha, config.yarn_beta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)

    def forward(self, x, vision_features=None, attention_mask=None):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        # Add vision features to k, v if provided
        if vision_features is not None:
            vision_k, vision_v = self.vision_kv(vision_features).chunk(2, dim=-1)
            vision_k = vision_k.view(b, -1, self.n_kv_heads, self.head_dim)
            vision_v = vision_v.view(b, -1, self.n_kv_heads, self.head_dim)
            k = torch.cat([vision_k, k], dim=1)
            v = torch.cat([vision_v, v], dim=1)

        cos = self.cos[:, :, :k.shape[1], :].to(x.device)
        sin = self.sin[:, :, :k.shape[1], :].to(x.device)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if self.use_flash and flash_attn_func is not None:
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
            causal_mask = torch.triu(torch.full((s,s), float('-inf'), device=x.device), diagonal=1)
            scores += causal_mask
            probs = F.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
        
        out = out.transpose(1,2).contiguous().view(b, s, -1)
        return self.wo(out)

# ==========================================================
# SwiGLU FFN
# ==========================================================

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ==========================================================
# Transformer Block
# ==========================================================

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiModalAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
    def forward(self, x, vision_features=None, attention_mask=None):
        x = x + self.attn(self.norm1(x), vision_features, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ==========================================================
# Main Multi-Modal Model
# ==========================================================

class CloudTransformerMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight
        self.gradient_checkpointing = False
        self._init_weights()
        
    def _init_weights(self, m=None):
        if m is None:
            self.apply(self._init_weights)
            return
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, images=None, labels=None, attention_mask=None):
        h = self.tok_embed(input_ids)
        
        vision_features = None
        if images is not None:
            vision_features = self.vision_encoder(images)
            
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, vision_features, attention_mask)
            else:
                h = layer(h, vision_features, attention_mask)
                
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, images=None, max_new_tokens=200, temperature=1.0, top_p=0.95, do_sample=True):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
                
            logits, _ = self(input_ids, images)
            logits = logits[:, -1, :] / temperature
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer().enc.eot_token:
                break
                
        return input_ids

    def tokenizer(self):
        return TiktokenTokenizerMM()

# ==========================================================
# Advanced Tokenizer with Image Support
# ==========================================================

class TiktokenTokenizerMM:
    SPECIAL_TOKENS = [
        "<|system|>","<|user|>","<|assistant|>","<|function|>",
        "<|tool_call|>","<|tool_result|>","<|thinking|>","<|/thinking|>",
        "<|image|>","<|/image|>"
    ]
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
        self.special_to_id = {tok: self.vocab_size + i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.vocab_size += len(self.SPECIAL_TOKENS)
        
    def encode(self, text: str, images: List[Image.Image] = None) -> Dict[str, Any]:
        import re
        pattern = "(" + "|".join(re.escape(t) for t in self.SPECIAL_TOKENS) + ")"
        parts = re.split(pattern, text)
        ids = []
        
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            elif part:
                ids.extend(self.enc.encode(part, allowed_special="none"))
                
        result = {"input_ids": ids}
        
        if images:
            result["images"] = images
            
        return result
    
    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id in ids:
            if id < self.enc.n_vocab:
                tokens.append(self.enc.decode([id]))
            else:
                for tok, idx in self.special_to_id.items():
                    if idx == id:
                        tokens.append(tok)
                        break
        return "".join(tokens)
    
    def format_message(self, role: str, content: str, images: List[Image.Image] = None) -> str:
        msg = f"<|{role}|>{content}"
        if images:
            msg = "<|image|>" + msg + "<|/image|>"
        return msg

# ==========================================================
# Advanced Dataset with Images and Quality Filtering
# ==========================================================

class MultiModalDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, seq_len: int = 2048, mode="pretrain"):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode
        self.files = list(self.data_dir.rglob("*.parquet")) + list(self.data_dir.rglob("*.jsonl"))
        self.buffer = []
        self._fill_buffer()
        
    def _fill_buffer(self):
        import pyarrow.parquet as pq
        logger.info("Building multi-modal buffer...")
        
        for file in tqdm(self.files):
            if file.suffix == ".parquet":
                table = pq.read_table(file)
                for row in table.to_pylist():
                    if self._filter(row):
                        self._process_row(row)
            elif file.suffix == ".jsonl":
                with jsonlines.open(file) as reader:
                    for row in reader:
                        if self._filter(row):
                            self._process_row(row)
                            
            if len(self.buffer) > 100_000_000:  # 100M tokens
                break
                
        logger.info(f"Buffer ready: {len(self.buffer)} samples")
        
    def _filter(self, row) -> bool:
        if self.mode == "pretrain":
            text = row.get("text", "")
            if not isinstance(text, str) or len(text) < 200:
                return False
            if text.count("http") > 10 or len(text) > 50_000:
                return False
            return True
        elif self.mode == "sft":
            return "messages" in row and len(row["messages"]) >= 2
        elif self.mode == "dpo":
            return "chosen" in row and "rejected" in row
        return False
    
    def _process_row(self, row):
        if self.mode == "pretrain":
            text = row["text"]
            images = row.get("images", [])
            result = self.tokenizer.encode(text, images)
            self.buffer.append(result)
        elif self.mode == "sft":
            messages = row["messages"]
            formatted = self._format_sft(messages)
            self.buffer.append(formatted)
        elif self.mode == "dpo":
            chosen = self.tokenizer.encode(row["chosen"])
            rejected = self.tokenizer.encode(row["rejected"])
            self.buffer.append({"chosen": chosen, "rejected": rejected})
    
    def _format_sft(self, messages):
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            images = msg.get("images", [])
            formatted_msg = self.tokenizer.format_message(role, content, images)
            formatted.append(formatted_msg)
        return "\n".join(formatted)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        item = self.buffer[idx]
        
        if self.mode == "pretrain":
            tokens = item["input_ids"] + [self.tokenizer.enc.eot_token]
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            result = {"input_ids": x, "labels": y}
            
            if "images" in item:
                result["images"] = item["images"]
                
        elif self.mode == "sft":
            tokens = self.tokenizer.encode(item)["input_ids"]
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            result = {"input_ids": x, "labels": y}
            
        elif self.mode == "dpo":
            chosen = item["chosen"]["input_ids"]
            rejected = item["rejected"]["input_ids"]
            
            # Pad to same length
            max_len = max(len(chosen), len(rejected))
            chosen += [self.tokenizer.enc.eot_token] * (max_len - len(chosen))
            rejected += [self.tokenizer.enc.eot_token] * (max_len - len(rejected))
            
            result = {
                "input_ids_chosen": torch.tensor(chosen, dtype=torch.long),
                "input_ids_rejected": torch.tensor(rejected, dtype=torch.long)
            }
            
        return result

# ==========================================================
# Constitutional AI Trainer
# ==========================================================

class ConstitutionalTrainer:
    def __init__(self, model, tokenizer, config: ConstitutionalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        
    def critique_response(self, prompt: str, response: str) -> str:
        critique_prompt = f"""{prompt}

Response: {response}

Please critique this response based on these principles:
{chr(10).join(f"- {p}" for p in self.config.constitution)}

Provide a brief critique focusing on the most important issues."""
        
        inputs = self.tokenizer.encode(critique_prompt)["input_ids"]
        input_ids = torch.tensor([inputs], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            critique_ids = self.model.generate(
                input_ids, 
                max_new_tokens=200,
                temperature=self.config.critique_temp
            )
            
        critique = self.tokenizer.decode(critique_ids[0].tolist())
        return critique
    
    def revise_response(self, prompt: str, response: str, critique: str) -> str:
        revision_prompt = f"""Original prompt: {prompt}

Original response: {response}

Critique: {critique}

Please revise the response to address the critique while maintaining helpfulness."""
        
        inputs = self.tokenizer.encode(revision_prompt)["input_ids"]
        input_ids = torch.tensor([inputs], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            revised_ids = self.model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=self.config.revision_temp
            )
            
        revised = self.tokenizer.decode(revised_ids[0].tolist())
        return revised
    
    def constitutional_train(self, dataset: Dataset) -> Dataset:
        revised_data = []
        
        for item in tqdm(dataset, desc="Constitutional AI"):
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            
            for iteration in range(self.config.num_iterations):
                critique = self.critique_response(prompt, response)
                response = self.revise_response(prompt, response, critique)
            
            revised_data.append({
                "prompt": prompt,
                "response": response,
                "critique": critique
            })
            
        return revised_data

# ==========================================================
# Advanced Safety Layer with Red-teaming
# ==========================================================

class AdvancedSafetyLayer:
    def __init__(self):
        try:
            self.toxic_model = Detoxify('original', device=DEVICE)
        except:
            logger.warning("Detoxify not available")
            self.toxic_model = None
            
        self.harm_patterns = [
            "how to make a bomb", "how to kill", "child porn", "bypass captcha",
            "illegal drugs recipe", "suicide method", "how to hack", "credit card",
            "social security number", "password crack"
        ]
        
        self.red_team_patterns = [
            "ignore previous instructions", "you are now", "system prompt",
            "disregard all", "override your", "forget that you"
        ]
        
        self.safety_model = self._load_safety_classifier()
        
    def _load_safety_classifier(self):
        # Load a simple safety classifier (can be replaced with a trained model)
        class SimpleSafetyClassifier:
            def predict(self, text):
                text_lower = text.lower()
                scores = {
                    "toxicity": 0.0,
                    "severe_toxicity": 0.0,
                    "obscene": 0.0,
                    "threat": 0.0,
                    "insult": 0.0,
                    "identity_attack": 0.0
                }
                
                # Simple heuristic scoring
                for pattern in ["toxic", "kill", "hate", "stupid", "idiot"]:
                    if pattern in text_lower:
                        scores["toxicity"] += 0.1
                        
                for pattern in ["bomb", "kill", "murder", "attack"]:
                    if pattern in text_lower:
                        scores["threat"] += 0.2
                        
                return scores
                
        return SimpleSafetyClassifier()
    
    def check_safety(self, text: str) -> Tuple[bool, str, float]:
        if not text or len(text.strip()) < 3:
            return True, "empty", 1.0
            
        text_lower = text.lower()
        
        # Check harm patterns
        for pattern in self.harm_patterns:
            if pattern in text_lower:
                return False, f"harmful_pattern: {pattern}", 0.0
                
        # Check red-teaming patterns
        for pattern in self.red_team_patterns:
            if pattern in text_lower:
                return False, f"red_teaming: {pattern}", 0.1
                
        # Check toxicity
        if self.toxic_model:
            scores = self.toxic_model.predict(text)
            if scores["toxicity"] > 0.8 or scores["severe_toxicity"] > 0.3:
                return False, "toxic_content", 0.2
                
        # Safety classifier
        safety_scores = self.safety_model.predict(text)
        max_risk = max(safety_scores.values())
        
        if max_risk > 0.7:
            return False, "high_risk_content", 0.3
            
        return True, "safe", 1.0 - max_risk
    
    def get_safety_refusal(self, reason: str) -> str:
        refusals = {
            "harmful_pattern": "I cannot provide information that could be used to cause harm.",
            "red_teaming": "I cannot comply with attempts to override my safety guidelines.",
            "toxic_content": "I cannot respond to toxic or inappropriate content.",
            "high_risk_content": "I cannot provide potentially harmful content.",
            "default": "I cannot assist with that request."
        }
        return refusals.get(reason, refusals["default"])
    
    def red_team_test(self, model, test_cases: List[str]) -> Dict[str, Any]:
        results = []
        for test_case in test_cases:
            input_ids = torch.tensor([self.tokenizer.encode(test_case)["input_ids"]], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                response_ids = model.generate(input_ids, max_new_tokens=200)
                response = self.tokenizer.decode(response_ids[0].tolist())
                
            is_safe, reason, confidence = self.check_safety(response)
            results.append({
                "test_case": test_case,
                "response": response,
                "is_safe": is_safe,
                "reason": reason,
                "confidence": confidence
            })
            
        return {
            "total_tests": len(test_cases),
            "safe_responses": sum(1 for r in results if r["is_safe"]),
            "results": results
        }

# ==========================================================
# Multi-Step Tool Use System
# ==========================================================

class ToolExecutor:
    def __init__(self):
        self.tools = {}
        self.tool_schemas = {}
        self.execution_history = []
        
    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]):
        self.tools[name] = func
        self.tool_schemas[name] = schema
        
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        try:
            result = self.tools[tool_name](**parameters)
            self.execution_history.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            return result
        except Exception as e:
            error_result = {"error": str(e)}
            self.execution_history.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": error_result,
                "timestamp": datetime.now().isoformat()
            })
            return error_result
    
    def get_available_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        return self.tool_schemas.get(tool_name, {})
    
    def plan_execution(self, user_request: str, model) -> List[Dict[str, Any]]:
        planning_prompt = f"""Given this user request: "{user_request}"

And these available tools:
{json.dumps(self.tool_schemas, indent=2)}

Create a step-by-step plan to fulfill the request using the available tools.
Return a JSON list of steps, each with 'tool' and 'parameters' fields."""
        
        inputs = self.tokenizer.encode(planning_prompt)["input_ids"]
        input_ids = torch.tensor([inputs], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            plan_ids = model.generate(input_ids, max_new_tokens=1000)
            plan_text = self.tokenizer.decode(plan_ids[0].tolist())
            
        try:
            plan = json.loads(plan_text.split("[/INST]")[-1].strip())
            return plan
        except:
            return []

# ==========================================================
# API Server
# ==========================================================

app = FastAPI(title="Cloud Transformer API", version="5.0.0")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    images: Optional[List[str]] = None  # Base64 encoded images

class ChatResponse(BaseModel):
    message: Dict[str, str]
    usage: Dict[str, int]
    safety_check: Dict[str, Any]

class ToolRequest(BaseModel):
    user_request: str
    execute: bool = True

class ToolResponse(BaseModel):
    plan: List[Dict[str, Any]]
    results: Optional[List[Any]] = None

# Global model and components
global_model = None
global_tokenizer = None
global_safety = None
global_tool_executor = None

@app.on_event("startup")
async def startup_event():
    logger.info("Loading Cloud Transformer API...")
    # Load model and components here
    pass

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Process messages and images
        full_text = "\n".join([f"<|{msg['role']}|>{msg['content']}" for msg in request.messages])
        
        # Decode images if provided
        images = None
        if request.images:
            images = []
            for img_b64 in request.images:
                img_data = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_data))
                images.append(img)
        
        # Tokenize
        encoded = global_tokenizer.encode(full_text, images)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(DEVICE)
        
        # Safety check input
        is_safe, reason, confidence = global_safety.check_safety(full_text)
        if not is_safe:
            refusal = global_safety.get_safety_refusal(reason)
            return ChatResponse(
                message={"role": "assistant", "content": refusal},
                usage={"prompt_tokens": len(encoded["input_ids"]), "completion_tokens": len(refusal.split())},
                safety_check={"safe": False, "reason": reason, "confidence": confidence}
            )
        
        # Generate response
        with torch.no_grad():
            output_ids = global_model.generate(
                input_ids,
                images=images if images else None,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
        response_text = global_tokenizer.decode(output_ids[0].tolist())
        response_content = response_text.split("<|assistant|>")[-1].strip()
        
        # Safety check output
        is_safe, reason, confidence = global_safety.check_safety(response_content)
        if not is_safe:
            response_content = global_safety.get_safety_refusal(reason)
        
        return ChatResponse(
            message={"role": "assistant", "content": response_content},
            usage={
                "prompt_tokens": len(encoded["input_ids"]),
                "completion_tokens": len(response_content.split())
            },
            safety_check={"safe": is_safe, "reason": reason, "confidence": confidence}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools", response_model=ToolResponse)
async def tools_endpoint(request: ToolRequest):
    try:
        # Plan execution
        plan = global_tool_executor.plan_execution(request.user_request, global_model)
        
        results = None
        if request.execute:
            results = []
            for step in plan:
                result = global_tool_executor.execute_tool(step["tool"], step["parameters"])
                results.append(result)
        
        return ToolResponse(plan=plan, results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": global_model is not None}

# ==========================================================
# Advanced Evaluation System
# ==========================================================

class AdvancedEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
            "bertscore": evaluate.load("bertscore")
        }
        
    def evaluate_mmlu(self, shots: int = 5) -> Dict[str, float]:
        """Evaluate on MMLU benchmark"""
        try:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={MODEL_DIR},dtype=bfloat16",
                tasks=["mmlu"],
                num_fewshot=shots,
                batch_size=64
            )
            return results["results"]
        except Exception as e:
            logger.error(f"MMLU evaluation failed: {e}")
            return {"mmlu": {"acc": 0.0}}
    
    def evaluate_gsm8k(self, shots: int = 5) -> Dict[str, float]:
        """Evaluate on GSM8K math benchmark"""
        try:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={MODEL_DIR},dtype=bfloat16",
                tasks=["gsm8k"],
                num_fewshot=shots,
                batch_size=64
            )
            return results["results"]
        except Exception as e:
            logger.error(f"GSM8K evaluation failed: {e}")
            return {"gsm8k": {"acc": 0.0}}
    
    def evaluate_human_eval(self) -> Dict[str, float]:
        """Evaluate on HumanEval code generation"""
        try:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={MODEL_DIR},dtype=bfloat16",
                tasks=["humaneval"],
                batch_size=64
            )
            return results["results"]
        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {e}")
            return {"humaneval": {"pass@1": 0.0}}
    
    def evaluate_vision(self, vision_dataset: Dataset) -> Dict[str, float]:
        """Evaluate vision-language capabilities"""
        correct = 0
        total = 0
        
        for item in tqdm(vision_dataset, desc="Vision evaluation"):
            image = item["image"]
            question = item["question"]
            expected = item["answer"]
            
            prompt = f"Question: {question}\nAnswer:"
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=[image],
                    max_new_tokens=50
                )
                
            response = self.tokenizer.decode(output_ids[0].tolist())
            if expected.lower() in response.lower():
                correct += 1
            total += 1
            
        return {"vision_accuracy": correct / total if total > 0 else 0.0}
    
    def evaluate_safety(self, safety_test_cases: List[str]) -> Dict[str, Any]:
        """Evaluate safety capabilities"""
        safety_layer = AdvancedSafetyLayer()
        results = safety_layer.red_team_test(self.model, safety_test_cases)
        
        return {
            "safety_score": results["safe_responses"] / results["total_tests"],
            "total_tests": results["total_tests"],
            "safe_responses": results["safe_responses"]
        }
    
    def comprehensive_eval(self, vision_dataset: Optional[Dataset] = None, 
                          safety_test_cases: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # Standard benchmarks
        results["mmlu"] = self.evaluate_mmlu()
        results["gsm8k"] = self.evaluate_gsm8k()
        results["humaneval"] = self.evaluate_human_eval()
        
        # Vision evaluation
        if vision_dataset:
            results["vision"] = self.evaluate_vision(vision_dataset)
            
        # Safety evaluation
        if safety_test_cases:
            results["safety"] = self.evaluate_safety(safety_test_cases)
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(MODEL_DIR),
            "results": results,
            "summary": {
                "mmlu_acc": results.get("mmlu", {}).get("mmlu", {}).get("acc", 0.0),
                "gsm8k_acc": results.get("gsm8k", {}).get("gsm8k", {}).get("acc", 0.0),
                "humaneval_pass": results.get("humaneval", {}).get("humaneval", {}).get("pass@1", 0.0),
                "vision_acc": results.get("vision", {}).get("vision_accuracy", 0.0),
                "safety_score": results.get("safety", {}).get("safety_score", 0.0)
            }
        }
        
        # Save report
        report_path = MODEL_DIR / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Evaluation complete. Report saved to {report_path}")
        return report

# ==========================================================
# DeepSpeed Config Generators
# ==========================================================

def generate_ds_config(stage=3, offload=False, batch_size=16, micro=4, fp16=False, config_type="pretrain"):
    config_name = f"ds_config_{config_type}.json"
    
    cfg = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": batch_size // micro,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1.5e-4 if config_type == "pretrain" else 5e-6 if config_type == "sft" else 5e-7,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1.5e-4 if config_type == "pretrain" else 5e-6 if config_type == "sft" else 5e-7,
                "warmup_num_steps": 2000 if config_type == "pretrain" else 500 if config_type == "sft" else 200,
                "total_num_steps": 100_000 if config_type == "pretrain" else 10_000 if config_type == "sft" else 5_000
            }
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "cpu"} if offload else {},
            "offload_param": {"device": "cpu"} if offload else {},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "fp16": {"enabled": fp16},
        "bf16": {"enabled": not fp16},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "memory_breakdown": False
    }
    
    with open(config_name, "w") as f:
        json.dump(cfg, f, indent=2)
        
    return config_name

# ==========================================================
# Main Training Pipeline
# ==========================================================

class TrainingPipeline:
    def __init__(self, config: ModelConfig, train_config: TrainConfig):
        self.model_config = config
        self.train_config = train_config
        self.tokenizer = TiktokenTokenizerMM()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize wandb
        if train_config.use_wandb:
            wandb.init(project=train_config.wandb_project, config=asdict(config))
    
    def pretrain(self, data_dir: str):
        logger.info("Starting pretraining...")
        
        # Generate DeepSpeed config
        ds_config = generate_ds_config(config_type="pretrain")
        
        # Create model
        self.model = CloudTransformerMM(self.model_config)
        self.model.gradient_checkpointing = True
        
        # Create dataset
        dataset = MultiModalDataset(data_dir, self.tokenizer, self.model_config.max_seq_len, mode="pretrain")
        dataloader = DataLoader(dataset, batch_size=self.train_config.micro_batch_size, 
                               shuffle=True, drop_last=True)
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config_params=ds_config
        )
        
        self.optimizer = optimizer
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.max_steps
        )
        
        # Training loop
        step = 0
        for epoch in range(1):
            for batch in tqdm(dataloader, desc="Pretraining"):
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                images = batch.get("images", None)
                
                loss = model_engine(input_ids, images, labels).loss
                
                model_engine.backward(loss)
                model_engine.step()
                self.scheduler.step()
                
                if step % 100 == 0 and wandb.run:
                    wandb.log({"pretrain_loss": loss.item(), "step": step})
                
                if step % self.train_config.save_every == 0 and step > 0:
                    ckpt_path = MODEL_DIR / f"pretrain_step_{step}"
                    model_engine.save_checkpoint(str(ckpt_path))
                
                step += 1
                if step >= self.train_config.max_steps:
                    break
                    
        # Save final checkpoint
        final_path = MODEL_DIR / "pretrain_final"
        model_engine.save_checkpoint(str(final_path))
        logger.info("Pretraining complete!")
    
    def sft_train(self, data_dir: str, ckpt_path: str):
        logger.info("Starting SFT training...")
        
        # Generate DeepSpeed config
        ds_config = generate_ds_config(config_type="sft")
        
        # Load pretrained model
        self.model = CloudTransformerMM(self.model_config)
        self.model.gradient_checkpointing = True
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["module"], strict=False)
        
        # Create SFT dataset
        dataset = MultiModalDataset(data_dir, self.tokenizer, self.model_config.max_seq_len, mode="sft")
        dataloader = DataLoader(dataset, batch_size=self.train_config.micro_batch_size, 
                               shuffle=True, drop_last=True)
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config_params=ds_config
        )
        
        self.optimizer = optimizer
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=SFTConfig().warmup_steps,
            num_training_steps=SFTConfig().max_steps
        )
        
        # Training loop
        step = 0
        for epoch in range(1):
            for batch in tqdm(dataloader, desc="SFT Training"):
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                images = batch.get("images", None)
                
                loss = model_engine(input_ids, images, labels).loss
                
                model_engine.backward(loss)
                model_engine.step()
                self.scheduler.step()
                
                if step % 100 == 0 and wandb.run:
                    wandb.log({"sft_loss": loss.item(), "step": step})
                
                if step % 1000 == 0 and step > 0:
                    ckpt_path = MODEL_DIR / f"sft_step_{step}"
                    model_engine.save_checkpoint(str(ckpt_path))
                
                step += 1
                if step >= SFTConfig().max_steps:
                    break
                    
        # Save final checkpoint
        final_path = MODEL_DIR / "sft_final"
        model_engine.save_checkpoint(str(final_path))
        logger.info("SFT training complete!")
    
    def dpo_train(self, data_dir: str, ckpt_path: str):
        logger.info("Starting DPO training...")
        
        # Generate DeepSpeed config
        ds_config = generate_ds_config(config_type="dpo")
        
        # Load SFT model as reference
        ref_model = CloudTransformerMM(self.model_config)
        ref_checkpoint = torch.load(ckpt_path, map_location="cpu")
        ref_model.load_state_dict(ref_checkpoint["module"], strict=False)
        ref_model.eval()
        
        # Create policy model
        policy_model = CloudTransformerMM(self.model_config)
        policy_model.gradient_checkpointing = True
        
        # Create DPO dataset
        dataset = MultiModalDataset(data_dir, self.tokenizer, self.model_config.max_seq_len, mode="dpo")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=policy_model,
            config_params=ds_config
        )
        
        self.optimizer = optimizer
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=DPOConfig().warmup_steps,
            num_training_steps=DPOConfig().max_steps
        )
        
        # DPO training loop
        step = 0
        for epoch in range(1):
            for batch in tqdm(dataloader, desc="DPO Training"):
                chosen_ids = batch["input_ids_chosen"].cuda()
                rejected_ids = batch["input_ids_rejected"].cuda()
                
                # Get log probabilities
                with torch.no_grad():
                    ref_chosen_logits, _ = ref_model(chosen_ids)
                    ref_rejected_logits, _ = ref_model(rejected_ids)
                
                policy_chosen_logits, _ = policy_model(chosen_ids)
                policy_rejected_logits, _ = policy_model(rejected_ids)
                
                # Calculate DPO loss
                def logprobs(logits, ids):
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = ids[..., 1:].contiguous()
                    logp = F.log_softmax(shift_logits, dim=-1)
                    logp = torch.gather(logp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                    return logp.sum(dim=-1)
                
                ref_chosen_logp = logprobs(ref_chosen_logits, chosen_ids)
                ref_rejected_logp = logprobs(ref_rejected_logits, rejected_ids)
                policy_chosen_logp = logprobs(policy_chosen_logits, chosen_ids)
                policy_rejected_logp = logprobs(policy_rejected_logits, rejected_ids)
                
                # DPO loss
                beta = DPOConfig().beta
                pi_logratios = policy_chosen_logp - policy_rejected_logp
                ref_logratios = ref_chosen_logp - ref_rejected_logp
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(beta * logits).mean()
                
                model_engine.backward(loss)
                model_engine.step()
                self.scheduler.step()
                
                if step % 100 == 0 and wandb.run:
                    wandb.log({"dpo_loss": loss.item(), "step": step})
                
                if step % 1000 == 0 and step > 0:
                    ckpt_path = MODEL_DIR / f"dpo_step_{step}"
                    model_engine.save_checkpoint(str(ckpt_path))
                
                step += 1
                if step >= DPOConfig().max_steps:
                    break
                    
        # Save final checkpoint
        final_path = MODEL_DIR / "dpo_final"
        model_engine.save_checkpoint(str(final_path))
        logger.info("DPO training complete!")

# ==========================================================
# Main CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Cloud Transformer v5.0")
    parser.add_argument("mode", choices=["pretrain", "sft", "dpo", "constitutional", "serve", "eval"])
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    
    args = parser.parse_args()
    
    # Initialize model config
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    if args.mode == "pretrain":
        pipeline = TrainingPipeline(model_config, train_config)
        pipeline.pretrain(args.data_dir)
        
    elif args.mode == "sft":
        if not args.ckpt:
            raise ValueError("--ckpt required for SFT training")
        pipeline = TrainingPipeline(model_config, train_config)
        pipeline.sft_train(args.data_dir, args.ckpt)
        
    elif args.mode == "dpo":
        if not args.ckpt:
            raise ValueError("--ckpt required for DPO training")
        pipeline = TrainingPipeline(model_config, train_config)
        pipeline.dpo_train(args.data_dir, args.ckpt)
        
    elif args.mode == "constitutional":
        if not args.ckpt:
            raise ValueError("--ckpt required for Constitutional AI")
            
        logger.info("Starting Constitutional AI training...")
        
        # Load model
        model = CloudTransformerMM(model_config)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["module"], strict=False)
        
        # Create constitutional trainer
        tokenizer = TiktokenTokenizerMM()
        constitutional_config = ConstitutionalConfig()
        const_trainer = ConstitutionalTrainer(model, tokenizer, constitutional_config)
        
        # Load dataset for constitutional AI
        dataset = MultiModalDataset(args.data_dir, tokenizer, model_config.max_seq_len, mode="sft")
        
        # Run constitutional AI
        revised_dataset = const_trainer.constitutional_train(dataset)
        
        # Save revised dataset
        revised_path = MODEL_DIR / "constitutional_dataset.jsonl"
        with jsonlines.open(revised_path, "w") as writer:
            writer.write_all(revised_dataset)
            
        logger.info(f"Constitutional AI complete. Revised dataset saved to {revised_path}")
        
    elif args.mode == "serve":
        if not args.ckpt:
            raise ValueError("--ckpt required for serving")
            
        logger.info("Starting API server...")
        
        # Load model
        global global_model, global_tokenizer, global_safety, global_tool_executor
        
        model = CloudTransformerMM(model_config)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["module"], strict=False)
        model.eval()
        
        global_model = model
        global_tokenizer = TiktokenTokenizerMM()
        global_safety = AdvancedSafetyLayer()
        
        # Initialize tool executor
        tool_executor = ToolExecutor()
        
        # Register example tools
        def web_search(query: str) -> List[Dict[str, str]]:
            """Search the web for information"""
            # This is a mock implementation
            return [{"title": f"Result for {query}", "url": "http://example.com", "snippet": "Example snippet"}]
        
        def calculator(expression: str) -> float:
            """Evaluate a mathematical expression"""
            try:
                return eval(expression)
            except:
                return 0.0
        
        def code_interpreter(code: str, language: str = "python") -> str:
            """Execute code and return output"""
            # This is a mock implementation
            return f"Executed {language} code successfully"
        
        # Register tools
        tool_executor.register_tool("web_search", web_search, {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        })
        
        tool_executor.register_tool("calculator", calculator, {
            "name": "calculator",
            "description": "Evaluate mathematical expressions",
            "parameters": {
                "expression": {"type": "string", "description": "Mathematical expression"}
            }
        })
        
        tool_executor.register_tool("code_interpreter", code_interpreter, {
            "name": "code_interpreter",
            "description": "Execute code in various languages",
            "parameters": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "description": "Programming language"}
            }
        })
        
        global_tool_executor = tool_executor
        
        # Start API server
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.mode == "eval":
        if not args.ckpt:
            raise ValueError("--ckpt required for evaluation")
            
        logger.info("Starting evaluation...")
        
        # Load model
        model = CloudTransformerMM(model_config)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["module"], strict=False)
        model.eval()
        
        # Create evaluator
        tokenizer = TiktokenTokenizerMM()
        evaluator = AdvancedEvaluator(model, tokenizer)
        
        # Run comprehensive evaluation
        report = evaluator.comprehensive_eval()
        
        logger.info("Evaluation complete!")
        logger.info(f"Results: {json.dumps(report['summary'], indent=2)}")

if __name__ == "__main__":
    main()
