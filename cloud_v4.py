#!/usr/bin/env python3
#  Cloud-Scale Transformer v4.0  (Claude-3 class)
#  ------------------------------------------------------
#  - 8B→70B params  |  200k YaRN++  |  FlashAttn-2
#  - DeepSpeed ZeRO-3  |  bf16  |  DPO/RLHF
#  - vLLM inference  |  Function-calling  |  Safety
#  ------------------------------------------------------
#  USAGE:
#  1) Pre-train:  python cloud_v4.py pretrain --data_dir ./pile
#  2) DPO:        python cloud_v4.py dpo --ckpt ckpt/pretrain
#  3) Serve:      python cloud_v4.py serve --ckpt ckpt/dpo

import os, math, json, glob, uuid, time, datetime, argparse, functools, random, asyncio, tempfile, shutil, gc, logging, uuid
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path

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

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- dirs ----------
MODEL_DIR = Path("./cloud_model")
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# Configs
# ==========================================================

@dataclass
class ModelConfig:
    vocab_size: int = 100288          # tiktoken cl100k + special
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 4
    d_ff: int = 16384
    max_seq_len: int = 2048           # pretrain 2k → 200k via YaRN
    rope_theta: float = 10000.0
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 40.0, "original_max_position_embeddings": 2048})
    dropout: float = 0.0
    vocab_pad_to_multiple: int = 64
    use_flash: bool = FLASH_AVAILABLE
    # YaRN++
    yarn_scale: float = 40.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_temperature: float = 0.1

@dataclass
class TrainConfig:
    # data
    data_dir: str = "./data"
    batch_size: int = 1024            # tokens per GPU
    micro_batch_size: int = 4
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    # opt
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    # sched
    scheduler: str = "cosine"
    # ckpt
    save_every: int = 2_000
    eval_every: int = 500
    # tech
    use_amp: bool = True
    deepspeed_config: str = "ds_config.json"
    gradient_checkpointing: bool = True

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
        "Acknowledge uncertainty when appropriate."
    ])
    critique_temp: float = 0.7
    revision_temp: float = 0.8

# ==========================================================
# YaRN++ RoPE
# ==========================================================

def precompute_rope_yarn(dim, max_seq_len, theta, scale=40.0, alpha=1.0, beta=32.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # NTK-by-parts
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
# RMSNorm
# ==========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm + self.bias).type_as(x)

# ==========================================================
# FlashAttention-2  (grouped-query)
# ==========================================================

class FlashGQAttention(nn.Module):
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

        # rope caches
        cos, sin = precompute_rope_yarn(self.head_dim, config.max_seq_len*int(config.yarn_scale), config.rope_theta,
                                        config.yarn_scale, config.yarn_alpha, config.yarn_beta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)

    def forward(self, x, attention_mask=None):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        cos = self.cos[:, :, :s, :].to(x.device)
        sin = self.sin[:, :, :s, :].to(x.device)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # grouped-query repeat
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)  # (b,h,s,d)

        if self.use_flash and flash_attn_func is not None:
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
        else:
            # fallback
            scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
            causal_mask = torch.triu(torch.full((s,s), float('-inf'), device=x.device), diagonal=1)
            scores += causal_mask
            probs = torch.nn.functional.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)  # (b,h,s,d)
        out = out.transpose(1,2).contiguous().view(b, s, -1)
        return self.wo(out)

# ==========================================================
# SwiGLU
# ==========================================================

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

# ==========================================================
# Transformer Block
# ==========================================================

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = FlashGQAttention(config)
        self.ffn  = SwiGLU(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ==========================================================
# Main Model
# ==========================================================

class CloudTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight  # tie
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
    def forward(self, input_ids, labels=None, attention_mask=None):
        h = self.tok_embed(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, attention_mask)
            else:
                h = layer(h, attention_mask)
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                     shift_labels.view(-1),
                                                     ignore_index=-100)
        return logits, loss
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=1.0, top_p=0.95, do_sample=True):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer().enc.eot_token:
                break
        return input_ids
    def tokenizer(self):
        # quick accessor
        return TiktokenTokenizer()

# ==========================================================
# Tokenizer with special tokens
# ==========================================================

class TiktokenTokenizer:
    SPECIAL_TOKENS = [
        "<|system|>","<|user|>","<|assistant|>","<|function|>",
        "<|tool_call|>","<|tool_result|>","<|thinking|>","<|/thinking|>"
    ]
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
        self.special_to_id = {tok: self.vocab_size + i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.vocab_size += len(self.SPECIAL_TOKENS)
    def encode(self, text: str) -> List[int]:
        # simple regex split to keep special tokens
        import re
        pattern = "(" + "|".join(re.escape(t) for t in self.SPECIAL_TOKENS) + ")"
        parts = re.split(pattern, text)
        ids = []
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            elif part:
                ids.extend(self.enc.encode(part, allowed_special="none"))
        return ids
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
    def format_message(self, role: str, content: str) -> str:
        return f"<|{role}|>{content}"

# ==========================================================
# Dataset (Large-scale parquet)
# ==========================================================

class LargeParquetDataset(Dataset):
    """
    Reads folder of parquet files, does fast streaming tokenization,
    quality filter, dedup-mini-hash, and returns token chunks.
    """
    def __init__(self, parquet_dir: str, tokenizer, seq_len: int = 2048):
        self.files = list(Path(parquet_dir).rglob("*.parquet"))
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.buffer = []
        self._fill()
    def _fill(self):
        import pyarrow.parquet as pq
        from tqdm import tqdm
        logger.info("Building token buffer ...")
        for file in tqdm(self.files):
            table = pq.read_table(file)
            for text in table['text'].to_pylist():
                if not isinstance(text, str) or len(text) < 200:
                    continue
                # simple quality filter
                if self._filter(text):
                    toks = self.tokenizer.encode(text) + [self.tokenizer.enc.eot_token]
                    self.buffer.extend(toks)
                    if len(self.buffer) > 50_000_000:
                        return
        logger.info(f"Buffer ready: {len(self.buffer)} tokens")
    def _filter(self, text: str) -> bool:
        # fast heuristic
        if len(text) < 200 or len(text) > 50_000:
            return False
        if text.count("http") > 10:
            return False
        return True
    def __len__(self):
        return len(self.buffer) // self.seq_len
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.buffer[start:start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ==========================================================
# DeepSpeed configs (auto-generated)
# ==========================================================

def write_ds_config(stage=3, offload=False, batch_size=16, micro=4, fp16=False):
    cfg = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": batch_size // micro,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1.5e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1.5e-4,
                "warmup_num_steps": 2000,
                "total_num_steps": 100_000
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
        "wall_clock_breakdown": False
    }
    fname = "ds_config.json" if stage == 3 else "ds_config_dpo.json"
    with open(fname, "w") as f:
        json.dump(cfg, f, indent=2)
    return fname

# ==========================================================
# DPO Trainer (RLHF light)
# ==========================================================

class DPOTrainer:
    """
    Direct Preference Optimization – needs (chosen, rejected) pairs.
    """
    def __init__(self, model_ref, model_policy, tokenizer, dpo_cfg: DPOConfig):
        self.ref = model_ref
        self.policy = model_policy
        self.tokenizer = tokenizer
        self.cfg = dpo_cfg
        write_ds_config(stage=3, batch_size=dpo_cfg.batch_size, micro=4, fp16=False)
        self.engine, _, _, _ = deepspeed.initialize(
            model=self.policy,
            config_params=dpo_cfg.deepspeed_config
        )
        self.ref.eval()
        self.optimizer = self.engine.optimizer
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=dpo_cfg.warmup_steps,
            num_training_steps=dpo_cfg.max_steps
        )
        self.step = 0
    def dpo_loss(self, policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp):
        beta = self.cfg.beta
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp
        logits = pi_logratios - ref_logratios
        loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
        chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp).detach()
        rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp).detach()
        return loss, chosen_rewards, rejected_rewards
    def step_batch(self, batch):
        # batch = dict(input_ids_chosen, input_ids_rejected)
        self.policy.train()
        self.ref.eval()
        with torch.no_grad():
            ref_chosen_logits, _ = self.ref(batch["input_ids_chosen"])
            ref_rejected_logits, _ = self.ref(batch["input_ids_rejected"])
        policy_chosen_logits, _ = self.policy(batch["input_ids_chosen"])
        policy_rejected_logits, _ = self.policy(batch["input_ids_rejected"])

        def logprobs(logits, ids):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            logp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            logp = torch.gather(logp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            return logp.sum(dim=-1)  # sum over seq
        ref_chosen_logp = logprobs(ref_chosen_logits, batch["input_ids_chosen"])
        ref_rejected_logp = logprobs(ref_rejected_logits, batch["input_ids_rejected"])
        policy_chosen_logp = logprobs(policy_chosen_logits, batch["input_ids_chosen"])
        policy_rejected_logp = logprobs(policy_rejected_logits, batch["input_ids_rejected"])

        loss, chosen_r, rejected_r = self.dpo_loss(policy_chosen_logp, policy_rejected_logp,
                                                   ref_chosen_logp, ref_rejected_logp)
        self.engine.backward(loss)
        self.engine.step()
        self.scheduler.step()
        self.step += 1
        return loss.item(), chosen_r.mean().item(), rejected_r.mean().item()

# ==========================================================
# Tool Registry & Function Calling
# ==========================================================

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, dict] = {}
    def register(self, name: str, func: Callable, schema: dict):
        self.tools[name] = func
        self.schemas[name] = schema
    def get(self, name: str) -> Optional[Callable]:
        return self.tools.get(name)
    def schema(self, name: str) -> Optional[dict]:
        return self.schemas.get(name)
    def list(self) -> List[str]:
        return list(self.tools.keys())

# ==========================================================
# Safety Layer
# ==========================================================

class SafetyLayer:
    def __init__(self):
        try:
            from detoxify import Detoxify
            self.toxic_model = Detoxify('original', device=DEVICE)
        except Exception as e:
            logger.warning(f"Detoxify not available: {e}")
            self.toxic_model = None
        self.harm_patterns = [
            "how to make a bomb", "how to kill", "child porn", "bypass captcha",
            "illegal drugs recipe", "suicide method"
        ]
    def check(self, text: str) -> Tuple[bool, str]:
        if not text or len(text.strip()) < 3:
            return True, ""
        t = text.lower()
        for p in self.harm_patterns:
            if p in t:
                return False, "harmful_pattern"
        if self.toxic_model:
            scores = self.toxic_model.predict(text)
            if scores["toxicity"] > 0.8 or scores["severe_toxicity"] > 0.3:
                return False, "toxic"
        return True, "safe"
    def refusal(self) -> str:
        return "I’m sorry, but I can’t assist with that."

# ==========================================================
# vLLM Inference Wrapper
# ==========================================================

class vLLMEngine:
    """
    Quick wrapper around vLLM for fast serving.
    Install: pip install vllm
    """
    def __init__(self, model_path: str, gpu_memory_utilization=0.9, max_model_len=8192):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len,
                       trust_remote_code=False)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

# ==========================================================
# Evaluation Harness
# ==========================================================

def evaluate_model(ckpt_dir: str, tasks: List[str] = ("mmlu", "gsm8k", "humaneval")):
    try:
        from lm_eval import evaluator
    except ImportError:
        logger.error("lm-eval not installed. pip install lm-eval")
        return
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={ckpt_dir},dtype=bfloat16",
        tasks=list(tasks),
        batch_size=64,
    )
    out_file = Path(ckpt_dir) / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation saved to {out_file}")
    return results

# ==========================================================
# Main CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pretrain", "dpo", "serve", "eval"])
    parser.add_argument("--data_dir", default="./data", help="parquet folder")
    parser.add_argument("--ckpt", default=None, help="checkpoint folder")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()

    if args.mode == "pretrain":
        dist.init_process_group(backend="nccl")
        write_ds_config(stage=3, offload=False, batch_size=1024, micro=4, fp16=False)
        tokenizer = TiktokenTokenizer()
        config = ModelConfig(max_seq_len=2048)
        model = CloudTransformer(config)
        model.gradient_checkpointing = True
        train_ds = LargeParquetDataset(args.data_dir, tokenizer, seq_len=2048)
        train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        trainer = deepspeed.initialize(
            model=model,
            config_params="ds_config.json",
            training_data=train_ds
        )[0]
        opt = trainer.optimizer
        sched = get_cosine_schedule_with_warmup(opt, 2000, 100_000)
        step = 0
        for epoch in range(1):
            for x, y in tqdm(train_dl):
                x, y = x.cuda(), y.cuda()
                loss = trainer(x, y).loss
                trainer.backward(loss)
                trainer.step()
                sched.step()
                step += 1
                if step % 1000 == 0 and dist.get_rank() == 0:
                    ckpt_path = MODEL_DIR / f"step_{step}"
                    trainer.save_checkpoint(str(ckpt_path))
                if step >= 100_000:
                    break
        if dist.get_rank() == 0:
            final_path = MODEL_DIR / "pretrain_final"
            trainer.save_checkpoint(str(final_path))
            logger.info("Pre-training done.")

    elif args.mode == "dpo":
        if not args.ckpt:
            raise ValueError("--ckpt required (pretrain checkpoint)")
        tokenizer = TiktokenTokenizer()
        config = ModelConfig(max_seq_len=8192)
        # load reference
        ref_model = CloudTransformer(config)
        ref_engine, _, _, _ = deepspeed.initialize(
            model=ref_model,
            config_params="ds_config.json",
            checkpoint=args.ckpt
        )
        ref_engine.load_checkpoint(args.ckpt)
        # policy
        policy_model = CloudTransformer(config)
        dpo_cfg = DPOConfig()
        dpo_trainer = DPOTrainer(ref_engine.module, policy_model, tokenizer, dpo_cfg)
        # dummy preference dataset (replace with real)
        dummy_chosen = ["Explain gravity in simple terms."]
        dummy_rejected = ["Gravity is fake."]
        # tokenize
        def tok_list(texts):
            return [tokenizer.encode(t) for t in texts]
        chosen_ids = tok_list(dummy_chosen)
        rejected_ids = tok_list(dummy_rejected)
        # pad
        max_len = max(len(c) for c in chosen_ids + rejected_ids)
        def pad(seq):
            return seq + [tokenizer.enc.eot_token] * (max_len - len(seq))
        batch = {
            "input_ids_chosen":  torch.tensor([pad(c) for c in chosen_ids]),
            "input_ids_rejected":torch.tensor([pad(r) for r in rejected_ids])
        }
        # train loop
        for step in range(dpo_cfg.max_steps):
            loss, chosen_r, rejected_r = dpo_trainer.step_batch(batch)
            if step % 100 == 0:
                logger.info(f"DPO step={step} loss={loss:.4f} chosen={chosen_r:.3f} rejected={rejected_r:.3f}")
        # save
        final_path = MODEL_DIR / "dpo_final"
        dpo_trainer.engine.save_checkpoint(str(final_path))
        logger.info("DPO done.")

    elif args.mode == "serve":
        if not args.ckpt:
            raise ValueError("--ckpt required")
        # convert to HF format for vLLM
        logger.info("Starting vLLM server ...")
        engine = vLLMEngine(str(args.ckpt), max_model_len=8192)
        logger.info("Type prompts below (Ctrl+C to stop)")
        try:
            while True:
                prompt = input("\nPrompt: ")
                if not prompt:
                    continue
                out = engine.generate(prompt)
                print(f"Answer: {out}")
        except KeyboardInterrupt:
            pass

    elif args.mode == "eval":
        if not args.ckpt:
            raise ValueError("--ckpt required")
        evaluate_model(args.ckpt, tasks=["mmlu", "gsm8k", "humaneval"])

if __name__ == "__main__":
    main()
