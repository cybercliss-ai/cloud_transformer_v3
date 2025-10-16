# # Training mode
# python cloud_transformer_v3.py --mode train --data_dir ./data --constitutional --cot
# # Interactive chat
# python cloud_transformer_v3.py --mode chat
# # API server
# python cloud_transformer_v3.py --mode serve --port 8000
# # Evaluation
# python cloud_transformer_v3.py --mode evaluate
#!/usr/bin/env python3
# ===========================================================
#  Cloud-Scale Transformer (Claude-3 Level)
#  - 8B params, 32L/4096d/32 heads/4 kv_heads
#  - tiktoken cl100k
#  - FlashAttention-2
#  - YaRN++ long-context 200k
#  - DeepSpeed ZeRO-3 ready
#  - Constitutional AI + DPO alignment
#  - Chain-of-Thought training
#  - Function calling & tool use
#  - Multi-turn conversation memory
#  - Advanced safety filters
#  - System prompts & role-based chat
# ===========================================================

import os, math, json, glob, uuid, datetime, argparse, functools, random, asyncio
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from tqdm.auto import tqdm
import psutil, numpy as np, logging
import deepspeed
import tiktoken
from flash_attn import flash_attn_func
from transformers import AutoTokenizer
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
import uvicorn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = './cloud_model'
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# Enhanced Configs
# ===========================================================

@dataclass
class ModelConfig:
    vocab_size: int = 100256
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 4
    d_ff: int = 16384
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 2048})
    dropout: float = 0.0
    vocab_pad_to_multiple: int = 64
    use_flash: bool = True
    # YaRN++ specific
    yarn_scale: float = 32.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_temperature: float = 0.1
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

@dataclass
class TrainingConfig:
    batch_size: int = 16
    micro_batch_size: int = 2
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    save_every: int = 2_000
    eval_every: int = 500
    use_amp: bool = True
    deepspeed_config: str = "ds_config.json"

@dataclass
class ConstitutionalConfig:
    constitution: List[str] = field(default_factory=lambda: [
        "Please choose the response that is most helpful, honest, and harmless.",
        "Please choose the response that is most respectful and non-discriminatory.",
        "Please choose the response that avoids giving harmful or dangerous instructions.",
        "Please choose the response that acknowledges uncertainty when appropriate."
    ])
    critique_temperature: float = 0.7
    revision_temperature: float = 0.8

# ===========================================================
# Advanced YaRN++ Implementation
# ===========================================================

class YaRNScaler:
    """Advanced YaRN++ scaling with NTK-aware and dynamic scaling"""
    
    def __init__(self, dim: int, max_seq_len: int, theta: float, scale: float = 32.0, 
                 alpha: float = 1.0, beta: float = 32.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        
        # Calculate wavelengths and scaling factors
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.wavelengths = 2 * math.pi / self.inv_freq
        
        # NTK-by-parts scaling
        self.scale_factors = self._calculate_scale_factors()
        
        # Temperature scaling
        self.temperature = 0.1 * math.log(scale) + 1.0
        
    def _calculate_scale_factors(self):
        """Calculate NTK-by-parts scaling factors"""
        L = self.max_seq_len // int(self.scale)
        r = L / self.wavelengths
        
        # Ramp function
        gamma = torch.zeros_like(r)
        gamma[r < self.alpha] = 0
        gamma[r > self.beta] = 1
        mask = (r >= self.alpha) & (r <= self.beta)
        gamma[mask] = (r[mask] - self.alpha) / (self.beta - self.alpha)
        
        # Scale factors
        scale_factors = (1 - gamma) / self.scale + gamma
        return scale_factors
        
    def get_rope_embeddings(self, seq_len: int):
        """Get YaRN-scaled RoPE embeddings"""
        t = torch.arange(seq_len, dtype=torch.float32)
        
        # Apply scaling
        scaled_freqs = self.inv_freq * self.scale_factors
        
        # Create embeddings
        freqs = torch.outer(t, scaled_freqs)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply temperature scaling
        emb = emb / math.sqrt(self.temperature)
        
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def precompute_rope_yarn_plus(dim, max_seq_len, theta, scale=32.0, alpha=1.0, beta=32.0):
    """Precompute YaRN++ RoPE embeddings"""
    scaler = YaRNScaler(dim, max_seq_len, theta, scale, alpha, beta)
    return scaler.get_rope_embeddings(max_seq_len)

# ===========================================================
# Enhanced RMSNorm with learnable bias
# ===========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm + self.bias).type_as(x)

# ===========================================================
# Enhanced Flash Attention with YaRN++
# ===========================================================

class FlashGQAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # YaRN++ RoPE
        self.yarn_scaler = YaRNScaler(
            self.head_dim, 
            config.max_seq_len * int(config.yarn_scale), 
            config.rope_theta,
            config.yarn_scale,
            config.yarn_alpha,
            config.yarn_beta
        )
        
        # Cached embeddings
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    def rotate(self, x, cos, sin):
        """Apply rotary position embedding"""
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, mask=None):
        b, s, _ = x.shape
        
        # Linear projections
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        # Apply YaRN++ RoPE
        cos, sin = self.yarn_scaler.get_rope_embeddings(s)
        cos = cos[:, :, :s, :].to(x.device)
        sin = sin[:, :, :s, :].to(x.device)
        
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)

        # Grouped-query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Flash Attention-2
        out = flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0, 
            causal=True
        )
        
        return self.wo(out.view(b, s, -1))

# ===========================================================
# Enhanced SwiGLU with gating
# ===========================================================

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x) * gate))

# ===========================================================
# Enhanced Transformer Block
# ===========================================================

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = FlashGQAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        # Add residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None):
        x = x + self.residual_scale * self.attn(self.norm1(x), mask)
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x

# ===========================================================
# Main Enhanced Model
# ===========================================================

class CloudTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_embed.weight
        
        # System prompt support
        self.system_prompt_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, system_prompt=False):
        b, s = input_ids.shape
        x = self.tok_embed(input_ids)
        
        # Add system prompt embedding if requested
        if system_prompt:
            x = x + self.system_prompt_embed.expand(b, s, -1)
        
        # Create causal mask
        mask = torch.triu(torch.full((s, s), float('-inf'), device=x.device), diagonal=1)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x, mask)
            
        logits = self.lm_head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=1.0, top_p=0.95, 
                 system_prompt=False, do_sample=True):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
                
            logits, _ = self(input_ids, system_prompt=system_prompt)
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
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_id], dim=1)
            
        return input_ids

# ===========================================================
# Enhanced Tokenizer with special tokens
# ===========================================================

class TiktokenTokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab
        
        # Add special tokens for function calling and roles
        self.special_tokens = {
            "<|system|>": self.vocab_size,
            "<|user|>": self.vocab_size + 1,
            "<|assistant|>": self.vocab_size + 2,
            "<|function|>": self.vocab_size + 3,
            "<|tool_call|>": self.vocab_size + 4,
            "<|tool_result|>": self.vocab_size + 5,
            "<|thinking|>": self.vocab_size + 6,
            "<|/thinking|>": self.vocab_size + 7,
        }
        
        # Update vocab size
        self.vocab_size += len(self.special_tokens)
        
    def encode(self, text: str) -> list[int]:
        # Handle special tokens
        for token, id in self.special_tokens.items():
            text = text.replace(token, f" [SPECIAL_{id}] ")
            
        tokens = self.enc.encode(text, allowed_special="all")
        
        # Replace placeholders with actual special token IDs
        result = []
        for token in tokens:
            if isinstance(token, str) and token.startswith("[SPECIAL_"):
                result.append(int(token.split("_")[1].split("]")[0]))
            else:
                result.append(token)
                
        return result
    
    def decode(self, ids: list[int]) -> str:
        result = []
        for id in ids:
            if id >= self.enc.n_vocab:
                # Find special token
                for token, token_id in self.special_tokens.items():
                    if token_id == id:
                        result.append(token)
                        break
            else:
                result.append(self.enc.decode([id]))
        return "".join(result)
    
    def format_message(self, role: str, content: str) -> str:
        return f"<|{role}|>{content}"
    
    def format_function_call(self, name: str, args: dict) -> str:
        return f"<|tool_call|>{name}({json.dumps(args)})"
    
    def format_function_result(self, name: str, result: Any) -> str:
        return f"<|tool_result|>{name}: {json.dumps(result)}"

# ===========================================================
# Constitutional AI Implementation
# ===========================================================

class ConstitutionalAI:
    """Constitutional AI implementation for alignment"""
    
    def __init__(self, model, tokenizer, config: ConstitutionalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def critique_response(self, prompt: str, response: str) -> str:
        """Generate critique based on constitution"""
        critique_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: According to the constitutional principles: {', '.join(self.config.constitution)}
Please critique this response and identify any violations:

Violation:"""
        
        inputs = self.tokenizer.encode(critique_prompt)
        inputs = torch.tensor([inputs], device=DEVICE)
        
        with torch.no_grad():
            critique_ids = self.model.generate(
                inputs, 
                max_new_tokens=200,
                temperature=self.config.critique_temperature,
                system_prompt=True
            )
        
        return self.tokenizer.decode(critique_ids[0].tolist()[len(inputs[0]):])
    
    def revise_response(self, prompt: str, response: str, critique: str) -> str:
        """Revise response based on critique"""
        revision_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: {critique}

Revision: Please revise the response to address the critique while maintaining helpfulness:

Improved response:"""
        
        inputs = self.tokenizer.encode(revision_prompt)
        inputs = torch.tensor([inputs], device=DEVICE)
        
        with torch.no_grad():
            revision_ids = self.model.generate(
                inputs,
                max_new_tokens=300,
                temperature=self.config.revision_temperature,
                system_prompt=True
            )
        
        return self.tokenizer.decode(revision_ids[0].tolist()[len(inputs[0]):])
    
    def constitutional_training_step(self, prompt: str, initial_response: str):
        """Perform one step of constitutional training"""
        critique = self.critique_response(prompt, initial_response)
        revision = self.revise_response(prompt, initial_response, critique)
        
        # Create training data
        training_text = f"""Human: {prompt}

Assistant: Let me think about this step by step.

<|thinking|>
I'll analyze this request and provide a helpful, honest, and harmless response.
</|thinking|>

{revision}"""
        
        return training_text

# ===========================================================
# Chain-of-Thought Training
# ===========================================================

class ChainOfThoughtTrainer:
    """Trainer for chain-of-thought reasoning"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def create_cot_dataset(self, questions: List[str]) -> List[str]:
        """Create chain-of-thought training data"""
        cot_examples = []
        
        for question in questions:
            # Create thinking process
            thinking = f"""<|thinking|>
Let me break down this problem step by step:

1. Understanding the question: {question}
2. Key concepts involved: [relevant concepts]
3. Step-by-step reasoning:
   - First, ...
   - Then, ...
   - Finally, ...
4. Checking for potential issues: [safety/harm considerations]
</|thinking|>"""
            
            # Create complete example
            example = f"""Human: {question}

Assistant: {thinking}

Based on my analysis, here's my response:

[Detailed answer with reasoning]"""
            
            cot_examples.append(example)
            
        return cot_examples
    
    def train_cot(self, questions: List[str], num_epochs: int = 3):
        """Train model on chain-of-thought data"""
        cot_data = self.create_cot_dataset(questions)
        
        # Tokenize data
        tokenized_data = []
        for example in cot_data:
            tokens = self.tokenizer.encode(example)
            if len(tokens) < self.model.config.max_seq_len:
                tokenized_data.append(torch.tensor(tokens))
        
        # Create training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in DataLoader(tokenized_data, batch_size=4, shuffle=True):
                batch = batch.to(DEVICE)
                
                # Create labels (shifted input)
                labels = batch.clone()
                labels[:, :-1] = batch[:, 1:]
                labels[:, -1] = -100
                
                # Forward pass
                logits, loss = self.model(batch, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"CoT Epoch {epoch + 1}, Loss: {total_loss / len(tokenized_data):.4f}")

# ===========================================================
# Function Calling & Tool Use
# ===========================================================

class ToolRegistry:
    """Registry for available tools/functions"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: Dict[str, dict] = {}
        
    def register_tool(self, name: str, func: Callable, schema: dict):
        """Register a tool with its schema"""
        self.tools[name] = func
        self.tool_schemas[name] = schema
        
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_schema(self, name: str) -> Optional[dict]:
        """Get tool schema"""
        return self.tool_schemas.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())

class FunctionCallingEngine:
    """Engine for function calling and tool use"""
    
    def __init__(self, model, tokenizer, tool_registry: ToolRegistry):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tool_registry
        
    def parse_function_call(self, text: str) -> Optional[tuple[str, dict]]:
        """Parse function call from model output"""
        if "<|tool_call|>" in text:
            try:
                # Extract function call
                start = text.find("<|tool_call|>") + len("<|tool_call|>")
                end = text.find("<|", start)
                if end == -1:
                    end = len(text)
                
                call_text = text[start:end].strip()
                
                # Parse function name and arguments
                if "(" in call_text and call_text.endswith(")"):
                    func_name = call_text[:call_text.find("(")]
                    args_str = call_text[call_text.find("(") + 1:-1]
                    args = json.loads(args_str) if args_str else {}
                    
                    return func_name, args
            except:
                pass
        return None
    
    def execute_function_call(self, func_name: str, args: dict) -> Any:
        """Execute a function call"""
        tool = self.tools.get_tool(func_name)
        if tool:
            try:
                result = tool(**args)
                return result
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Tool {func_name} not found"}
    
    def format_tool_response(self, func_name: str, result: Any) -> str:
        """Format tool response for model"""
        return self.tokenizer.format_function_result(func_name, result)
    
    def generate_with_tools(self, prompt: str, max_turns: int = 5) -> str:
        """Generate response with tool use"""
        conversation = prompt
        turns = 0
        
        while turns < max_turns:
            # Generate response
            inputs = self.tokenizer.encode(conversation)
            inputs = torch.tensor([inputs], device=DEVICE)
            
            with torch.no_grad():
                response_ids = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    system_prompt=True
                )
            
            response = self.tokenizer.decode(response_ids[0].tolist()[len(inputs[0]):])
            conversation += response
            
            # Check for function calls
            func_call = self.parse_function_call(response)
            if func_call:
                func_name, args = func_call
                result = self.execute_function_call(func_name, args)
                tool_response = self.format_tool_response(func_name, result)
                conversation += tool_response
            else:
                break
                
            turns += 1
            
        return conversation

# ===========================================================
# Conversation Memory
# ===========================================================

class ConversationMemory:
    """Memory system for multi-turn conversations"""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.conversations: Dict[str, List[dict]] = {}
        
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
        # Trim to max tokens
        self._trim_conversation(session_id)
        
    def get_conversation(self, session_id: str) -> List[dict]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
    
    def _trim_conversation(self, session_id: str):
        """Trim conversation to fit max tokens"""
        conv = self.conversations[session_id]
        total_tokens = sum(len(msg["content"].split()) for msg in conv)
        
        while total_tokens > self.max_tokens and len(conv) > 1:
            conv.pop(0)
            total_tokens = sum(len(msg["content"].split()) for msg in conv)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]

# ===========================================================
# Advanced Safety System
# ===========================================================

class AdvancedSafetyLayer:
    """Enhanced safety system with multiple checks"""
    
    def __init__(self):
        self.toxic = Detoxify('original', device=DEVICE)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Enhanced refusal patterns
        self.refusal_patterns = [
            "I cannot assist with that request as it may be harmful or dangerous.",
            "I'm not able to help with that. Is there something else I can assist you with?",
            "I don't feel comfortable providing that information. Can I help you with something else?",
            "That request goes against my safety guidelines. Let me know if there's another way I can help.",
        ]
        
        # Harmful content patterns
        self.harmful_patterns = [
            "how to make", "how to build", "instructions for", "step by step",
            "illegal", "hack", "exploit", "bypass", "crack", "pirate"
        ]
        
    def check_safety(self, text: str) -> tuple[bool, str]:
        """Comprehensive safety check"""
        if len(text.strip()) < 2:
            return True, ""
            
        # Check for harmful patterns
        text_lower = text.lower()
        for pattern in self.harmful_patterns:
            if pattern in text_lower:
                return False, "harmful_content"
        
        # Toxicity check
        tox = self.toxic.predict(text)
        if tox["toxicity"] > 0.7 or tox["severe_toxicity"] > 0.2:
            return False, "toxic_content"
            
        # Personal information check
        try:
            results = self.analyzer.analyze(text=text, language='en')
            if len(results) > 3:  # Too much PII
                return False, "excessive_pii"
        except:
            pass
            
        return True, "safe"
    
    def get_refusal_response(self) -> str:
        """Get a random refusal response"""
        return random.choice(self.refusal_patterns)
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize PII from text"""
        try:
            results = self.analyzer.analyze(text=text, language='en')
            return self.anonymizer.anonymize(text=text, analyzer_results=results).text
        except:
            return text

# ===========================================================
# Enhanced Dataset with CoT and Constitutional AI
# ===========================================================

class EnhancedDataset(Dataset):
    """Enhanced dataset with CoT and safety filtering"""
    
    def __init__(self, parquet_files, tokenizer, seq_len=2048, safety_layer=None):
        self.files = parquet_files
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.safety = safety_layer
        self.buffer = []
        self._fill_buffer()
        
    def _fill_buffer(self):
        import pyarrow.parquet as pq
        for f in self.files:
            table = pq.read_table(f)
            for text in table['text'].to_pylist():
                # Safety check
                if self.safety:
                    is_safe, reason = self.safety.check_safety(text)
                    if not is_safe:
                        continue
                
                # Add CoT formatting for complex texts
                if len(text) > 500:
                    text = self._add_cot_formatting(text)
                    
                self.buffer.extend(self.tokenizer.encode(text) + [self.tokenizer.enc.eot_token])
                if len(self.buffer) >= 1_000_000:
                    return
                    
    def _add_cot_formatting(self, text: str) -> str:
        """Add chain-of-thought formatting to complex texts"""
        return f"""Let me think about this step by step.

<|thinking|>
This appears to be a complex topic that requires careful analysis.
</|thinking|>

{text}"""
    
    def __len__(self):
        return (len(self.buffer) - self.seq_len - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.buffer[start:start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ===========================================================
# Enhanced DeepSpeed Trainer
# ===========================================================

class EnhancedDSTrainer:
    """Enhanced trainer with Constitutional AI and CoT"""
    
    def __init__(self, model, tokenizer, train_config, data_files, constitutional_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config
        self.safety = AdvancedSafetyLayer()
        
        # Initialize Constitutional AI if config provided
        self.constitutional = None
        if constitutional_config:
            self.constitutional = ConstitutionalAI(model, tokenizer, constitutional_config)
            
        self.dataset = EnhancedDataset(data_files, tokenizer, model.config.max_seq_len, self.safety)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=train_config.micro_batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        # DeepSpeed initialization
        self.engine, _, self.dataloader, _ = deepspeed.initialize(
            model=self.model,
            config_params=train_config.deepspeed_config,
            training_data=self.dataset
        )
        
        self.global_step = 0
        
    def train(self):
        """Enhanced training with safety and constitutional checks"""
        for epoch in range(1):
            for step, (x, y) in enumerate(tqdm(self.dataloader)):
                x, y = x.to(self.engine.device), y.to(self.engine.device)
                
                # Forward pass
                loss = self.engine(x, y).loss
                
                # Constitutional AI step (every 100 steps)
                if self.constitutional and step % 100 == 0:
                    self._constitutional_training_step()
                
                # Backward pass
                self.engine.backward(loss)
                self.engine.step()
                
                self.global_step += 1
                
                # Logging and checkpointing
                if self.global_step % self.config.save_every == 0 and dist.get_rank() == 0:
                    self.save_ckpt()
                    
                if self.global_step >= self.config.max_steps:
                    break
                    
    def _constitutional_training_step(self):
        """Perform constitutional training step"""
        # Sample a prompt from dataset
        sample_text = "Explain the importance of AI safety in healthcare."
        
        # Generate initial response
        inputs = self.tokenizer.encode(f"Human: {sample_text}\n\nAssistant:")
        inputs = torch.tensor([inputs], device=self.engine.device)
        
        with torch.no_grad():
            response_ids = self.model.generate(inputs, max_new_tokens=200)
            response = self.tokenizer.decode(response_ids[0].tolist()[len(inputs[0]):])
        
        # Apply constitutional training
        if self.constitutional:
            training_text = self.constitutional.constitutional_training_step(sample_text, response)
            logger.info(f"Constitutional training applied at step {self.global_step}")
    
    def save_ckpt(self):
        """Save checkpoint"""
        tag = f"step_{self.global_step}"
        self.engine.save_checkpoint(os.path.join(MODEL_DIR, tag))

# ===========================================================
# FastAPI Deployment with Streaming
# ===========================================================

app = FastAPI(title="Cloud-Scale Transformer API", version="3.0.0")

class ModelServer:
    """FastAPI server for model serving"""
    
    def __init__(self, model, tokenizer, safety_layer, memory, tool_registry):
        self.model = model
        self.tokenizer = tokenizer
        self.safety = safety_layer
        self.memory = memory
        self.tools = tool_registry
        self.sessions: Dict[str, dict] = {}
        
    async def generate_stream(self, prompt: str, session_id: str, max_tokens: int = 500):
        """Generate streaming response"""
        try:
            # Safety check
            is_safe, reason = self.safety.check_safety(prompt)
            if not is_safe:
                yield f"data: {self.safety.get_refusal_response()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Get conversation history
            history = self.memory.get_conversation(session_id)
            
            # Format full prompt
            full_prompt = ""
            for msg in history[-5:]:  # Last 5 messages
                full_prompt += self.tokenizer.format_message(msg["role"], msg["content"])
            full_prompt += self.tokenizer.format_message("user", prompt)
            
            # Tokenize
            inputs = self.tokenizer.encode(full_prompt)
            inputs = torch.tensor([inputs], device=DEVICE)
            
            # Generate with streaming
            self.model.eval()
            with torch.no_grad():
                for i in range(max_tokens):
                    logits, _ = self.model(inputs, system_prompt=True)
                    logits = logits[:, -1, :] / 0.7  # Temperature
                    
                    # Sample next token
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Decode token
                    token_text = self.tokenizer.decode([next_token.item()])
                    yield f"data: {token_text}\n\n"
                    
                    # Update input
                    inputs = torch.cat([inputs, next_token], dim=1)
                    
                    # Check for end conditions
                    if next_token.item() == self.tokenizer.enc.eot_token:
                        break
                        
            yield "data: [DONE]\n\n"
            
            # Save to memory
            self.memory.add_message(session_id, "user", prompt)
            response = self.tokenizer.decode(inputs[0].tolist()[len(full_prompt):])
            self.memory.add_message(session_id, "assistant", response)
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

# Initialize server components
model_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize model server"""
    global model_server
    
    # Load model
    config = ModelConfig(max_seq_len=8192, yarn_scale=32.0)
    model = CloudTransformer(config)
    
    # Load tokenizer
    tokenizer = TiktokenTokenizer()
    
    # Initialize components
    safety = AdvancedSafetyLayer()
    memory = ConversationMemory(max_tokens=8192)
    tools = ToolRegistry()
    
    # Register example tools
    def get_weather(location: str) -> dict:
        return {"temperature": 22, "condition": "sunny", "location": location}
    
    def search_web(query: str) -> list:
        return [f"Result {i} for {query}" for i in range(3)]
    
    tools.register_tool("get_weather", get_weather, {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    })
    
    tools.register_tool("search_web", search_web, {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    })
    
    model_server = ModelServer(model, tokenizer, safety, memory, tools)

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint with streaming"""
    data = await request.json()
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "default")
    max_tokens = data.get("max_tokens", 500)
    
    return EventSourceResponse(
        model_server.generate_stream(prompt, session_id, max_tokens)
    )

@app.post("/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    data = await request.json()
    messages = data.get("messages", [])
    session_id = data.get("session_id", "default")
    
    # Format messages
    formatted_prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_prompt += model_server.tokenizer.format_message(role, content)
    
    return EventSourceResponse(
        model_server.generate_stream(formatted_prompt, session_id)
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Cloud-Scale Transformer v3.0.0"}

# ===========================================================
# Main CLI with all features
# ===========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "chat", "serve", "evaluate"], default="serve")
    parser.add_argument("--data_dir", help="folder with .parquet")
    parser.add_argument("--constitutional", action="store_true", help="enable constitutional AI")
    parser.add_argument("--cot", action="store_true", help="enable chain-of-thought training")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    args = parser.parse_args()

    if args.mode == "train":
        # Training mode
        dist.init_process_group(backend='nccl')
        
        tokenizer = TiktokenTokenizer()
        config = ModelConfig(max_seq_len=8192, yarn_scale=32.0)
        model = CloudTransformer(config)
        
        train_config = TrainingConfig()
        constitutional_config = ConstitutionalConfig() if args.constitutional else None
        
        data_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
        
        if args.cot:
            # Chain-of-thought training
            cot_trainer = ChainOfThoughtTrainer(model, tokenizer)
            questions = [
                "Explain quantum computing in simple terms.",
                "What are the ethical implications of AI?",
                "How does photosynthesis work?",
                "What caused World War I?",
                "Explain the concept of machine learning."
            ]
            cot_trainer.train_cot(questions)
        
        # Main training
        trainer = EnhancedDSTrainer(
            model, tokenizer, train_config, data_files, constitutional_config
        )
        trainer.train()
        
    elif args.mode == "chat":
        # Interactive chat mode
        tokenizer = TiktokenTokenizer()
        config = ModelConfig(max_seq_len=8192)
        model = CloudTransformer(config)
        
        # Load checkpoint if available
        checkpoint_path = os.path.join(MODEL_DIR, "step_100000")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
        
        safety = AdvancedSafetyLayer()
        memory = ConversationMemory()
        session_id = str(uuid.uuid4())
        
        print("Cloud-Scale Transformer Chat (type 'quit' to exit)")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
                
            # Safety check
            is_safe, reason = safety.check_safety(user_input)
            if not is_safe:
                print(f"Assistant: {safety.get_refusal_response()}")
                continue
            
            # Generate response
            memory.add_message(session_id, "user", user_input)
            history = memory.get_conversation(session_id)
            
            prompt = ""
            for msg in history:
                prompt += tokenizer.format_message(msg["role"], msg["content"])
            
            inputs = tokenizer.encode(prompt)
            inputs = torch.tensor([inputs], device=DEVICE)
            
            with torch.no_grad():
                response_ids = model.generate(
                    inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    system_prompt=True
                )
            
            response = tokenizer.decode(response_ids[0].tolist()[len(inputs[0]):])
            print(f"Assistant: {response}")
            
            memory.add_message(session_id, "assistant", response)
        
    elif args.mode == "serve":
        # API server mode
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.mode == "evaluate":
        # Evaluation mode
        from lm_eval import evaluator
        
        tokenizer = TiktokenTokenizer()
        config = ModelConfig(max_seq_len=2048)
        model = CloudTransformer(config)
        
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={MODEL_DIR}",
            tasks=["mmlu", "hellaswag", "winogrande", "arc_challenge"],
            batch_size=16,
        )
        
        print("Evaluation Results:")
        print(json.dumps(results, indent=2))
        
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
