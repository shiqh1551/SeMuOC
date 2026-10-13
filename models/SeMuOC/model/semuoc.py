import hashlib
import math
import os
import threading
import time
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from torch import nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


_SENTENCE_TRANSFORMER_LOAD_LOCK = threading.Lock()
_SENTENCE_TRANSFORMER_ENCODE_LOCK = threading.Lock()
_SHARED_SENTENCE_TRANSFORMERS = {}
_GLOBAL_EMBEDDING_CACHE = {}
_PRECOMPUTED_PROMPT_TABLES = {}
_PRECOMPUTED_ANOMALY_TABLES = {}
_PRECOMPUTED_TABLE_LOCK = threading.Lock()


def _sentence_transformer_source(model_name):
    local_model = "embedding/e5-base-v2"
    return local_model if os.path.exists(local_model) else model_name


def _get_shared_sentence_transformer(model_name):
    source = _sentence_transformer_source(model_name)
    with _SENTENCE_TRANSFORMER_LOAD_LOCK:
        if source not in _SHARED_SENTENCE_TRANSFORMERS:
            _SHARED_SENTENCE_TRANSFORMERS[source] = SentenceTransformer(source)
        return source, _SHARED_SENTENCE_TRANSFORMERS[source]


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class TemporalEncoder(nn.Module):
    """Encode each variable as a temporal-space representation."""

    def __init__(self, d_model=128, layers=3):
        super(TemporalEncoder, self).__init__()
        self.proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, dilation=2 ** idx) for idx in range(layers)
        ])

    def forward(self, x):
        # x: [B, T, N]
        batch_size, window_size, channel_num = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * channel_num, window_size, 1)
        x = self.proj(x).permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=-1).reshape(batch_size, channel_num, -1)


class BehaviorPromptGenerator:
    """Turn numeric channel behavior into structured behavior language."""

    STATE_COUNT = 24

    def __init__(self, domain_text="", variable_desc=None):
        self.domain_text = domain_text
        self.variable_desc = variable_desc or {}

    @staticmethod
    def _summarize_channels(x):
        # x: [B, T, N]. Keep numeric work on the current device when possible.
        values = x.detach().float()
        batch_size, window_size, channel_num = values.shape
        mean = values.mean(dim=1)
        std = values.std(dim=1, unbiased=False) + 1e-6
        slope = (values[:, -1, :] - values[:, 0, :]) / (window_size + 1e-6)
        diff = values[:, 1:, :] - values[:, :-1, :]
        burst = diff.abs().amax(dim=1) / std if window_size > 1 else torch.zeros_like(std)

        fft = torch.fft.rfft(values - mean.unsqueeze(1), dim=1).abs()
        if fft.size(1) > 2:
            periodicity = fft[:, 1:, :].amax(dim=1) / (fft.sum(dim=1) + 1e-6)
        else:
            periodicity = torch.zeros(batch_size, channel_num, device=values.device)

        trend = torch.full_like(std, 2, dtype=torch.long)
        trend = torch.where(slope > 0.01 * std, torch.zeros_like(trend), trend)
        trend = torch.where(slope < -0.01 * std, torch.ones_like(trend), trend)
        return (
            trend.cpu().tolist(),
            (std > 0.5).cpu().tolist(),
            (burst > 3).cpu().tolist(),
            (periodicity > 0.25).cpu().tolist(),
        )

    @staticmethod
    def _behavior_text(trend, high_variance, local_burst, periodic):
        trend_text = ["an upward trend", "a downward trend", "a stable trend"][trend]
        variance_text = "high variance" if high_variance else "low variance"
        burst_text = "abrupt local bursts" if local_burst else "smooth local changes"
        periodic_text = "clear periodic behavior" if periodic else "no clear periodic pattern"
        return (
            "The recent window shows "
            f"{trend_text}, {variance_text}, {burst_text}, and {periodic_text}."
        )

    @staticmethod
    def state_id(trend, high_variance, local_burst, periodic):
        return trend * 8 + int(high_variance) * 4 + int(local_burst) * 2 + int(periodic)

    @classmethod
    def prompt_id(cls, channel_idx, trend, high_variance, local_burst, periodic):
        return channel_idx * cls.STATE_COUNT + cls.state_id(
            trend, high_variance, local_burst, periodic
        )

    def all_prompts(self, channel_num):
        prompts = []
        for channel_idx in range(channel_num):
            var_name = f"Channel-{channel_idx}"
            desc = self.variable_desc.get(var_name, "unknown variable")
            for trend in range(3):
                for high_variance in [False, True]:
                    for local_burst in [False, True]:
                        for periodic in [False, True]:
                            behavior = self._behavior_text(
                                trend, high_variance, local_burst, periodic
                            )
                            prompts.append(
                                f"Context: {self.domain_text}\n"
                                f"Variable: {var_name}, {desc}.\n"
                                f"Observed behavior: {behavior}\n"
                                "Compare this behavior with known normal and anomalous time-series patterns."
                            )
        return prompts

    def __call__(self, x):
        # x: [B, T, N]
        batch_size, _, channel_num = x.shape
        with torch.no_grad():
            try:
                trend, high_variance, local_burst, periodic = self._summarize_channels(x)
            except RuntimeError:
                trend, high_variance, local_burst, periodic = self._summarize_channels(x.cpu())
        prompt_ids = torch.empty(batch_size, channel_num, dtype=torch.long)
        for batch_idx in range(batch_size):
            for channel_idx in range(channel_num):
                prompt_ids[batch_idx, channel_idx] = self.prompt_id(
                    channel_idx,
                    trend[batch_idx][channel_idx],
                    high_variance[batch_idx][channel_idx],
                    local_burst[batch_idx][channel_idx],
                    periodic[batch_idx][channel_idx],
                )
        return prompt_ids.to(x.device)


class FrozenHashTextEncoder(nn.Module):
    """Deterministic fallback when sentence-transformers is unavailable."""

    def __init__(self, text_dim=384):
        super(FrozenHashTextEncoder, self).__init__()
        self.text_dim = text_dim

    def get_sentence_embedding_dimension(self):
        return self.text_dim

    @torch.no_grad()
    def encode(self, prompts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False):
        del show_progress_bar
        rows = []
        for prompt in prompts:
            values = torch.zeros(self.text_dim)
            for token in prompt.split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.text_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                values[idx] += sign
            if normalize_embeddings:
                values = F.normalize(values, dim=0)
            rows.append(values)
        output = torch.stack(rows, dim=0)
        if convert_to_tensor:
            return output
        return output.numpy()


class FrozenTextEncoder(nn.Module):
    """Frozen LLM/text embedding encoder with a trainable projection head."""

    def __init__(
        self,
        model_name="intfloat/e5-base-v2",
        out_dim=128,
        fallback_dim=384,
        allow_hash_fallback=False,
        enable_cache=True,
        cache_on_device=True,
    ):
        super(FrozenTextEncoder, self).__init__()
        self.model_name = model_name
        self.uses_hash_encoder = False
        self.prompt_prefix = "query: " if "e5" in model_name.lower() else ""
        self.enable_cache = enable_cache
        self.cache_on_device = cache_on_device
        self.embedding_cache = {}
        if model_name in ["hash", "fallback", "none"]:
            self._set_encoder(FrozenHashTextEncoder(fallback_dim))
            self.uses_hash_encoder = True
        else:
            if SentenceTransformer is None:
                if allow_hash_fallback:
                    self._set_encoder(FrozenHashTextEncoder(fallback_dim))
                    self.uses_hash_encoder = True
                else:
                    raise ImportError(
                        "sentence-transformers is required to load text_model="
                        f"{model_name!r}. Install a compatible sentence-transformers "
                        "version or set text_model='hash'. Original import error: "
                        f"{SENTENCE_TRANSFORMERS_IMPORT_ERROR}"
                    ) from SENTENCE_TRANSFORMERS_IMPORT_ERROR
            else:
                try:
                    source, encoder = _get_shared_sentence_transformer(model_name)
                    self.model_name = source
                    self._set_encoder(encoder)
                except Exception as exc:
                    if allow_hash_fallback:
                        self._set_encoder(FrozenHashTextEncoder(fallback_dim))
                        self.uses_hash_encoder = True
                    else:
                        raise RuntimeError(
                            f"Failed to load SentenceTransformer model {model_name!r}. "
                            "Make sure the model is installed/cached or network access is available."
                        ) from exc
        if hasattr(self.encoder, "get_embedding_dimension"):
            text_dim = self.encoder.get_embedding_dimension()
        else:
            text_dim = self.encoder.get_sentence_embedding_dimension()
        self.proj = nn.Linear(text_dim, out_dim)

    @property
    def encoder(self):
        return self._encoder_ref[0]

    def _set_encoder(self, encoder):
        if hasattr(encoder, "eval"):
            encoder.eval()
        if hasattr(encoder, "parameters"):
            for param in encoder.parameters():
                param.requires_grad_(False)
        self._encoder_ref = [encoder]

    def _prepare_prompts(self, prompts):
        if not self.prompt_prefix:
            return prompts
        return [
            prompt if prompt.startswith(("query:", "passage:"))
            else self.prompt_prefix + prompt
            for prompt in prompts
        ]

    @torch.no_grad()
    def _encode_uncached(self, prompts, device, cache_device=None):
        encode_kwargs = {
            "convert_to_tensor": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if not self.uses_hash_encoder:
            encode_kwargs["device"] = str(device)
        try:
            with _SENTENCE_TRANSFORMER_ENCODE_LOCK:
                z = self.encoder.encode(prompts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("device", None)
            with _SENTENCE_TRANSFORMER_ENCODE_LOCK:
                z = self.encoder.encode(prompts, **encode_kwargs)
        if cache_device is None:
            cache_device = torch.device("cpu")
        return z.clone().detach().to(cache_device)

    @torch.no_grad()
    def encode(self, prompts, device):
        prompts = self._prepare_prompts(prompts)
        if not self.enable_cache:
            return self._encode_uncached(prompts, device, cache_device=device)

        cache_device = device if self.cache_on_device else torch.device("cpu")

        cache = _GLOBAL_EMBEDDING_CACHE if self.enable_cache else self.embedding_cache
        missing_prompts = []
        seen_missing = set()
        for prompt in prompts:
            cache_key = (self.model_name, prompt)
            if cache_key not in cache and prompt not in seen_missing:
                missing_prompts.append(prompt)
                seen_missing.add(prompt)

        if missing_prompts:
            z_missing = self._encode_uncached(missing_prompts, device, cache_device)
            for prompt, row in zip(missing_prompts, z_missing):
                cache[(self.model_name, prompt)] = row.clone().detach()

        z = torch.stack([cache[(self.model_name, prompt)] for prompt in prompts], dim=0)
        return z if z.device == device else z.to(device)

    def forward(self, prompts, batch_size, channel_num, device):
        z = self.encode(prompts, device)
        z = self.proj(z)
        return z.reshape(batch_size, channel_num, -1)


class PrecomputedPromptEmbedding(nn.Module):
    """Precompute behavior prompt embeddings and use tensor lookup in forward."""

    def __init__(self, text_encoder, prompt_gen, channel_num):
        super(PrecomputedPromptEmbedding, self).__init__()
        self.text_encoder = text_encoder
        self.channel_num = channel_num
        prompts = prompt_gen.all_prompts(channel_num)
        cache_key = (text_encoder.model_name, tuple(prompts))
        with _PRECOMPUTED_TABLE_LOCK:
            if cache_key not in _PRECOMPUTED_PROMPT_TABLES:
                raw = text_encoder.encode(prompts, torch.device("cpu")).cpu()
                _PRECOMPUTED_PROMPT_TABLES[cache_key] = raw.clone().detach()
            raw_table = _PRECOMPUTED_PROMPT_TABLES[cache_key]
        self.register_buffer("raw_table", raw_table.clone().detach())

    def forward(self, prompt_ids):
        batch_size, channel_num = prompt_ids.shape
        z = self.raw_table[prompt_ids.reshape(-1)]
        z = self.text_encoder.proj(z)
        return z.reshape(batch_size, channel_num, -1)


class RelationAwareAttention(nn.Module):
    """Inject variable-relation awareness in the semantic space."""

    def __init__(self, d_model=128):
        super(RelationAwareAttention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, z):
        query, key, value = self.q(z), self.k(z), self.v(z)
        attn = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(z.size(-1))
        prior = torch.matmul(
            F.normalize(z, dim=-1),
            F.normalize(z, dim=-1).transpose(-1, -2),
        )
        attn = F.softmax(attn + prior, dim=-1)
        return self.o(torch.matmul(attn, value))


class CrossModalFusion(nn.Module):
    """Cross-attend temporal representations to semantic representations."""

    def __init__(self, d_model=128, heads=4, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z_temporal, z_semantic):
        z, attn_weights = self.attn(
            z_temporal, z_semantic, z_semantic, need_weights=True
        )
        z = self.norm1(z_temporal + z)
        z = self.norm2(z + self.ffn(z))
        return z, attn_weights


class ReconstructionDecoder(nn.Module):
    """Reconstruct each variable window from unified channel features."""

    def __init__(self, d_model=128, window_size=128, dropout=0.1):
        super(ReconstructionDecoder, self).__init__()
        self.window_size = window_size
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, window_size),
        )

    def forward(self, z):
        # z: [B, N, d] -> x_hat: [B, T, N]
        return self.decoder(z).transpose(1, 2)


class KnowledgeAnomalyMemory(nn.Module):
    def __init__(self, text_encoder, anomaly_texts, d_model=128):
        super(KnowledgeAnomalyMemory, self).__init__()
        self.text_encoder = text_encoder
        self.anomaly_texts = anomaly_texts
        self.d_model = d_model
        self.register_buffer("memory", torch.empty(0, d_model))
        if len(self.anomaly_texts) == 0:
            self.register_buffer("raw_anomaly_table", torch.empty(0))
        else:
            anomaly_prompts = [
                text if text.startswith(("query:", "passage:")) else "passage: " + text
                for text in self.anomaly_texts
            ]
            cache_key = (text_encoder.model_name, tuple(anomaly_prompts))
            with _PRECOMPUTED_TABLE_LOCK:
                if cache_key not in _PRECOMPUTED_ANOMALY_TABLES:
                    raw = text_encoder.encode(anomaly_prompts, torch.device("cpu")).cpu()
                    _PRECOMPUTED_ANOMALY_TABLES[cache_key] = raw.clone().detach()
                raw_table = _PRECOMPUTED_ANOMALY_TABLES[cache_key]
            self.register_buffer("raw_anomaly_table", raw_table.clone().detach())

    @torch.no_grad()
    def build(self, device):
        if len(self.anomaly_texts) == 0:
            self.memory = F.normalize(torch.zeros(1, self.d_model, device=device), dim=-1)
            return
        z = self.raw_anomaly_table.to(device)
        z = self.text_encoder.proj(z)
        self.memory = F.normalize(z, dim=-1)

    def forward(self, z):
        if self.memory.numel() == 0:
            self.build(z.device)
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, self.memory.t())
        return sim.max(dim=-1).values.mean(dim=-1), sim


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use stored statistics for a singleton tail batch in the new OC head."""

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=self.momentum, eps=self.eps,
            )
        return super().forward(x)


class OneClassProjectionHead(nn.Module):
    """AOC-style projection of a fused window; neither q nor c is L2-normalized."""

    def __init__(self, channel_num, d_model, project_dim=32, hidden_dim=None, pooling="flatten"):
        super().__init__()
        if pooling not in {"flatten", "mean"}:
            raise ValueError("oc_pooling must be 'flatten' or 'mean'")
        hidden_dim = max(1, d_model // 2) if hidden_dim is None else hidden_dim
        if project_dim < 1 or hidden_dim < 1:
            raise ValueError("OC projection dimensions must be positive")
        self.pooling = pooling
        input_dim = channel_num * d_model if pooling == "flatten" else d_model
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            SingletonSafeBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, project_dim, bias=False),
        )

    def forward(self, z_unified):
        window = z_unified.flatten(start_dim=1) if self.pooling == "flatten" else z_unified.mean(dim=1)
        return self.projection(window)


class SeMuOC(nn.Module):
    def __init__(
        self,
        d_model=128,
        tcn_layers=3,
        attention_heads=4,
        project_dim=32,
        oc_hidden_dim=None,
        oc_pooling="flatten",
        window_size=128,
        dropout=0.1,
        text_model="intfloat/e5-base-v2",
        allow_hash_text_encoder=False,
        text_embedding_cache=True,
        text_cache_on_device=True,
        profile_timing=False,
        input_channels=None,
        domain_text="",
        variable_desc=None,
        anomaly_texts=None,
    ):
        super(SeMuOC, self).__init__()
        self.profile_timing = profile_timing
        self.timing = defaultdict(float)
        self.temporal = TemporalEncoder(d_model, tcn_layers)
        channel_num = input_channels or len(variable_desc or {}) or 1
        self.prompt_gen = BehaviorPromptGenerator(domain_text, variable_desc)
        self.text_encoder = FrozenTextEncoder(
            text_model,
            d_model,
            allow_hash_fallback=allow_hash_text_encoder,
            enable_cache=text_embedding_cache,
            cache_on_device=text_cache_on_device,
        )
        self.prompt_embedding = PrecomputedPromptEmbedding(
            self.text_encoder,
            self.prompt_gen,
            channel_num,
        )
        self.relation = RelationAwareAttention(d_model)
        self.fusion = CrossModalFusion(d_model, attention_heads, dropout)
        self.reconstruction = ReconstructionDecoder(d_model, window_size, dropout)
        self.one_class_head = OneClassProjectionHead(
            channel_num, d_model, project_dim, oc_hidden_dim, oc_pooling,
        )
        self.register_buffer("oc_center", torch.zeros(project_dim))
        self.register_buffer("oc_center_initialized", torch.tensor(False))

    def forward(self, x, x_context=None):
        del x_context
        # x: [B, T, N]
        batch_size, _, channel_num = x.shape
        device = x.device

        start = time.perf_counter()
        z_temporal = self.temporal(x.float())
        if self.profile_timing:
            _sync_if_cuda(device)
            self.timing["temporal"] += time.perf_counter() - start

        start = time.perf_counter()
        prompt_ids = self.prompt_gen(x)
        if self.profile_timing:
            self.timing["prompt_gen"] += time.perf_counter() - start

        start = time.perf_counter()
        z_semantic = self.prompt_embedding(prompt_ids)
        if self.profile_timing:
            _sync_if_cuda(device)
            self.timing["prompt_lookup_project"] += time.perf_counter() - start

        start = time.perf_counter()
        z_semantic = self.relation(z_semantic)
        z_unified, cross_attn = self.fusion(z_temporal, z_semantic)
        if self.profile_timing:
            _sync_if_cuda(device)
            self.timing["relation_fusion"] += time.perf_counter() - start

        start = time.perf_counter()
        x_recon = self.reconstruction(z_unified)
        q = self.one_class_head(z_unified)
        oc_distance = (q - self.oc_center).square().mean(dim=-1)
        if self.profile_timing:
            _sync_if_cuda(device)
            self.timing["decode_one_class"] += time.perf_counter() - start

        return {
            "x_recon": x_recon,
            "q": q,
            "oc_distance": oc_distance,
            "z_temporal": z_temporal,
            "z_semantic": z_semantic,
            "z_unified": z_unified,
            "cross_attention": cross_attn,
            "prompt_ids": prompt_ids,
        }

    @torch.no_grad()
    def set_oc_center(self, center, eps=0.1):
        """Install a detached training-only center, including an exact-zero guard."""
        eps = float(eps)
        if eps < 0:
            raise ValueError("center_eps must be nonnegative")
        center = torch.as_tensor(center, device=self.oc_center.device, dtype=self.oc_center.dtype)
        if center.shape != self.oc_center.shape or not torch.isfinite(center).all():
            raise ValueError("OC center must be finite and match project_dim")
        center = center.detach().clone()
        small = center.abs() < eps
        center[small] = torch.where(center[small] < 0, -eps, eps)
        self.oc_center.copy_(center)
        self.oc_center_initialized.fill_(True)


class base_Model(SeMuOC):
    """Repository entry point for SeMuOC."""

    def __init__(self, configs, device):
        del device
        d_model = getattr(configs, "d_model", getattr(configs, "latent_dim", 128))
        domain_text = getattr(configs, "domain_text", None) or (
            "This is a multivariate time-series monitoring system. Variables may "
            "represent sensor readings, actuator states, workload indicators, or "
            "other operational measurements. Normal behavior should preserve "
            "reasonable temporal dynamics and cross-variable consistency."
        )
        super(base_Model, self).__init__(
            d_model=d_model,
            tcn_layers=getattr(configs, "tcn_layers", 3),
            attention_heads=getattr(configs, "attention_heads", getattr(configs, "num_heads", 4)),
            project_dim=getattr(configs, "project_dim", 32),
            oc_hidden_dim=getattr(configs, "oc_hidden_dim", None),
            oc_pooling=getattr(configs, "oc_pooling", "flatten"),
            window_size=getattr(configs, "window_size", 128),
            dropout=getattr(configs, "dropout", 0.1),
            text_model=getattr(configs, "text_model", "intfloat/e5-base-v2"),
            allow_hash_text_encoder=getattr(configs, "allow_hash_text_encoder", False),
            text_embedding_cache=getattr(configs, "text_embedding_cache", True),
            text_cache_on_device=getattr(configs, "text_cache_on_device", True),
            profile_timing=getattr(configs, "profile_timing", False),
            input_channels=getattr(configs, "input_channels", None),
            domain_text=domain_text,
            variable_desc=getattr(configs, "variable_desc", None),
            anomaly_texts=None,
        )


def warmup_precomputed_text_tables(configs):
    """Build shared text encoder and precomputed prompt tables before parallel trials."""
    d_model = getattr(configs, "d_model", getattr(configs, "latent_dim", 128))
    domain_text = getattr(configs, "domain_text", None) or (
        "This is a multivariate time-series monitoring system. Variables may "
        "represent sensor readings, actuator states, workload indicators, or "
        "other operational measurements. Normal behavior should preserve "
        "reasonable temporal dynamics and cross-variable consistency."
    )
    prompt_gen = BehaviorPromptGenerator(
        domain_text,
        getattr(configs, "variable_desc", None),
    )
    text_encoder = FrozenTextEncoder(
        getattr(configs, "text_model", "intfloat/e5-base-v2"),
        d_model,
        allow_hash_fallback=getattr(configs, "allow_hash_text_encoder", False),
        enable_cache=getattr(configs, "text_embedding_cache", True),
        cache_on_device=False,
    )
    channel_num = (
        getattr(configs, "input_channels", None)
        or len(getattr(configs, "variable_desc", None) or {})
        or 1
    )
    PrecomputedPromptEmbedding(text_encoder, prompt_gen, channel_num)
