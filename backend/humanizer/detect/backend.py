"""Local language-model backend for the Tier A statistical detectors.

One forward pass per model per window produces every accumulator that
Fast-DetectGPT, Binoculars, and plain perplexity need. Running them separately
would cost 3x for no benefit.

Memory discipline matters here: Qwen2.5's vocabulary is ~152k, so a 1024-token
window's float32 log-probability matrix is ~600 MB. We never materialize the
whole thing — position-chunked reductions keep peak usage bounded on a 16 GB
machine that is also holding two models.

Nothing in this module makes a network call after the first model download.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field

from ..config import settings

log = logging.getLogger(__name__)

# Tokens per scoring window, and how many of them actually get scored. The
# leading `WINDOW - STRIDE` tokens serve as context so that tokens near a
# window boundary aren't judged with an artificially empty history.
WINDOW = 1024
STRIDE = 512
# Positions reduced at once inside a window. Trades peak memory for speed.
CHUNK = 128


def resolve_device() -> str:
    import torch

    if settings.device != "auto":
        return settings.device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(slots=True)
class TokenStats:
    """Accumulated statistics over a whole document.

    All fields are sums, which is what makes windowing valid: every downstream
    statistic is a ratio of sums, so chunking the document and adding the parts
    gives the same answer as scoring it whole.
    """

    n_tokens: int = 0

    # Observed log-probability under the scorer model.
    sum_logp: float = 0.0

    # Fast-DetectGPT: expectation and variance of log p(x̃|x_<i) when x̃ is
    # drawn from the model's own conditional distribution.
    sum_mu: float = 0.0
    sum_var: float = 0.0

    # Binoculars: cross-entropy between scorer and reference predictions.
    sum_xent: float = 0.0
    has_reference: bool = False

    # Per-token detail, for sentence-level burstiness and the heatmap.
    token_logp: list[float] = field(default_factory=list)
    token_offsets: list[tuple[int, int]] = field(default_factory=list)
    sum_log_rank: float = 0.0
    truncated: bool = False

    @property
    def mean_logp(self) -> float:
        return self.sum_logp / self.n_tokens if self.n_tokens else 0.0

    @property
    def perplexity(self) -> float:
        return math.exp(-self.mean_logp) if self.n_tokens else 0.0

    @property
    def mean_log_rank(self) -> float:
        return self.sum_log_rank / self.n_tokens if self.n_tokens else 0.0


class LocalModels:
    """Lazily-loaded scorer/reference model pair."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._scorer = None
        self._reference = None
        self._device: str | None = None
        self._load_error: str | None = None

    # -- loading -----------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device()
        return self._device

    @property
    def dtype(self):
        import torch

        return torch.float32 if self.device == "cpu" else torch.float16

    def _load(self) -> bool:
        """Load both models. Returns False (once, loudly) if unavailable."""
        if self._scorer is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self._load_error = (
                "transformers not installed. Run: uv pip install -e '.[models]'"
            )
            log.warning(self._load_error)
            return False

        cache = str(settings.model_dir)
        try:
            log.info("Loading %s on %s...", settings.scorer_model, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.scorer_model, cache_dir=cache
            )
            self._scorer = AutoModelForCausalLM.from_pretrained(
                settings.scorer_model, cache_dir=cache, dtype=self.dtype
            ).to(self.device).eval()

            log.info("Loading %s...", settings.reference_model)
            self._reference = AutoModelForCausalLM.from_pretrained(
                settings.reference_model, cache_dir=cache, dtype=self.dtype
            ).to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            self._load_error = f"Could not load local models: {exc}"
            log.warning(self._load_error)
            self._scorer = None
            return False

        return True

    @property
    def available(self) -> bool:
        return self._load()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # -- scoring -----------------------------------------------------------

    def _reduce_window(self, stats: TokenStats, ids, offsets, first_scored: int) -> None:
        """Run both models over one window and fold the results into `stats`."""
        import torch

        with torch.no_grad():
            scorer_logits = self._scorer(ids).logits[0].float()
            reference_logits = self._reference(ids).logits[0].float()

        # Position i predicts token i+1, so drop the final position and the
        # first target. `first_scored` skips tokens already counted by the
        # previous window's stride.
        targets = ids[0, 1:]
        n_positions = targets.shape[0]
        start = max(first_scored - 1, 0)
        if start >= n_positions:
            return

        for lo in range(start, n_positions, CHUNK):
            hi = min(lo + CHUNK, n_positions)
            chunk_targets = targets[lo:hi]

            s_logits = scorer_logits[lo:hi]
            s_logp = torch.log_softmax(s_logits, dim=-1)
            s_prob = s_logp.exp()

            observed = s_logp.gather(1, chunk_targets.unsqueeze(1)).squeeze(1)

            # E[log p] and Var[log p] under the model's own distribution —
            # the analytic form of Fast-DetectGPT's curvature, which needs no
            # sampling and is exact.
            mu = (s_prob * s_logp).sum(dim=-1)
            second_moment = (s_prob * s_logp.pow(2)).sum(dim=-1)
            var = (second_moment - mu.pow(2)).clamp(min=0.0)

            # Rank of the observed token, without a full sort.
            observed_logit = s_logits.gather(1, chunk_targets.unsqueeze(1))
            rank = (s_logits > observed_logit).sum(dim=-1).float() + 1.0

            stats.sum_logp += float(observed.sum())
            stats.sum_mu += float(mu.sum())
            stats.sum_var += float(var.sum())
            stats.sum_log_rank += float(rank.log().sum())
            stats.n_tokens += hi - lo
            stats.token_logp.extend(observed.tolist())
            stats.token_offsets.extend(offsets[lo + 1 : hi + 1])

            # Binoculars denominator: how surprising the reference model's
            # predictions are to the scorer model.
            r_logp = torch.log_softmax(reference_logits[lo:hi], dim=-1)
            stats.sum_xent += float(-(s_prob * r_logp).sum())

            del s_logits, s_logp, s_prob, r_logp

        stats.has_reference = True
        del scorer_logits, reference_logits

    def analyze(self, text: str, max_tokens: int = 8192) -> TokenStats | None:
        """Score `text`, returning None when the models are unavailable."""
        if not self._load():
            return None

        import torch

        encoded = self._tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=False
        )
        input_ids = encoded["input_ids"][0]
        offsets = [tuple(o) for o in encoded["offset_mapping"][0].tolist()]

        stats = TokenStats()
        if input_ids.shape[0] > max_tokens:
            input_ids = input_ids[:max_tokens]
            offsets = offsets[:max_tokens]
            stats.truncated = True

        if input_ids.shape[0] < 2:
            return stats

        total = input_ids.shape[0]
        position = 0
        while position < total:
            end = min(position + WINDOW, total)
            window_ids = input_ids[position:end].unsqueeze(0).to(self.device)
            window_offsets = offsets[position:end]
            # Everything before `first_scored` is context, already counted.
            first_scored = 0 if position == 0 else WINDOW - STRIDE
            self._reduce_window(stats, window_ids, window_offsets, first_scored)

            if end == total:
                break
            position += STRIDE

        if self.device == "mps":
            torch.mps.empty_cache()
        return stats


@functools.lru_cache(maxsize=1)
def get_models() -> LocalModels:
    return LocalModels()
