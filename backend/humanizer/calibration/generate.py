"""Generate machine-text samples locally for the calibration corpus.

Runs entirely on-device using the same Qwen2.5 model already downloaded for
detection. No API, no cost, no text leaving the machine.

Three things are varied deliberately:

  register     Academic, professional, casual, creative — so the model cannot
               separate the classes on formality alone.
  temperature  Low-temperature output is far more machine-like than high;
               training on only one makes the detector brittle.
  prompt style Including "evasive" prompts that explicitly ask for varied
               sentence length and no stock phrases. Those are the hard
               negatives — they are what our own humanizer will produce, so
               training on them is what keeps the detector honest about its
               own output.
"""

from __future__ import annotations

import logging
import random

from ..config import settings
from . import corpus

log = logging.getLogger(__name__)

TOPICS: dict[str, list[str]] = {
    "academic": [
        "the reliability of self-reported survey data",
        "urban heat islands and municipal planning",
        "the replication crisis in social psychology",
        "bilingualism and executive function in children",
        "supply chain resilience after 2020",
        "peer review as a quality-control mechanism",
        "microplastics in freshwater ecosystems",
        "the economics of public transit subsidies",
    ],
    "professional": [
        "why our onboarding process needs rework",
        "a postmortem on a failed product launch",
        "choosing between two database vendors",
        "quarterly hiring plan for a small engineering team",
        "the case for reducing meeting load",
        "customer churn in the last two quarters",
        "migrating a legacy billing system",
        "remote work policy tradeoffs",
    ],
    "casual": [
        "what I learned from a year of cooking at home",
        "why I stopped using a smartwatch",
        "moving to a new city in your thirties",
        "the best and worst things about cycling to work",
        "learning an instrument as an adult",
        "how my reading habits changed",
        "a trip that did not go to plan",
        "living with an old car",
    ],
    "creative": [
        "a lighthouse keeper on the last night of the job",
        "two strangers waiting out a delayed train",
        "a locksmith who never locks their own door",
        "the last shop on a closing high street",
        "an argument about a inherited house",
        "someone returning to a childhood swimming pool",
    ],
}

_PLAIN = (
    "Write roughly {words} words about {topic}. "
    "Use full paragraphs. Do not use headings, bullet points, or lists."
)

_EVASIVE = (
    "Write roughly {words} words about {topic}. Use full paragraphs, no headings "
    "or bullets. Write the way a real person writes: vary your sentence length a "
    "lot, including some very short sentences. Avoid stock phrases and avoid "
    "words like 'delve', 'crucial', 'landscape', 'moreover', and 'furthermore'. "
    "Be specific and concrete rather than general."
)


def _build_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = str(settings.model_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(settings.reference_model, cache_dir=cache)
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings.reference_model, cache_dir=cache, dtype=dtype
        )
        .to(device)
        .eval()
    )
    return tokenizer, model, device


def generate(count: int = 40, *, seed: int = 0, evasive_fraction: float = 0.35) -> int:
    """Generate `count` machine samples into the corpus. Returns how many were
    stored (short or duplicate generations are dropped)."""
    import torch

    rng = random.Random(seed)
    tokenizer, model, device = _build_model()
    stored = 0

    for index in range(count):
        register = rng.choice(list(TOPICS))
        topic = rng.choice(TOPICS[register])
        evasive = rng.random() < evasive_fraction
        words = rng.choice([250, 350, 450])
        temperature = rng.choice([0.6, 0.8, 0.9, 1.0])

        template = _EVASIVE if evasive else _PLAIN
        prompt = template.format(words=words, topic=topic)
        if register == "academic":
            prompt += " Write in an academic register."
        elif register == "creative":
            prompt = (
                f"Write a short piece of narrative fiction, roughly {words} words, "
                f"about {topic}. Prose only."
            )

        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=int(words * 2.0),
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(
            output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        sample = corpus.add(
            text,
            "machine",
            register=register,
            topic=topic,
            generator=settings.reference_model,
            notes=f"temp={temperature} evasive={evasive}",
        )
        if sample:
            stored += 1
        log.info(
            "[%d/%d] %s / %s (temp=%.1f, evasive=%s) -> %s",
            index + 1,
            count,
            register,
            topic[:40],
            temperature,
            evasive,
            "stored" if sample else "skipped",
        )

    if device == "mps":
        torch.mps.empty_cache()
    return stored
