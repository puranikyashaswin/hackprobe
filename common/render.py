"""Turn-by-turn rendering of a HackTrace trajectory for Qwen3-8B.

Turn t is rendered the way the model saw it when it generated that turn:
apply_chat_template(messages[:2t], add_generation_prompt=True,
enable_thinking=False) followed by that turn's assistant text. Never the whole
conversation at once (the Qwen3 template treats past assistant turns
differently from the current one).

messages = [system, user_1, assistant_1, user_2, assistant_2, ...], so the
assistant text of turn t is messages[2t].
"""

import re
from dataclasses import dataclass

FENCE = re.compile(r"```")


@dataclass
class Rendered:
    text: str               # prompt + assistant text of this turn
    prompt_chars: int       # where the assistant text starts in `text`
    ids: list               # token ids of `text`
    offsets: list           # (start, end) char span of each token
    assistant_start: int    # first token index of the assistant text
    code_mask: list         # per token: inside a fenced code block of the assistant text


def n_turns(messages: list) -> int:
    return sum(m["role"] == "assistant" for m in messages)


def prompt_text(tok, messages: list, turn: int) -> str:
    return tok.apply_chat_template(messages[: 2 * turn], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def code_spans(text: str) -> list:
    """Char spans of fenced code in assistant text. An unclosed fence runs to the end.
    No fence at all means the whole text counts (HackTrace App. B)."""
    marks = [m.start() for m in FENCE.finditer(text)]
    if not marks:
        return [(0, len(text))]
    spans = []
    for i in range(0, len(marks), 2):
        start = text.find("\n", marks[i])
        start = len(text) if start == -1 else start + 1
        end = marks[i + 1] if i + 1 < len(marks) else len(text)
        spans.append((start, end))
    return spans


def render_turn(tok, messages: list, turn: int) -> Rendered:
    prompt = prompt_text(tok, messages, turn)
    answer = messages[2 * turn]["content"]
    text = prompt + answer
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    start = next(i for i, (a, b) in enumerate(offsets) if b > len(prompt))
    spans = [(len(prompt) + a, len(prompt) + b) for a, b in code_spans(answer)]
    mask = [i >= start and any(a < e and b > s for s, e in spans) for i, (a, b) in enumerate(offsets)]
    return Rendered(text, len(prompt), enc["input_ids"], offsets, start, mask)


def assistant_tokens(tok, messages: list) -> int:
    """Assistant tokens over all turns (for the generated_tokens check)."""
    return sum(len(tok(messages[2 * t]["content"], add_special_tokens=False)["input_ids"])
               for t in range(1, n_turns(messages) + 1))
