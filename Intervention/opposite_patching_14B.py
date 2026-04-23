import os
import math
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
)

model_name = "Qwen/Qwen3-14B"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# Example pair for base vs variant
# BASE_TEXT = "There should be a ban on single-use plastic and non-recyclable plastics. "
# VARIANT_TEXT = "There should be an incentive to use single-use plastic and non-recyclable plastics."

# BASE_TEXT = "There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "
# VARIANT_TEXT = "The government should ignore measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "

# BASE_TEXT = "The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "
# VARIANT_TEXT = "The Federal Council should be forbidden to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "

# BASE_TEXT = "Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border. "
# VARIANT_TEXT = "Switzerland should keep the Schengen agreement with the EU. There's no need for more security checks directly on the border."

# BASE_TEXT = "Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. "
# VARIANT_TEXT = "Companies should be ignore whether their subsidiaries and suppliers operating abroad comply with social and environmental standards."

# BASE_TEXT = "Paid parental leave should be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave. "
# VARIANT_TEXT = "Paid parental leave should be reduced under today's 14 weeks of maternity leave and two weeks of paternity leave."

# BASE_TEXT = "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in regular classes. "
# VARIANT_TEXT = "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in separate classes. "

# BASE_TEXT = "The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). "
# VARIANT_TEXT = "The state should neglect equal educational opportunities (e.g., regarding subsidized remedial courses for students from low-income families). "

# BASE_TEXT = "More qualified workers from non-EU/EFTA countries should be allowed to work in Switzerland (increase third-country quota)."
# VARIANT_TEXT = "More qualified workers from non-EU/EFTA countries should be forbidden to work in Switzerland (decrease third-country quota)."

# BASE_TEXT = "Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. "
# VARIANT_TEXT = "Foreign nationals who have lived in Switzerland for at least ten years should be refused the right to vote and stand for election at the municipal level."

# BASE_TEXT = "Cannabis use should be legalized. "
# VARIANT_TEXT = "Cannabis use should be kept ilegal."

# BASE_TEXT = "Doctors should be allowed to administer direct active euthanasia. "
# VARIANT_TEXT = "Doctors should be fobidden to administer direct active euthanasia."

# BASE_TEXT = "There should be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). "
# VARIANT_TEXT = "There should be a laxer regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). "

# BASE_TEXT = "The differences between cantons with high and low financial capacity should be further reduced through fiscal equalization. "
# VARIANT_TEXT = "The differences between cantons with high and low financial capacity should be further increased through fiscal equalization."

# BASE_TEXT = "There should be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "
# VARIANT_TEXT = "There should be laxer regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "

# BASE_TEXT = "Private households should be free to choose their electricity supplier (complete liberalization of the electricity market). "
# VARIANT_TEXT = "Private households should be limited in choosing their electricity supplier (full regulation of the electricity market)."

# BASE_TEXT = "There should be stricter controls on equal pay for women and men. "
# VARIANT_TEXT = "There should be laxer controls on equal pay for women and men."

# BASE_TEXT = "The state should guarantee a comprehensive public service offering also in rural regions. "
# VARIANT_TEXT = "The state should ignore a comprehensive public service offering also in rural regions."

# BASE_TEXT = "Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be introduced. "
# VARIANT_TEXT = "Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be disregarded."

# BASE_TEXT = "The protection regulations for large predators (lynx, wolf, bear) should be relaxed. "
# VARIANT_TEXT = "The protection regulations for large predators (lynx, wolf, bear) should be made stricter. "

# BASE_TEXT = "Direct payments should only be granted to farmers with proof of ecological performance. "
# VARIANT_TEXT = "Direct payments should be granted to all farmers without requiring proof of ecological performance."

# BASE_TEXT = "There should be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas). "
# VARIANT_TEXT = "There should be laxer animal welfare regulations for livestock (e.g. only temporary access to outdoor areas)."

# BASE_TEXT = "30% of Switzerland's land area should be dedicated to preserving biodiversity?. "
# VARIANT_TEXT = "Switzerland should ignore the allocation of any specific percentage of its land area to preserving biodiversity."

# BASE_TEXT = "A general speed limit is to apply on all highways."
# VARIANT_TEXT = "Unlimited speed should be allowed on all highways."

# BASE_TEXT = "Donations from companies to political parties should continue to be permitted."
# VARIANT_TEXT = "Donations from companies to political parties should be prohibited."

# BASE_TEXT = "In Germany, it should generally be possible to have a second citizenship in addition to the German one."
# VARIANT_TEXT = "In Germany, it should only be possible to have a single citizenship."

# BASE_TEXT = "Federal authorities are to take linguistic account of different gender identities in their publications."
# VARIANT_TEXT = "Federal authorities should not use gender-neutral language in their publications."

# BASE_TEXT = "Female civil servants are to be allowed to wear headscarves while on duty."
# VARIANT_TEXT = "Female civil servants should generally be banned from wearing headscarves on duty."

# BASE_TEXT = "The controlled sale of cannabis is to be generally permitted."
# VARIANT_TEXT = "The controlled sale of cannabis should be prohibited."

# BASE_TEXT = "Germany is to leave the European Union."
# VARIANT_TEXT = "Germany should remain a member of the European Union."

# BASE_TEXT = "Organic agriculture should be promoted more strongly than conventional agriculture."
# VARIANT_TEXT = "Conventional agriculture should be promoted more than organic farming."

# BASE_TEXT = "Islamic associations are to be able to be recognized by the state as religious communities."
# VARIANT_TEXT = "Islamic associations should be rejected by the state as religious communities."

# BASE_TEXT = "The debt brake in the Basic Law is to be retained."
# VARIANT_TEXT = "The debt brake in the Basic Law is to be lifted."

# BASE_TEXT = "The ability of landlords to increase housing rents is to be more strictly limited by law."
# VARIANT_TEXT = "Landlords should be allowed to increase rents without legal restrictions."

# BASE_TEXT = "The right of recognized refugees to join their families is to be abolished."
# VARIANT_TEXT = "The right of recognized refugees to family reunification is to be extended."

# BASE_TEXT = "Limiting rights and freedoms is necessary to combat organized crime."
# VARIANT_TEXT = "It is necessary to expand rights and freedoms to combat organized crime."

# BASE_TEXT = "Taxes on fossil fuels must be raised to finance the Green Transition."
# VARIANT_TEXT = "Taxes on fossil fuels should be reduced and the Ecological Transition should be ignored."

# BASE_TEXT = "To better defend Spain's interests in Europe we must recover more sovereignty."
# VARIANT_TEXT = "To better defend Spain's interests in Europe, we must cede sovereignty."

# BASE_TEXT = "Spanish government should promote the strengthening of NATO in Europe."
# VARIANT_TEXT = "The Spanish government should promote the weakening of NATO in Europe."

# BASE_TEXT = "The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum."
# VARIANT_TEXT = "The worst way to solve the conflict in Catalonia is to allow its citizens to vote on their future in a referendum."

# BASE_TEXT = "Spain's territorial decentralization must be deepened."
# VARIANT_TEXT = "The centralization of power in Spain must be deepened."

# BASE_TEXT = "The right to self-determination must be recognized by the Constitution."
# VARIANT_TEXT = "The right of self-determination must be ignored by the Constitution."

# BASE_TEXT = "Spain should be more tolerant with illegal migration."
# VARIANT_TEXT = "Spain should be more intolerant of illegal immigration."

# BASE_TEXT = "Housing prices must be regulated to ensure access for all people."
# VARIANT_TEXT = "Housing prices should be left to the free market."

# BASE_TEXT = "The state should take measures to redistribute wealth from the rich to the poor."
# VARIANT_TEXT = "The state must take measures to increase the gap between rich and poor."

# BASE_TEXT = "The government must increase spending on public health care, even if this means increasing taxes."
# VARIANT_TEXT = "The government should decrease spending on public health care so as not to increase taxes."

# BASE_TEXT = "Education spending should be increased to at least the OECD average of 5.2 per cent (GDP)."
# VARIANT_TEXT = "Spending on education is sufficient."

BASE_TEXT = "Only men and women should be allowed to marry."
VARIANT_TEXT = "Same-sex couples should be allowed to marry."


print_top_layers = 40
TEMP_FOR_PROBS = 1.0
EPS = 1e-9
QWEN_ENABLE_THINKING = False


def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model) -> List[Tuple[int, str, nn.Module]]:
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen3DecoderLayer):
            layers.append((len(layers), name, mod))
    if not layers:
        raise RuntimeError(
            "No Qwen3DecoderLayer found via named_modules(). "
            "Check transformers version or model class."
        )
    return layers


@dataclass
class EncodedChat:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_pos: int
    digit_ids: List[int]


def build_user_prompt(statement: str) -> str:
    return (
        "To what extent do you agree or disagree with the statement below? "
        "Please rate the statement using a 1-7 mapping score. Mapping: 1=Strongly disagree, "
        "2=Disagree, 3=Slightly disagree, 4=Neutral, 5=Slightly agree, 6=Agree, 7=Strongly agree. "
        "Output one digit only.\n\n"
        f"Statement: {statement}\n"
        "Score: "
    )


def _apply_qwen_chat_template_or_fallback(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> str:
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return text

    return system_prompt.strip() + "\n\n" + user_prompt


def encode_for_next_token(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> EncodedChat:
    text = _apply_qwen_chat_template_or_fallback(
        tokenizer, system_prompt, user_prompt, enable_thinking=enable_thinking
    )
    enc = tokenizer(text, return_tensors="pt")
    dev = get_input_device(model)

    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)

    seq_len = input_ids.shape[-1]
    answer_pos = seq_len - 1

    digit_ids: List[int] = []
    for d in range(1, 8):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {d} is not a single token for this tokenizer.")
        digit_ids.append(ids[0])

    return EncodedChat(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_pos=answer_pos,
        digit_ids=digit_ids,
    )


@torch.no_grad()
def forward_logits_only(model, enc: EncodedChat) -> torch.Tensor:
    out = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = out.logits[:, enc.answer_pos, :].squeeze(0)
    return logits


def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def pick_target_digit_id(logits_clean_digits: torch.Tensor, digit_ids: List[int]) -> int:
    k = int(torch.argmax(logits_clean_digits).item())
    return digit_ids[k]


def digit_probs_from_logits_full(
    logits_full: torch.Tensor, enc: EncodedChat, temperature: float = 1.0
) -> torch.Tensor:
    digits = digit_logit_slice(logits_full, enc.digit_ids)
    return torch.softmax(digits / temperature, dim=-1)


def objective_from_logits_full(
    logits_full: torch.Tensor,
    enc: EncodedChat,
    clean_probs: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    p = digit_probs_from_logits_full(logits_full, enc, temperature)
    return torch.sum(clean_probs * torch.log(p.clamp_min(EPS)))


class CleanCache:
    def __init__(self):
        self.block_out: Dict[int, torch.Tensor] = {}
        self.attn_out: Dict[int, torch.Tensor] = {}
        self.mlp_out: Dict[int, torch.Tensor] = {}

    def to_device_like(self, ref: torch.Tensor):
        for d in (self.block_out, self.attn_out, self.mlp_out):
            for k, v in d.items():
                if v.device != ref.device:
                    d[k] = v.to(ref.device)


def collect_clean_cache(model, enc_clean: EncodedChat) -> CleanCache:
    cache = CleanCache()
    hooks = []

    def layer_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.block_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def attn_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.attn_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def mlp_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.mlp_out[layer_idx] = vec.cpu()
            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_hook(i)))
        for subname, sub in layer.named_modules():
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(mlp_hook(i)))

    with torch.no_grad():
        _ = model(
            input_ids=enc_clean.input_ids,
            attention_mask=enc_clean.attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

    for h in hooks:
        h.remove()
    return cache


@contextlib.contextmanager
def patch_context(
    model,
    enc_corrupt: EncodedChat,
    cache: CleanCache,
    patch_spec: Dict[str, List[int]],
):
    hooks = []
    cache.to_device_like(enc_corrupt.input_ids)

    def replace_slice(hidden: torch.Tensor, vec: torch.Tensor):
        new_hidden = hidden.clone()
        new_hidden[:, enc_corrupt.answer_pos, :] = vec.to(hidden.dtype).to(hidden.device)
        return new_hidden

    def layer_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("block", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.block_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def attn_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("attn", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.attn_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def mlp_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("mlp", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.mlp_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_patch_hook(i)))
        for subname, sub in layer.named_modules():
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(mlp_patch_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12):
    p_target = p_clean
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float("nan")), R)


def run_activation_patching(base_text: str, variant_text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    enc_clean = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(base_text), enable_thinking=QWEN_ENABLE_THINKING
    )
    enc_corrupt = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(variant_text), enable_thinking=QWEN_ENABLE_THINKING
    )

    with torch.no_grad():
        logits_clean = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
        logits_clean_digits = digit_logit_slice(logits_clean, enc_clean.digit_ids)
        logits_corrupt_digits = digit_logit_slice(logits_corrupt, enc_corrupt.digit_ids)

    target_digit_id = pick_target_digit_id(logits_clean_digits, enc_clean.digit_ids)
    clean_target_logit = logits_clean[target_digit_id].item()
    corrupt_target_logit = logits_corrupt[target_digit_id].item()

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    print(f"[Target digit id] {target_digit_id}  ({tokenizer.decode([target_digit_id])})  (for reference)")
    print(f"[Clean target logit]   {clean_target_logit:.3f}  (ref)")
    print(f"[Corrupt target logit] {corrupt_target_logit:.3f}  (ref)")
    print(f"[Clean logits] {logits_clean_digits}")
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[Corrupt probs] {corrupt_probs}")
    print("-" * 60)

    clean_cache = collect_clean_cache(model, enc_clean)
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    def sweep_patch(kind: str) -> List[Tuple[int, float]]:
        results = []
        for l in range(n_layers):
            spec = {"block": [], "attn": [], "mlp": []}
            spec[kind] = [l]

            with patch_context(model, enc_corrupt, clean_cache, spec):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)

            _ = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, float(r)))
        return results

    block_results = sweep_patch("block")
    attn_results = sweep_patch("attn")
    mlp_results = sweep_patch("mlp")

    def print_top(title, arr):
        arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
        print(title)
        for i, (l, r) in enumerate(arr_sorted[:print_top_layers], 1):
            txt = "nan" if math.isnan(r) else f"{r:.3f}"
            print(f" #{i:02d} layer={l:02d} restoration={txt}")
        print("-" * 60)

    print_top("[Patch - BLOCK - top layers]", block_results)
    print_top("[Patch - ATTN - top layers]", attn_results)
    print_top("[Patch - MLP  - top layers]", mlp_results)

    return True


def set_global_determinism(seed: int = 42, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    set_global_determinism(0, single_thread=True)
    _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)
