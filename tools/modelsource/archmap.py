"""HF-Tensornamen → kanonische (Role-name, layer) pro Architektur-Familie.
Gibt Role als STRING zurück (entkoppelt von naming.Role). Unbekannt → None,
damit der Aufrufer es loggt (kein stilles Raten)."""
import re

# Layered patterns: regex with a (?P<l>\d+) layer group -> Role name.
_LLAMALIKE_LAYERED = [
    # q/k/v/o_proj ship a .bias on Qwen2/Qwen2.5 — match BOTH .weight and .bias
    # so biases reach a canonical role (parity with the GGUF backend, which
    # yields blk.N.attn_*.bias). resolve() returns only (role, layer); the
    # weight-vs-bias component is derived downstream from the tensor-name suffix.
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.q_proj\.(?:weight|bias)$"), "ATTN_Q"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.k_proj\.(?:weight|bias)$"), "ATTN_K"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.v_proj\.(?:weight|bias)$"), "ATTN_V"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.o_proj\.(?:weight|bias)$"), "ATTN_OUT"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.gate_proj\.weight$"),    "FFN_GATE"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.up_proj\.weight$"),      "FFN_UP"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.down_proj\.weight$"),    "FFN_DOWN"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.input_layernorm\.weight$"),           "ATTN_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.post_attention_layernorm\.weight$"),  "FFN_NORM"),
]
# Global (no layer) -> Role name.
_LLAMALIKE_GLOBAL = [
    (re.compile(r"^model\.embed_tokens\.weight$"), "TOK_EMBD"),
    (re.compile(r"^lm_head\.weight$"),             "OUTPUT"),
    (re.compile(r"^model\.norm\.weight$"),         "OUTPUT_NORM"),
]
# --- Gemma (gemma / gemma2 / gemma3, TEXT models) -------------------------
# Gemma shares llama's q/k/v/o + gate/up/down + embed/norm naming, BUT its
# norms differ structurally: gemma2/3 use sandwich norms, so
# post_attention_layernorm is a SEPARATE post-attention norm (NOT the pre-FFN
# norm as in llama) and the pre-FFN norm is pre_feedforward_layernorm. gemma3
# additionally has per-layer q_norm/k_norm. Roles + stems mirror llama.cpp's
# gemma conversion (attn_post_norm / ffn_post_norm / attn_q_norm / attn_k_norm)
# so the HF and GGUF backends yield IDENTICAL canonical names (parity).
_GEMMA_LAYERED = [
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.q_proj\.weight$"), "ATTN_Q"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.k_proj\.weight$"), "ATTN_K"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.v_proj\.weight$"), "ATTN_V"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.o_proj\.weight$"), "ATTN_OUT"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.q_norm\.weight$"), "ATTN_Q_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.self_attn\.k_norm\.weight$"), "ATTN_K_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.gate_proj\.weight$"),    "FFN_GATE"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.up_proj\.weight$"),      "FFN_UP"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.mlp\.down_proj\.weight$"),    "FFN_DOWN"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.input_layernorm\.weight$"),            "ATTN_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.post_attention_layernorm\.weight$"),   "ATTN_POST_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.pre_feedforward_layernorm\.weight$"),  "FFN_NORM"),
    (re.compile(r"^model\.layers\.(?P<l>\d+)\.post_feedforward_layernorm\.weight$"), "FFN_POST_NORM"),
]
FAMILIES = {
    "llamalike": (_LLAMALIKE_LAYERED, _LLAMALIKE_GLOBAL),
    # gemma reuses the llama global tensors (embed / lm_head / final norm)
    "gemma": (_GEMMA_LAYERED, _LLAMALIKE_GLOBAL),
}

_LLAMALIKE_TYPES = {"llama", "qwen2", "qwen3", "mistral"}
_GEMMA_TYPES = {"gemma", "gemma2", "gemma3"}

# Gemma TEXT arch only (Gemma*ForCausalLM). The multimodal
# Gemma*ForConditionalGeneration nests the LM under language_model. and adds a
# vision tower we cannot map → it stays inventory-only (honest, no dropped tower).
_GEMMA_ARCH_RE = re.compile(r"^gemma\d*forcausallm$")

# Real family architecture CLASS names (lowercased). Tightened from a loose
# substring check so an incidentally-named arch like "SomeNonLlamaThing" (which
# merely CONTAINS "llama") is NOT misclassified as a mappable family.
_LLAMALIKE_ARCH_RE = re.compile(r"^(llama|qwen2|qwen3|qwen|mistral)[a-z0-9]*forcausallm$")

# Config keys that signal a Mixture-of-Experts model even when the model_type /
# arch name looks like a plain family.
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts")


def _is_moe(config):
    """True iff the config describes a Mixture-of-Experts model.

    Detection is intentionally broader than mapping. family_for() only promotes
    known Qwen/Mixtral-style MoE layouts to the mappable ``moe`` family; generic
    expert-count configs remain inventory-only so we never badge an unknown
    expert layout as exact.
    """
    mt = (config.get("model_type") or "").lower()
    if mt in _MOE_TYPES or "moe" in mt:
        return True
    if any("moe" in (arch or "").lower()
           for arch in config.get("architectures") or []):
        return True
    return any(k in config for k in _MOE_CONFIG_KEYS)


_MOE_TYPES = {"qwen2_moe", "qwen3_moe", "mixtral"}
_MOE_ARCH_RE = re.compile(r"^(qwen2moe|qwen3moe|mixtral)[a-z0-9]*forcausallm$")


def _known_moe_family(config):
    """True only for MoE layouts we explicitly map.

    A generic ``num_experts`` key is not enough to claim coverage. Unknown MoE
    families stay inventory-only rather than producing a forest of
    tensor_unmapped warnings under an "exact" badge.
    """
    mt = (config.get("model_type") or "").lower()
    if mt in _MOE_TYPES:
        return True
    if mt in {"qwen2", "qwen3"} and any(k in config for k in _MOE_CONFIG_KEYS):
        return True
    return any(_MOE_ARCH_RE.match(str(arch).lower())
               for arch in config.get("architectures") or [])


def family_for(config):
    # MoE check takes precedence: only known Qwen/Mixtral MoE layouts are
    # mappable. Unknown MoE layouts remain inventory-only (never over-claim).
    if _is_moe(config):
        return "moe" if _known_moe_family(config) else None
    # Multimodal / encoder-decoder checkpoints (``*ForConditionalGeneration``)
    # nest the LM under ``language_model.*`` and add a vision/audio tower we
    # cannot map. This VETO runs before the model_type branch: a multimodal
    # gemma3 carries both ``model_type: "gemma3"`` AND this arch, and the
    # model_type must NOT short-circuit it to "gemma" (that would badge
    # precision=exact while every tensor is actually unmapped). Inventory-only.
    archs = [str(a).lower() for a in (config.get("architectures") or [])]
    if any(a.endswith("forconditionalgeneration") for a in archs):
        return None
    mt = (config.get("model_type") or "").lower()
    if mt in _LLAMALIKE_TYPES:
        return "llamalike"
    # gemma / gemma2 / gemma3 — incl. HF's "gemma3_text" model_type for the
    # text config of a multimodal gemma3 checkpoint.
    if mt in _GEMMA_TYPES or mt.startswith("gemma"):
        return "gemma"
    for arch in config.get("architectures") or []:
        a = arch.lower()
        if _MOE_ARCH_RE.match(a):
            return "moe"
        if _LLAMALIKE_ARCH_RE.match(a):
            return "llamalike"
        if _GEMMA_ARCH_RE.match(a):
            return "gemma"
    return None


def _resolve_moe(name):
    """Best-effort exact mapping for common HF MoE checkpoints.

    Qwen-MoE stores experts as ``mlp.experts.<id>.{gate,up,down}_proj``.
    Mixtral stores them as ``block_sparse_moe.experts.<id>.w{1,2,3}`` where
    w1=gate, w3=up, w2=down. We keep each expert as its own dashboard role
    (``ffn_gate_exps.e7`` etc.) instead of collapsing/averaging weights.
    """
    # Attention/norm/global tensors are the same as the llama-like family.
    r = resolve("llamalike", name)
    if r is not None:
        return r

    m = re.match(r"^model\.layers\.(?P<l>\d+)\.mlp\.gate\.weight$", name)
    if m:
        return ("FFN_GATE_INP", int(m.group("l")))
    m = re.match(r"^model\.layers\.(?P<l>\d+)\.block_sparse_moe\.gate\.weight$", name)
    if m:
        return ("FFN_GATE_INP", int(m.group("l")))

    shared = [
        (r"^model\.layers\.(?P<l>\d+)\.mlp\.shared_expert_gate\.weight$", "ffn_gate_shared_inp"),
        (r"^model\.layers\.(?P<l>\d+)\.mlp\.shared_expert\.gate_proj\.weight$", "ffn_gate_shared"),
        (r"^model\.layers\.(?P<l>\d+)\.mlp\.shared_expert\.up_proj\.weight$", "ffn_up_shared"),
        (r"^model\.layers\.(?P<l>\d+)\.mlp\.shared_expert\.down_proj\.weight$", "ffn_down_shared"),
    ]
    for rx, role in shared:
        m = re.match(rx, name)
        if m:
            return (role, int(m.group("l")))

    expert = re.match(
        r"^model\.layers\.(?P<l>\d+)\.mlp\.experts\.(?P<e>\d+)\."
        r"(?P<p>gate_proj|up_proj|down_proj)\.weight$", name)
    if expert:
        part = {"gate_proj": "gate", "up_proj": "up",
                "down_proj": "down"}[expert.group("p")]
        return (f"ffn_{part}_exps.e{expert.group('e')}", int(expert.group("l")))

    mixtral = re.match(
        r"^model\.layers\.(?P<l>\d+)\.block_sparse_moe\.experts\.(?P<e>\d+)\."
        r"(?P<p>w1|w2|w3)\.weight$", name)
    if mixtral:
        part = {"w1": "gate", "w3": "up", "w2": "down"}[mixtral.group("p")]
        return (f"ffn_{part}_exps.e{mixtral.group('e')}", int(mixtral.group("l")))

    return None


def resolve(family, name):
    if family == "moe":
        return _resolve_moe(name)
    fam = FAMILIES.get(family)
    if fam is None:
        return None
    layered, glob = fam
    for rx, role in layered:
        m = rx.match(name)
        if m:
            return (role, int(m.group("l")))
    for rx, role in glob:
        if rx.match(name):
            return (role, None)
    return None
