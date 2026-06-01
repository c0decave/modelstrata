"""Canonical tensor naming — Role enum + renderer to GGUF (llama.cpp) name strings.

The rest of the pipeline (analysis tools + dashboard) speaks the GGUF naming
convention `blk.N.<role>.weight`. We model roles structurally as an enum and
render them to those strings here; later the enum lets us drop the strings.
"""
import enum


class Role(enum.Enum):
    TOK_EMBD = "TOK_EMBD"
    OUTPUT = "OUTPUT"
    OUTPUT_NORM = "OUTPUT_NORM"
    ATTN_Q = "ATTN_Q"
    ATTN_K = "ATTN_K"
    ATTN_V = "ATTN_V"
    ATTN_OUT = "ATTN_OUT"
    ATTN_QKV = "ATTN_QKV"
    ATTN_NORM = "ATTN_NORM"
    # gemma2/3 sandwich norms + gemma3/qwen3 QK-norm (llama.cpp gemma stems)
    ATTN_POST_NORM = "ATTN_POST_NORM"
    ATTN_Q_NORM = "ATTN_Q_NORM"
    ATTN_K_NORM = "ATTN_K_NORM"
    FFN_GATE = "FFN_GATE"
    FFN_UP = "FFN_UP"
    FFN_DOWN = "FFN_DOWN"
    FFN_NORM = "FFN_NORM"
    FFN_POST_NORM = "FFN_POST_NORM"
    FFN_GATE_INP = "FFN_GATE_INP"
    FFN_GATE_EXPS = "FFN_GATE_EXPS"
    FFN_UP_EXPS = "FFN_UP_EXPS"
    FFN_DOWN_EXPS = "FFN_DOWN_EXPS"


# Role -> (gguf_stem, layered) using the llama.cpp stems.
_GGUF = {
    Role.TOK_EMBD: ("token_embd", False),
    Role.OUTPUT: ("output", False),
    Role.OUTPUT_NORM: ("output_norm", False),
    Role.ATTN_Q: ("attn_q", True),
    Role.ATTN_K: ("attn_k", True),
    Role.ATTN_V: ("attn_v", True),
    Role.ATTN_OUT: ("attn_output", True),
    Role.ATTN_QKV: ("attn_qkv", True),
    Role.ATTN_NORM: ("attn_norm", True),
    Role.ATTN_POST_NORM: ("attn_post_norm", True),
    Role.ATTN_Q_NORM: ("attn_q_norm", True),
    Role.ATTN_K_NORM: ("attn_k_norm", True),
    Role.FFN_GATE: ("ffn_gate", True),
    Role.FFN_UP: ("ffn_up", True),
    Role.FFN_DOWN: ("ffn_down", True),
    Role.FFN_NORM: ("ffn_norm", True),
    Role.FFN_POST_NORM: ("ffn_post_norm", True),
    Role.FFN_GATE_INP: ("ffn_gate_inp", True),
    Role.FFN_GATE_EXPS: ("ffn_gate_exps", True),
    Role.FFN_UP_EXPS: ("ffn_up_exps", True),
    Role.FFN_DOWN_EXPS: ("ffn_down_exps", True),
}


def canonical(role, layer=None, comp="weight"):
    """Render a Role to its GGUF tensor name string.

    Layered roles require a layer index — a missing one raises ValueError
    rather than silently defaulting.
    """
    stem, layered = _GGUF[role]
    if layered:
        if layer is None:
            raise ValueError(f"{role} requires a layer index")
        return f"blk.{layer}.{stem}.{comp}"
    return f"{stem}.{comp}"
