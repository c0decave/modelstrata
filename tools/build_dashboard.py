#!/usr/bin/env python3
"""
build_dashboard.py — verdichtet die gguf_inspect JSON-Reports zu einem
eigenstaendigen, offline-faehigen HTML-Dashboard (keine externen Daten,
keine externen Assets, keine JS-Build-Tools). Systemfonts mit Monospace-Fallback.

  python3 build_dashboard.py reports/disk_models.json reports/ollama_models.json -o reports/dashboard.html
"""
import argparse
import json
import os
import re
import sys

# ggml/safetensors tensor types that carry full (un-quantized) precision. Mirror
# of modelsource.gguf_backend._EXACT_TYPES so build_dashboard stays import-free
# (it must run with pure stdlib — gguf/numpy may be absent on the dashboard host).
_EXACT_TYPES = frozenset({"F32", "F16", "BF16"})


def _gguf_source_block(info):
    """Synthesize a ``source`` block for a GGUF fleet entry.

    precision is ``exact`` iff every tensor type in ``quant_breakdown`` is
    un-quantized (F32/F16/BF16) — same rule as
    ``modelsource.gguf_backend.GGUFSource._is_exact`` — else ``approx``. GGUF
    tensor names are already canonical so ``mapped`` is always True. Warnings
    are empty here (the deep tools' run_log carries per-tensor degradation).
    """
    breakdown = info.get("quant_breakdown") or {}
    exact = all(t in _EXACT_TYPES for t in breakdown)
    arch = (info.get("metadata") or {}).get("general.architecture")
    return {
        "format": "gguf",
        "precision": "exact" if exact else "approx",
        "arch": arch,
        "mapped": True,
        "warnings": [],
    }


# capture optional tower prefix (e.g. "v." vision, "a." audio) so multi-tower
# models don't collide text/vision/audio tensors onto the same heatmap cell
BLK_RE = re.compile(r"^(?P<pre>(?:[a-z]+\.)*)blk\.(?P<layer>\d+)\.(?P<role>.+?)\.(?P<suf>weight|bias)$")


def derive(info):
    m = info["metadata"]
    arch = m.get("general.architecture", "?")

    def scalarize(v):
        # arrays (per-layer head counts etc.) -> representative scalar.
        # Use the max of the numeric elements (bool excluded — it's an int
        # subclass); fall back to the first element for non-numeric arrays.
        if isinstance(v, dict) and v.get("_array"):
            sample = v.get("sample") or []
            nums = [s for s in sample
                    if isinstance(s, (int, float)) and not isinstance(s, bool)]
            if nums:
                return max(nums)
            return sample[0] if sample else None
        return v

    def a(suffix, default=None):
        # arch-prefixed key, e.g. qwen2.block_count
        return scalarize(m.get(f"{arch}.{suffix}", default))

    def num(x):
        # numeric scalar only — reject bool (int subclass) so degenerate
        # boolean array fallbacks never leak into head/layer/ctx math
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    n_layers = num(a("block_count"))
    d_model = num(a("embedding_length"))
    # Fallback for HF synth entries (analyze.synth_hf_entry): they have NO
    # gguf-style <arch>.block_count / embedding_length / attention.* keys, so
    # the above is None. Fall back to honest top-level config.json values
    # (layers, hidden, heads, ctx, intermediate size) — no faked gguf data. GGUF
    # entries always have the arch keys, so they never reach these fallbacks.
    if n_layers is None:
        n_layers = num(info.get("n_layers"))
    if d_model is None:
        d_model = num(info.get("hidden")) or num(info.get("d_model"))
    n_head = num(a("attention.head_count"))
    n_head_kv = num(a("attention.head_count_kv"))
    ffn = num(a("feed_forward_length"))
    ctx = num(a("context_length"))
    if n_head is None:
        n_head = num(info.get("n_heads")) or num(info.get("n_head"))
    if n_head_kv is None:
        n_head_kv = num(info.get("n_kv_heads")) or num(info.get("n_head_kv"))
    if ffn is None:
        ffn = num(info.get("ffn")) or num(info.get("intermediate"))
    if ctx is None:
        ctx = num(info.get("ctx")) or num(info.get("context_length"))

    # tensor grid: role x layer -> quant type (of the .weight tensor)
    cells = {}        # role -> {layer:int -> qtype}
    role_params = {}  # role -> total params (across layers, weight+bias)
    globals_t = {}    # non-block tensor name -> qtype
    roles_order = []
    max_layer = -1
    bias_count = 0    # 1-D bias tensors folded out of the heatmap
    vision = audio = False
    for t in info.get("tensors", []):
        name = t["name"]
        if name.startswith(("v.", "mm.", "vision")):
            vision = True
        if ".audio" in name or name.startswith("a."):
            audio = True
        mobj = BLK_RE.match(name)
        if mobj:
            layer = int(mobj.group("layer"))
            # prefix tower (v./a.) into the role so columns stay distinct
            role = mobj.group("pre") + mobj.group("role")
            suf = mobj.group("suf")
            max_layer = max(max_layer, layer)
            lyr = cells.setdefault(role, {})
            # .weight defines the cell quant; a .bias never overwrites a weight
            if suf == "weight" or layer not in lyr:
                lyr[layer] = t["type"]
            if suf == "bias":
                bias_count += 1
            role_params[role] = role_params.get(role, 0) + t.get("params", 0)
            if role not in roles_order:
                roles_order.append(role)
        else:
            globals_t[name] = t["type"]
    if "vision" in str(m.get("general.architecture", "")):
        vision = True
    if a("vision.block_count"):
        vision = True
    if a("audio.block_count"):
        audio = True

    # order roles canonically: norms, attn, ffn, then rest
    def role_rank(r):
        order = ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm",
                 "attn_k_norm", "attn_output", "post_attention_norm",
                 "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
                 "post_ffw_norm", "ffn_gate_inp"]
        return (order.index(r), r) if r in order else (len(order), r)
    roles_sorted = sorted(roles_order, key=role_rank)

    # ---- derived metrics (header-only, no weights) ----
    head_dim = a("attention.key_length") or (
        (d_model // n_head) if (d_model and n_head) else None)
    key_len = a("attention.key_length")
    val_len = a("attention.value_length")
    # KV-cache @ full ctx, fp16 (obere Schranke bei voller globaler Attention).
    # MLA (kv_lora_rank gesetzt, z.B. glm/DeepSeek-V) cached einen komprimierten
    # Latent statt K/V pro Head -> separat & viel kleiner. Sliding-Window (SWA)
    # reduziert es real ebenfalls; wir zeigen die obere Schranke.
    kv_lora_rank = a("attention.kv_lora_rank")
    rope_dim = a("rope.dimension_count")
    kv_cache = None
    kv_cache_kind = None
    if kv_lora_rank and n_layers and ctx:
        # DeepSeek/GLM MLA: latent dim = kv_lora_rank + rope (decoupled key)
        latent = kv_lora_rank + (rope_dim or 0)
        kv_cache = n_layers * latent * ctx * 2
        kv_cache_kind = "MLA (komprimierter Latent)"
    elif all(isinstance(x, (int, float)) for x in (n_layers, n_head_kv, head_dim, ctx)) \
            and None not in (n_layers, n_head_kv, head_dim, ctx):
        # use separate key/value dims when present (some models differ), else head_dim
        kdim = key_len or head_dim
        vdim = val_len or kdim
        kv_cache = n_layers * n_head_kv * (kdim + vdim) * ctx * 2
        kv_cache_kind = "GQA/MHA, obere Schranke"

    # context extension (YaRN / linear) detection
    rope_scaling_type = a("rope.scaling.type")
    rope_scaling_factor = a("rope.scaling.factor")
    rope_orig_ctx = a("rope.scaling.original_context_length")
    ctx_extended = bool(rope_scaling_type or rope_scaling_factor or rope_orig_ctx)

    # MoE
    moe = {
        "expert_count": a("expert_count"),
        "expert_used_count": a("expert_used_count"),
        "expert_shared_count": a("expert_shared_count"),
        "expert_ffn": a("expert_feed_forward_length"),
    } if a("expert_count") else None

    # tied embeddings: no separate output.weight => tied with token_embd.
    # This GGUF naming-convention recompute is always False for HF tensor names,
    # so an HF synth entry forwards an explicit top-level ``tied`` flag (from the
    # backend's tie_word_embeddings + lm_head check). PREFER that explicit value
    # when present; otherwise keep the gguf recompute (gguf cards unchanged).
    tied = ("output.weight" not in globals_t) and any(
        n.startswith("token_embd") for n in globals_t)
    if info.get("tied") is not None:
        tied = bool(info["tied"])

    # special tokens
    def tok(k):
        return m.get(f"tokenizer.ggml.{k}")
    special = {
        "bos": tok("bos_token_id"), "eos": tok("eos_token_id"),
        "pad": tok("padding_token_id"), "unk": tok("unknown_token_id"),
        "add_bos": tok("add_bos_token"), "add_eos": tok("add_eos_token"),
        "pre": tok("pre"),
    }
    merges = m.get("tokenizer.ggml.merges")
    merges_n = merges["len"] if isinstance(merges, dict) and merges.get("_array") else None

    # file_type label vs dominant quant (consistency)
    ftlabel = info.get("file_type_label")
    dom_quant = max((q for q in info["quant_breakdown"] if q not in ("F32", "F16", "BF16")),
                    key=lambda q: info["quant_breakdown"][q], default=None)

    label = info.get("label") or info["path"].split("/")[-1]
    raw_name = (m.get("general.name") or "").strip(" .")
    if not raw_name or raw_name in ("?",):
        bn = (m.get("general.basename") or "").strip()
        sz = (m.get("general.size_label") or "").strip()
        raw_name = (f"{bn} {sz}".strip() or
                    label.replace("ollama:", "").replace(".gguf", ""))

    # source block: a synthesized HF/inventory entry already carries one (it is
    # passed straight through); a raw gguf_inspect entry gets one synthesized
    # from its quant_breakdown. Missing/unrecognized -> None (card renders no
    # badge, gracefully).
    source = info.get("source") or _gguf_source_block(info)

    return {
        "label": label,
        "path": info["path"],
        "arch": arch,
        "source": source,
        "name": raw_name,
        "basename": m.get("general.basename", ""),
        "finetune": m.get("general.finetune", ""),
        "size_label": m.get("general.size_label", ""),
        "license": m.get("general.license", ""),
        "params": info["total_params"],
        "file_size": info["file_size"],
        "n_tensors": info["n_tensors"],
        "n_layers": n_layers,
        "d_model": d_model,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "gqa": (round(n_head / n_head_kv, 1) if (n_head and n_head_kv) else None),
        "ffn": ffn,
        "ffn_ratio": (round(ffn / d_model, 2) if (ffn and d_model) else None),
        "ctx": ctx,
        "vocab": _vocab(m),
        "quant": info["quant_breakdown"],
        "chat_template_len": len(m["tokenizer.chat_template"]) if isinstance(m.get("tokenizer.chat_template"), str) else 0,
        "tokenizer_model": m.get("tokenizer.ggml.model", ""),
        "vision": vision,
        "audio": audio,
        # --- derived (header-only) ---
        "head_dim": head_dim,
        "key_len": key_len,
        "val_len": val_len,
        "kv_cache": kv_cache,
        "kv_cache_kind": kv_cache_kind,
        "rope_freq_base": a("rope.freq_base"),
        "ctx_extended": ctx_extended,
        "rope_scaling_type": rope_scaling_type,
        "rope_scaling_factor": rope_scaling_factor,
        "rope_orig_ctx": rope_orig_ctx,
        "sliding_window": a("attention.sliding_window"),
        "moe": moe,
        "tied_embeddings": tied,
        "rms_eps": a("attention.layer_norm_rms_epsilon"),
        "special": special,
        "merges_n": merges_n,
        "quant_version": m.get("general.quantization_version"),
        "file_type": m.get("general.file_type"),
        "file_type_label": ftlabel,
        "dom_quant": dom_quant,
        "bits_per_weight": info.get("bits_per_weight"),
        "data_bytes": info.get("data_bytes"),
        "data_start": info.get("data_start"),
        "header_end": info.get("header_end"),
        "alignment": info.get("alignment"),
        "gguf_version": info.get("gguf_version"),
        "header_hex": info.get("header_hex"),
        "header_ascii": info.get("header_ascii"),
        "bias_count": bias_count,
        # full metadata (arrays already summarized by the inspector)
        "meta": m,
        "grid": {
            "roles": roles_sorted,
            "layers": max_layer + 1 if max_layer >= 0 else 0,
            "cells": cells,
            "globals": globals_t,
            "role_params": role_params,
        },
    }


def _vocab(m):
    t = m.get("tokenizer.ggml.tokens")
    if isinstance(t, dict) and t.get("_array"):
        return t["len"]
    v = m.get(f"{m.get('general.architecture','')}.vocab_size")
    return v


def fingerprint(d):
    return (d["arch"], d["name"], d["n_layers"], d["d_model"], d["params"],
            tuple(sorted(d["quant"].items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("-o", "--out", default="reports/dashboard.html")
    ap.add_argument("--forensics", help="tokenizer_forensics.json (Ebene 3)")
    ap.add_argument("--weight-stats", dest="weight_stats",
                    help="weight_stats.json (Ebene 4)")
    ap.add_argument("--spectral", help="spectral.json (Ebene 5)")
    ap.add_argument("--embedding", help="embedding.json (Ebene 6)")
    ap.add_argument("--diff", help="diff.json (Ebene 7)")
    ap.add_argument("--compare", help="static_compare.json (cross-model static comparisons)")
    ap.add_argument("--run-log", dest="run_log",
                    help="run_log.json (Task 15; default: reports/run_log.json if present)")
    args = ap.parse_args()

    raw = []
    for r in args.reports:
        with open(r) as fh:
            raw.extend(json.load(fh))

    def load_opt(path):
        if not path:
            return None
        with open(path) as fh:
            return json.load(fh)

    forensics = load_opt(args.forensics) or {}
    wstats = load_opt(args.weight_stats) or []
    spectral = load_opt(args.spectral) or []
    embedding = load_opt(args.embedding) or []
    diff = load_opt(args.diff) or []
    compare = load_opt(args.compare) or {}

    # run_log: explicit flag wins; else auto-load reports/run_log.json next to the
    # output if it exists. Missing/empty -> [] -> Log tab renders an empty state.
    run_log_path = args.run_log
    if not run_log_path:
        cand = os.path.join(os.path.dirname(args.out) or ".", "run_log.json")
        if os.path.exists(cand):
            run_log_path = cand
    run_log = []
    if run_log_path and os.path.exists(run_log_path):
        run_log = load_opt(run_log_path) or []

    derived = [derive(x) for x in raw]
    # dedup by fingerprint, prefer entry with shortest/cleanest label (disk path
    # vs ollama:name) — keep both labels merged
    seen = {}
    for d in derived:
        fp = fingerprint(d)
        if fp in seen:
            seen[fp]["aliases"].append(d["label"])
        else:
            d["aliases"] = []
            seen[fp] = d
    models = sorted(seen.values(), key=lambda x: -(x["params"] or 0))

    html = render(models, forensics, wstats, spectral, embedding, diff, run_log, compare)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"[dashboard: {args.out}  {len(models)} unique models "
          f"(from {len(derived)} entries)]")


def _js_embed(obj):
    """Serialize for embedding inside a <script> tag. json.dumps does NOT escape
    '<', so a metadata value containing '</script>' (or U+2028/2029) would break
    out of the data island. Escape to inert unicode escapes (valid in JS strings)."""
    s = json.dumps(obj, default=str)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e")
             .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render(models, forensics=None, wstats=None, spectral=None, embedding=None,
           diff=None, run_log=None, compare=None):
    return (HTML_TEMPLATE
            .replace("__DATA__", _js_embed(models))
            .replace("__FORENSICS__", _js_embed(forensics or {}))
            .replace("__WSTATS__", _js_embed(wstats or []))
            .replace("__SPECTRAL__", _js_embed(spectral or []))
            .replace("__EMBED__", _js_embed(embedding or []))
            .replace("__DIFF__", _js_embed(diff or []))
            .replace("__RUNLOG__", _js_embed(run_log or []))
            .replace("__COMPARE__", _js_embed(compare or {})))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modelstrata // static teardown</title>
<style>
:root{
  --bg:#0a0e0d; --panel:#0f1513; --panel2:#121a17; --line:#1d2925;
  --ink:#c8d6cf; --dim:#6f8279; --faint:#46554f;
  --amber:#ffb547; --phos:#4ade80; --cyan:#38e1d4; --red:#ff5d5d; --violet:#b39dff;
  --grid:rgba(74,222,128,.05);
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace;
  --sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.5}
body{
  background-image:
    linear-gradient(var(--grid) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:32px 32px;
  background-attachment:fixed;
}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,0) 0 2px,rgba(0,0,0,.18) 2px 3px);
  mix-blend-mode:multiply;opacity:.5}
a{color:var(--cyan);text-decoration:none}
.wrap{max-width:1320px;margin:0 auto;padding:28px 24px 80px}

header.top{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:24px;flex-wrap:wrap;gap:16px}
.title{font-weight:700;font-size:30px;letter-spacing:-.5px;color:var(--ink);text-transform:uppercase}
.title .x{color:var(--amber)}
.subtitle{color:var(--dim);font-size:12px;margin-top:6px;max-width:560px}
.stamp{font-size:11px;color:var(--faint);text-align:right}
.stamp b{color:var(--phos)}

.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:28px}
.kpi{background:var(--panel);border:1px solid var(--line);padding:14px 16px;position:relative;overflow:hidden}
.kpi::after{content:"";position:absolute;top:0;left:0;width:3px;height:100%;background:var(--phos);opacity:.6}
.kpi .v{font-size:26px;font-weight:700;color:var(--ink)}
.kpi .l{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-top:2px}

.section-h{display:flex;align-items:center;gap:12px;margin:34px 0 14px}
.section-h .n{color:var(--amber);font-weight:700}
.section-h .t{font-weight:600;text-transform:uppercase;letter-spacing:1.5px;font-size:13px}
.section-h .line{flex:1;height:1px;background:var(--line)}

/* tabs */
nav.tabs{display:flex;gap:2px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:24px}
nav.tabs .tab{padding:9px 16px;cursor:pointer;font-size:11.5px;text-transform:uppercase;letter-spacing:1px;
  color:var(--dim);border:1px solid transparent;border-bottom:none;background:transparent;
  position:relative;top:1px;user-select:none;white-space:nowrap}
nav.tabs .tab:hover{color:var(--ink)}
nav.tabs .tab.on{color:var(--bg);background:var(--phos);font-weight:600}
nav.tabs .tab.on::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;background:var(--phos)}
.tabpane{display:none}
.tabpane.on{display:block}

/* fleet */
.controls{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.chip{background:var(--panel);border:1px solid var(--line);color:var(--dim);
  padding:5px 11px;cursor:pointer;font-size:11px;text-transform:uppercase;letter-spacing:.5px;user-select:none}
.chip.on{color:var(--bg);background:var(--phos);border-color:var(--phos);font-weight:600}
.fleet{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);padding:0;cursor:pointer;
  transition:border-color .15s,transform .15s;position:relative}
.card:hover{border-color:var(--phos);transform:translateY(-2px)}
.card .hd{display:flex;justify-content:space-between;align-items:flex-start;padding:13px 14px 10px;border-bottom:1px solid var(--line)}
.card .nm{font-weight:600;font-size:13px;color:var(--ink);line-height:1.3}
.card .pa{font-size:10px;color:var(--dim);margin-top:3px}
.badge{font-size:9.5px;padding:3px 7px;border:1px solid;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;font-weight:600}
/* format · precision source badge (colour-hinted by precision) */
.srcb{font-size:9px;padding:3px 7px;border:1px solid;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;font-weight:600;display:inline-block}
.srcb.exact{color:var(--phos);border-color:var(--phos)}
.srcb.approx{color:var(--amber);border-color:var(--amber)}
.srcb.inv{color:var(--dim);border-color:var(--faint)}
.card .badges{display:flex;flex-direction:column;align-items:flex-end;gap:5px}
.card .body{padding:11px 14px 14px}
.specrow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:11px}
.spec .sv{font-size:16px;font-weight:600;color:var(--ink)}
.spec .sl{font-size:9px;color:var(--faint);text-transform:uppercase;letter-spacing:.5px}
.qbar{display:flex;height:10px;width:100%;border:1px solid var(--line);overflow:hidden}
.qbar i{height:100%}
.qleg{display:flex;flex-wrap:wrap;gap:6px 10px;margin-top:8px;font-size:9.5px;color:var(--dim)}
.qleg span{display:inline-flex;align-items:center;gap:4px}
.qleg b{width:9px;height:9px;display:inline-block}
.tags{display:flex;gap:5px;margin-top:9px;flex-wrap:wrap}
.tag{font-size:9px;padding:2px 6px;background:var(--panel2);border:1px solid var(--line);color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
.tag.hot{color:var(--red);border-color:var(--red)}
.tag.cy{color:var(--cyan);border-color:var(--cyan)}

/* plots */
.plots{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.plot{background:var(--panel);border:1px solid var(--line);padding:16px}
.plot h4{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--ink)}
.plot .cap{font-size:10px;color:var(--faint);margin-bottom:10px}
svg{width:100%;display:block;overflow:visible}
.axis{stroke:var(--line)}
.axislbl{fill:var(--faint);font-size:9px;font-family:var(--mono)}
.gridln{stroke:var(--line);stroke-dasharray:2 3;opacity:.5}

/* modal / internals */
.modal{position:fixed;inset:0;background:rgba(4,7,6,.86);backdrop-filter:blur(3px);
  display:none;z-index:1000;padding:30px;overflow:auto}
.modal.on{display:block}
.sheet{max-width:1180px;margin:0 auto;background:var(--panel);border:1px solid var(--phos);
  box-shadow:0 0 60px rgba(74,222,128,.12)}
.sheet .shd{display:flex;justify-content:space-between;align-items:center;
  padding:16px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}
.sheet .shd .nm{font-size:18px;font-weight:700}
.sheet .shd .sub{font-size:11px;color:var(--dim);margin-top:3px}
.close{cursor:pointer;color:var(--dim);border:1px solid var(--line);padding:5px 12px;font-size:11px}
.close:hover{color:var(--red);border-color:var(--red)}
.sheet .sbody{padding:20px}
.hm-wrap{overflow-x:auto;border:1px solid var(--line);background:var(--bg);padding:12px}
.hm{border-collapse:collapse;font-size:9px}
.hm th{color:var(--dim);font-weight:500;padding:2px 5px;text-align:right;white-space:nowrap;position:sticky;left:0;background:var(--bg)}
.hm .colh{writing-mode:vertical-rl;transform:rotate(180deg);text-align:left;height:64px;padding:4px 2px;color:var(--faint)}
.hm td{width:13px;height:13px;padding:0;border:1px solid var(--bg)}
.hm td:hover{outline:1px solid var(--ink);outline-offset:-1px}
.gl{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.gl .g{display:flex;align-items:center;gap:6px;font-size:10px;background:var(--bg);border:1px solid var(--line);padding:4px 8px;color:var(--dim)}
.gl .g i{width:10px;height:10px;display:inline-block}
.tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--phos);
  padding:5px 9px;font-size:10px;color:var(--ink);z-index:1100;display:none;white-space:nowrap}
.tip b{color:var(--phos)}
.tip .help{color:var(--dim);font-weight:400}
/* help "?" affordance */
.hq{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;
  border:1px solid var(--faint);border-radius:50%;color:var(--dim);font-size:9px;cursor:help;
  margin-left:5px;vertical-align:middle;user-select:none}
.hq:hover{color:var(--phos);border-color:var(--phos)}
.sl .hq{width:12px;height:12px;font-size:8px}
/* modal sections */
.msec{border-top:1px solid var(--line);padding:18px 0 4px;margin-top:6px}
.msec:first-of-type{border-top:none}
.mh{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--ink);
  display:flex;align-items:center;margin-bottom:4px}
.mcap{font-size:10px;color:var(--faint);margin-bottom:12px}
/* key-value grid */
.kv2{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px 18px}
.kv{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dotted var(--line);padding:4px 0;font-size:11px}
.kv .k{color:var(--dim)}.kv .k .hq{margin-left:3px}
.kv .v{color:var(--ink);font-weight:500;text-align:right;word-break:break-word}
.kv .v.warn{color:var(--amber)}.kv .v.good{color:var(--phos)}.kv .v.hot{color:var(--red)}
/* hex dump */
.hex{font-size:11px;line-height:1.55;background:var(--bg);border:1px solid var(--line);padding:12px;overflow-x:auto;white-space:pre}
.hex .off{color:var(--faint)}.hex .hxb{color:var(--ink)}.hex .asc{color:var(--cyan)}
.hex .ann{color:var(--amber)}
.hex .f-magic{color:var(--amber)}.hex .f-ver{color:var(--cyan)}
.hex .f-nt{color:var(--phos)}.hex .f-nkv{color:var(--violet)}
/* complete metadata */
.metagrp{margin-bottom:14px}
.metagrp .gh{color:var(--amber);font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:10px 0 6px;border-bottom:1px solid var(--line);padding-bottom:3px}
.mrow{display:grid;grid-template-columns:minmax(220px,360px) 1fr;gap:14px;font-size:10.5px;padding:2px 0;border-bottom:1px dotted #131c18}
.mrow .mk{color:var(--dim)}
.mrow .mv{color:var(--ink);word-break:break-word}
.mrow .arr{color:var(--violet)}
.mrow details summary{cursor:pointer;color:var(--cyan)}
.mrow details pre{white-space:pre-wrap;color:var(--ink);background:var(--bg);border:1px solid var(--line);padding:8px;margin-top:6px;max-height:240px;overflow:auto}
/* glossary */
.gloss-tools{display:flex;gap:12px;align-items:center;margin-bottom:18px;flex-wrap:wrap}
.gloss-search{flex:1;min-width:220px;max-width:420px;background:var(--bg);border:1px solid var(--line);
  color:var(--ink);font-family:var(--mono);font-size:12px;padding:8px 12px}
.gloss-search:focus{outline:none;border-color:var(--phos)}
.gloss-search::placeholder{color:var(--faint)}
.gloss-count{font-size:10px;color:var(--faint);white-space:nowrap}
.gcat{margin-bottom:26px}
.gcat-h{display:flex;align-items:center;gap:10px;margin:0 0 12px}
.gcat-h .gc-n{color:var(--amber);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1.5px}
.gcat-h .gc-c{color:var(--faint);font-size:10px}
.gcat-h .line{flex:1;height:1px;background:var(--line)}
.gloss{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}
.gcard{background:var(--panel);border:1px solid var(--line);padding:12px 14px}
.gcard .gt{color:var(--phos);font-weight:600;font-size:12px;margin-bottom:3px}
.gcard .gd{color:var(--dim);font-size:11px;line-height:1.5}
.gcard .gb{color:var(--amber);font-size:10px;margin-top:6px;display:block}
.gloss-empty{color:var(--faint);font-size:12px;padding:20px 0}
mark{background:rgba(255,181,71,.28);color:var(--ink);border-radius:2px}
/* expandable token lists (tokenizer forensics) */
.tokdex{margin-top:8px;border:1px solid var(--line);background:var(--bg)}
.tokdex>summary{cursor:pointer;padding:7px 10px;font-size:11px;color:var(--ink);
  user-select:none;list-style:none;display:flex;align-items:center;gap:8px}
.tokdex>summary::-webkit-details-marker{display:none}
.tokdex>summary::before{content:"▸";color:var(--amber);font-size:10px;transition:transform .12s}
.tokdex[open]>summary::before{transform:rotate(90deg)}
.tokdex>summary .cnt{color:var(--faint);font-weight:400}
.tokdex .body{padding:4px 10px 10px;border-top:1px solid var(--line);max-height:340px;overflow:auto}
.tokrow{display:flex;gap:10px;align-items:baseline;padding:3px 0;border-bottom:1px dotted #131c18;font-size:10.5px}
.tokrow code{color:var(--phos);background:#0d1512;border:1px solid var(--line);padding:1px 5px;white-space:nowrap;flex-shrink:0}
.tokrow.unk code{color:var(--red)}
.tokrow.res code{color:var(--amber)}
.tokrow .ex{color:var(--dim);line-height:1.45}
.tokrow .ex b{color:var(--cyan);font-weight:600}
.tokflat{padding:6px 0;display:flex;flex-wrap:wrap;gap:4px 6px}
.tokflat code{color:var(--amber);background:#0d1512;border:1px solid var(--line);padding:1px 5px;font-size:10px}
.tokflat.unk code{color:var(--red)}
/* run-log tab */
.logbadge{display:inline-flex;align-items:center;gap:3px;font-size:10px;font-weight:700;
  padding:2px 7px;margin-left:6px;border:1px solid var(--amber);color:var(--amber);
  position:relative;top:-1px;letter-spacing:.5px;white-space:nowrap}
.logbadge.err{border-color:var(--red);color:var(--red)}
.log-counts{display:flex;gap:14px;font-size:12px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.log-counts .c{display:inline-flex;align-items:center;gap:6px}
.log-counts .c b{font-size:16px;font-weight:700}
.log-counts .dot{width:9px;height:9px;display:inline-block;border-radius:50%}
.log-filters{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.log-filters select{background:var(--bg);border:1px solid var(--line);color:var(--ink);
  font-family:var(--mono);font-size:11px;padding:5px 9px}
.log-filters select:focus{outline:none;border-color:var(--phos)}
.log-filters .lf-l{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.5px}
table.logtbl{border-collapse:collapse;width:100%;font-size:11px}
table.logtbl th{text-align:left;color:var(--dim);font-weight:500;padding:7px 10px;
  border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.5px;font-size:10px;
  position:sticky;top:0;background:var(--bg);white-space:nowrap}
table.logtbl td{padding:6px 10px;border-bottom:1px dotted #131c18;vertical-align:top}
table.logtbl tr.sev-info{}
table.logtbl tr.sev-warn td{background:rgba(255,181,71,.06)}
table.logtbl tr.sev-error td{background:rgba(255,93,93,.08)}
table.logtbl .sevb{font-size:9px;font-weight:700;padding:2px 7px;text-transform:uppercase;
  letter-spacing:.5px;border:1px solid;white-space:nowrap}
table.logtbl .sevb.info{color:var(--phos);border-color:var(--phos)}
table.logtbl .sevb.warn{color:var(--amber);border-color:var(--amber)}
table.logtbl .sevb.error{color:var(--red);border-color:var(--red)}
table.logtbl .lts{color:var(--faint);white-space:nowrap;font-size:10px}
table.logtbl .lmodel{color:var(--cyan)}
table.logtbl .lmodel.jump{cursor:pointer;text-decoration:underline dotted}
table.logtbl .lstage{color:var(--dim)}
table.logtbl .lcode{color:var(--violet)}
table.logtbl .lmsg{color:var(--ink);word-break:break-word}
.log-empty{color:var(--faint);font-size:12px;padding:20px}
.cmp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:14px}
.cmp-card{background:var(--panel);border:1px solid var(--line);padding:12px}
.cmp-card .ct{font-size:11px;color:var(--phos);font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.cmp-card .cv{font-size:22px;color:var(--ink);font-weight:700}
.cmp-card .cd{font-size:10px;color:var(--dim);margin-top:4px}
.cmp-list{display:grid;gap:8px}
.cmp-row{border:1px solid var(--line);background:var(--panel);padding:10px;font-size:11px}
.cmp-row .pair{color:var(--ink);font-weight:700;margin-bottom:4px}
.cmp-row .meta{color:var(--dim)}
.cmp-row .score{color:var(--phos);font-weight:700}
.cmp-row .warn{color:var(--amber)}
.cmp-row .hot{color:var(--red)}
.cmp-tags{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}
.cmp-tags span{border:1px solid var(--line);color:var(--dim);padding:2px 6px;font-size:9px}
.tour-panel{position:fixed;right:18px;bottom:18px;z-index:1100;width:min(430px,calc(100vw - 36px));
  background:linear-gradient(180deg,#101815,#070b0a);border:1px solid var(--phos);
  box-shadow:0 18px 60px rgba(0,0,0,.62),0 0 0 1px rgba(74,222,128,.12) inset;
  padding:14px;display:none}
.tour-panel.on{display:block}
.tour-panel .th{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:8px}
.tour-panel .ttl{font-size:13px;color:var(--ink);font-weight:700}
.tour-panel .step{font-size:10px;color:var(--faint);white-space:nowrap}
.tour-panel .txt{font-size:12px;color:var(--dim);line-height:1.5;margin-bottom:12px}
.tour-panel .demo{border:1px solid var(--line);background:#0a0f0d;padding:9px;margin-bottom:12px;
  display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:10px}
.tour-panel .demo b{display:block;color:var(--ink);font-size:12px}
.tour-quiz{border:1px dashed var(--line);background:#080d0b;padding:9px;margin-bottom:12px;
  font-size:11px;color:var(--dim);line-height:1.45;display:none}
.tour-quiz.on{display:block}
.tour-quiz .q{color:var(--ink);font-weight:700;margin-bottom:6px}
.tour-quiz .a{color:var(--phos);margin-top:7px;display:none}
.tour-quiz .a.on{display:block}
.tour-actions{display:flex;gap:8px;justify-content:space-between;align-items:center}
.tour-actions .left,.tour-actions .right{display:flex;gap:8px;align-items:center}
.tour-btn{background:#0b100e;border:1px solid var(--line);color:var(--ink);font-size:11px;
  padding:7px 10px;cursor:pointer}
.tour-btn:hover{border-color:var(--phos);color:var(--phos)}
.tour-btn.primary{background:var(--phos);border-color:var(--phos);color:var(--bg);font-weight:700}
.tour-focus{position:relative;z-index:901;outline:2px solid var(--phos);
  box-shadow:0 0 0 5px rgba(74,222,128,.14),0 0 26px rgba(74,222,128,.28)}
.tour-fade{position:fixed;inset:0;z-index:900;pointer-events:none;background:rgba(0,0,0,.18);display:none}
.tour-fade.on{display:block}
.foot{margin-top:50px;color:var(--faint);font-size:10px;border-top:1px solid var(--line);padding-top:14px}
@media(max-width:880px){.kpis{grid-template-columns:repeat(2,1fr)}.plots{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <div class="title">model<span class="x">·</span>strata</div>
      <div class="subtitle" data-i18n="subtitle">Statische Modellanalyse für GGUF, HF-safetensors und Inventar-Formate. Das Dashboard lädt nur JSON-Reports: Architektur, Tokenizer, Quantisierung, Gewichts-Summaries und Degradationen — ohne Forward-Pass.</div>
    </div>
    <div class="stamp">
      <div id="langtog" class="chip" style="cursor:pointer;display:inline-block;margin-bottom:6px" onclick="setLang(LANG==='de'?'en':'de')" title="Sprache / language">EN</div><br>
      <div id="tutorial-start" class="chip" style="cursor:pointer;display:inline-block;margin-bottom:6px" data-i18n="tour-start">Tutorial</div><br>
      <div id="quiz-start" class="chip" style="cursor:pointer;display:inline-block;margin-bottom:6px" data-i18n="quiz-start">Quiz</div><br>
      static reports · CPU-only<br>
      generated by <b>modelstrata</b><br>
      <span id="stamp-count"></span>
    </div>
  </header>

  <nav class="tabs" id="tabs">
    <div class="tab" data-tab="fleet" data-i18n="tab-fleet">Flotte</div>
    <div class="tab" data-tab="tok" data-i18n="tab-tok">Tokenizer</div>
    <div class="tab" data-tab="weights" data-i18n="tab-weights">Gewichte</div>
    <div class="tab" data-tab="embdiff" data-i18n="tab-embdiff">Embedding &amp; Diff</div>
    <div class="tab" data-tab="compare" data-i18n="tab-compare">Vergleich</div>
    <div class="tab" data-tab="gloss" data-i18n="tab-gloss">Glossar</div>
    <div class="tab" data-tab="log" data-i18n="tab-log">Log</div><span id="log-badge" class="logbadge" style="display:none"></span>
  </nav>

  <div class="tabpane" data-tab="fleet">
  <div class="kpis" id="kpis"></div>

  <div class="section-h"><span class="n">01</span><span class="t" data-i18n="s1">Fleet</span><span class="line"></span></div>
  <div class="controls" id="filters"></div>
  <div class="fleet" id="fleet"></div>

  <div class="section-h"><span class="n">02</span><span class="t" data-i18n="s2">Architektur-Topologie</span><span class="line"></span></div>
  <div class="plots">
    <div class="plot">
      <h4 data-i18n="p-tw">Tiefe vs. Breite</h4>
      <div class="cap" data-i18n="cap-tw">x = Parameter (log) · y = Layer · Radius = d_model · Farbe = Architektur</div>
      <svg id="scatter" viewBox="0 0 560 360"></svg>
    </div>
    <div class="plot">
      <h4 data-i18n="p-gqa">GQA &amp; FFN-Verhältnis</h4>
      <div class="cap" data-i18n="cap-gqa">Balken = KV-Kompression (head/kv) · Punkt = FFN-Expansion (ffn/d_model)</div>
      <svg id="ratios" viewBox="0 0 560 360"></svg>
    </div>
  </div>
  </div><!-- /fleet pane -->

  <div class="tabpane" data-tab="tok">
  <div class="section-h"><span class="n">03</span><span class="t" data-i18n="s3">Tokenizer-Forensik</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-tok" style="margin-bottom:14px">Vokab-Overlap (Jaccard) zwischen allen Modellen — hell = gemeinsame Tokenizer-Herkunft. <span class="hq" data-help="Jaccard = gemeinsame Tokens ÷ Vereinigung. 1.0 = identisches Vokabular, oft ein Finetune des anderen.">?</span> Legende: <b style="color:var(--phos)">■</b> = identisch (≥99,5 %), Zahl = Jaccard in % (leer = &lt;10 %). Klick auf eine Zelle öffnet das Modell-Detail.</div>
  <div id="overlap-wrap" class="hm-wrap" style="max-width:100%"></div>
  </div><!-- /tok pane -->

  <div class="tabpane" data-tab="weights">
  <div class="section-h"><span class="n">04</span><span class="t" data-i18n="s4">Gewichts-Statistik</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-ws" style="margin-bottom:12px">Aus den <b>dequantisierten</b> Gewichten berechnet (Ebene 4) — kein Forward-Pass. <span class="hq" data-help="GGUF-Quants werden rekonstruiert (approx). HF-safetensors werden in der gespeicherten Float-Präzision gelesen und nach fp32 dekodiert.">?</span> Wähle Modell und Metrik; Heatmap = Layer × Komponente.</div>
  <div id="ws-controls" style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap"></div>
  <div id="ws-wrap" class="hm-wrap" style="max-width:100%"></div>

  <div class="section-h"><span class="n">05</span><span class="t" data-i18n="s5">Spektral-Analyse (WeightWatcher)</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-sp" style="margin-bottom:12px">Singulärwerte pro Gewichtsmatrix (Ebene 5). <b>alpha</b> = geschätzter Heavy-Tail-Exponent; WeightWatcher interpretiert Werte grob im Bereich 2–6 als typisch für gut korrelierte trainierte Layer, &gt;6 als Warnsignal. <span class="hq" data-help="Hier nur eine einfache Hill-Schätzung auf dequantisierten 2D-Gewichten, nicht die vollständige WeightWatcher-Analyse.">?</span></div>
  <div id="sp-controls" style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap"></div>
  <div id="sp-wrap" class="hm-wrap" style="max-width:100%"></div>
  </div><!-- /weights pane -->

  <div class="tabpane" data-tab="embdiff">
  <div class="section-h"><span class="n">06</span><span class="t" data-i18n="s6">Embedding-Geometrie</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-emb" style="margin-bottom:12px">Statische Token-Embedding-Matrix (Ebene 6): 2D-PCA (Punkt = Token, Farbe = Norm), Norm-Histogramm und niedrigste Normen (stärkere Glitch-Token-Hinweise als Namensheuristik). <span class="hq" data-help="Aus dequantisiertem token_embd. Tokens mit ~0-Norm können untertrainiert oder selten gesehen sein (à la SolidGoldMagikarp).">?</span></div>
  <div id="emb-controls" style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap"></div>
  <div id="emb-stats" class="kv2" style="margin-bottom:12px"></div>
  <div style="display:grid;grid-template-columns:1.3fr 1fr;gap:14px" id="emb-grid">
    <div class="plot"><h4>PCA (2D)</h4><div class="cap" id="emb-evr"></div><svg id="emb-scatter" viewBox="0 0 480 360"></svg></div>
    <div class="plot"><h4>Norm-Histogramm</h4><div class="cap">Per-Token-L2-Norm</div><svg id="emb-hist" viewBox="0 0 480 360"></svg></div>
  </div>
  <div id="emb-low" style="margin-top:12px;font-size:10px;color:var(--dim)"></div>

  <div class="section-h"><span class="n">07</span><span class="t" data-i18n="s7">Modell-Diff ★</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-diff" style="margin-bottom:10px">Per-Tensor-Veränderung zwischen zwei Modellen (Ebene 7) — <b>wo</b> hat ein Finetune/eine Abliteration das Modell verändert, ohne Forward-Pass. <span class="hq" data-help="delta = relative Frobenius-Differenz, cosine = Richtungs-Ähnlichkeit. Hohes delta / niedriges cosine = stark verändert.">?</span></div>
  <div id="df-note" style="margin-bottom:8px"></div>
  <div id="df-controls" style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap"></div>
  <div id="df-wrap" class="hm-wrap" style="max-width:100%"></div>
  <div id="df-top" style="margin-top:12px;font-size:10px;color:var(--dim)"></div>
  </div><!-- /embdiff pane -->

  <div class="tabpane" data-tab="compare">
  <div class="section-h"><span class="n">08</span><span class="t" data-i18n="s10">Statische Vergleiche</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-compare" style="margin-bottom:12px">Architektur-, Tokenizer-, Quantisierungs-, Lineage- und Anomalie-Vergleiche aus den vorhandenen Reports. <span class="hq" data-help="Alles bleibt statisch: kein Prompt, kein Forward-Pass. Scores sind Heuristiken aus messbaren Datei-/Gewichts-Signalen.">?</span></div>
  <div id="cmp-summary" class="cmp-grid"></div>
  <div class="section-h"><span class="n">08a</span><span class="t">Lineage</span><span class="line"></span></div>
  <div id="cmp-lineage" class="cmp-list"></div>
  <div class="section-h"><span class="n">08b</span><span class="t">Architektur &amp; Quant</span><span class="line"></span></div>
  <div id="cmp-archquant" class="cmp-list"></div>
  <div class="section-h"><span class="n">08c</span><span class="t">Anomalien &amp; Diff-Erklärung</span><span class="line"></span></div>
  <div id="cmp-anomdiff" class="cmp-list"></div>
  </div><!-- /compare pane -->

  <div class="tabpane" data-tab="gloss">
  <div class="section-h"><span class="n">09</span><span class="t" data-i18n="s8">Glossar · was bedeutet was</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-gloss" style="margin-bottom:14px">Jede Kennzahl und jeder Fachbegriff im Klartext, inkl. „mehr oder weniger besser?“. Dieselben Texte erscheinen als <span class="hq" data-help="So sieht ein Hilfe-Hinweis aus.">?</span>-Tooltip überall im Dashboard. Nach Themen gruppiert; tippen filtert live.</div>
  <div class="gloss-tools">
    <input type="search" id="gloss-search" class="gloss-search" data-i18n-ph="gloss-ph" placeholder="suchen … (Titel + Beschreibung)" autocomplete="off" spellcheck="false">
    <span class="gloss-count" id="gloss-count"></span>
  </div>
  <div id="gloss"></div>
  </div><!-- /gloss pane -->

  <div class="tabpane" data-tab="log">
  <div class="section-h"><span class="n">10</span><span class="t" data-i18n="s9">Run-Log · Degradationen</span><span class="line"></span></div>
  <div class="mcap" data-i18n="c-log" style="margin-bottom:12px">Strukturierte Lauf-Protokolleinträge aus <b>reports/run_log.json</b> (Task 15) — jeder Fallback, Skip oder Fehler, der die Analyse degradiert hat. <span class="hq" data-help="EINE Wahrheit: Was hier nicht steht, ist nicht passiert. info = ok, warn = degradiert/approximiert, error = Stufe fehlgeschlagen.">?</span> Neueste zuerst.</div>
  <div id="log-counts" class="log-counts"></div>
  <div id="log-filters" class="log-filters"></div>
  <div id="log-wrap" class="hm-wrap" style="max-width:100%;padding:0"></div>
  </div><!-- /log pane -->

  <div class="foot" id="foot"></div>
</div>

<div class="modal" id="modal">
  <div class="sheet">
    <div class="shd">
      <div><div class="nm" id="m-nm"></div><div class="sub" id="m-sub"></div></div>
      <div class="close" id="m-close">[ schliessen ]</div>
    </div>
    <div class="sbody">
      <div class="specrow" id="m-specs" style="grid-template-columns:repeat(6,1fr)"></div>

      <div class="msec" id="m-warn-sec" style="display:none">
        <div class="mh" data-i18n="m-warn">Quelle &amp; Degradationen <span class="hq" data-help="Format/Präzision der Quelle und alle Degradations-Hinweise (source.warnings) für dieses Modell — z.B. übersprungene Tensoren, fehlende lm_head, nicht gemappte Architektur.">?</span></div>
        <div class="mcap" data-i18n="mc-warn">Pro-Modell-Degradationen aus dem source-Block (Multi-Format-Pipeline). Leer = nichts degradiert.</div>
        <div id="m-warnings"></div>
      </div>

      <div class="msec">
        <div class="mh" data-i18n="m1">Steckbrief · abgeleitete Größen <span class="hq" data-help="Alles hier ist rein aus dem Header berechnet — kein Forward-Pass, keine Gewichte gelesen.">?</span></div>
        <div class="mcap" data-i18n="mc-steck">Effizienz, Provenienz und Plausibilität — komplett header-only abgeleitet (Ebene 1b).</div>
        <div class="kv2" id="m-derived"></div>
      </div>

      <div class="msec">
        <div class="mh" data-i18n="m2">Tokenizer &amp; Special Tokens <span class="hq" data-help="Wie das Modell Text in Tokens zerlegt und welche Steuer-Tokens es kennt. Bestimmt Chat-Format und Kompatibilität.">?</span></div>
        <div class="kv2" id="m-tok"></div>
        <div id="m-toksample" style="margin-top:10px;font-size:10px;color:var(--dim)"></div>
      </div>

      <div class="msec">
        <div class="mh" data-i18n="m3">Roher GGUF-Header · hex / ascii <span class="hq" data-help="Die ersten 64 Bytes der Datei, wie sie auf der Platte liegen. Magic 'GGUF' (ASCII), dann Version, Tensor- und KV-Anzahl als little-endian Integer.">?</span></div>
        <div class="hex" id="m-hex"></div>
        <div class="kv2" id="m-layout" style="margin-top:12px"></div>
      </div>

      <div class="msec">
        <div class="mh" data-i18n="m4">Tensor-Internals · Layer × Komponente <span class="hq" data-help="Jede Zelle = ein Gewichts-Tensor, eingefärbt nach Quantisierungstyp. Spalte = Komponente (q/k/v/o, ffn, norms), Zeile = Block. Norm-Spalten bleiben meist F32 (hell). Biases sind 1-D und ausgeblendet (separat gezählt).">?</span></div>
        <div class="mcap" data-i18n="mc-tensor">Layer-Topologie + Quant-Map aus dem Tensor-Verzeichnis (Ebene 2). Hover für Details.</div>
        <div class="hm-wrap"><table class="hm" id="m-hm"></table></div>
        <div class="gl" id="m-leg"></div>
        <div id="m-glob" style="margin-top:16px;font-size:10px;color:var(--dim)"></div>
      </div>

      <div class="msec">
        <div class="mh" data-i18n="m5">Komplette Metadaten · alle <span id="m-kvn"></span> Key-Values <span class="hq" data-help="Jeder einzelne Metadaten-Eintrag aus dem GGUF-Header, nichts weggelassen. Große Arrays (Tokens/Merges) als Länge + Stichprobe.">?</span></div>
        <div class="mcap" data-i18n="mc-meta">Vollständiger Header-Dump (Ebene 1), gruppiert. Lange Strings (z.B. chat_template) aufklappbar.</div>
        <div id="m-meta"></div>
      </div>
    </div>
  </div>
</div>
<div class="tip" id="tip"></div>
<div class="tour-fade" id="tour-fade"></div>
<div class="tour-panel" id="tour-panel" role="dialog" aria-live="polite" aria-label="Tutorial">
  <div class="th"><div class="ttl" id="tour-title"></div><div class="step" id="tour-step"></div></div>
  <div class="txt" id="tour-text"></div>
  <div class="demo" id="tour-demo"></div>
  <div class="tour-quiz" id="tour-quiz"></div>
  <div class="tour-actions">
    <div class="left"><button class="tour-btn" id="tour-exit" type="button">Schliessen</button></div>
    <div class="right">
      <button class="tour-btn" id="tour-answer" type="button">Antwort</button>
      <button class="tour-btn" id="tour-prev" type="button">Zurueck</button>
      <button class="tour-btn primary" id="tour-next" type="button">Weiter</button>
    </div>
  </div>
</div>

<script>
const MODELS = __DATA__;
const FORENSICS = __FORENSICS__;
const WSTATS = __WSTATS__;   // Ebene 4: weight statistics (list of per-model dicts)
const SPECTRAL = __SPECTRAL__;  // Ebene 5: spectral metrics (same shape as WSTATS)
const EMBED = __EMBED__;        // Ebene 6: embedding geometry (list of per-model dicts)
const DIFF = __DIFF__;          // Ebene 7: model diff (statGrid shape + note/top)
const RUNLOG = __RUNLOG__;      // Task 15/16: [{ts,model,stage,severity,code,msg}]
const COMPARE = __COMPARE__;    // static cross-model comparisons
// resolve a model to its tokenizer-forensics entry via label or any alias
function forOf(m){
  const fm=(FORENSICS&&FORENSICS.models)||{};
  for(const k of [m.label,...(m.aliases||[])]) if(fm[k]) return fm[k];
  return null;
}
function forLabel(m){
  const fm=(FORENSICS&&FORENSICS.models)||{};
  for(const k of [m.label,...(m.aliases||[])]) if(fm[k]) return k;
  return null;
}

// ---- quant color scale (phosphor->amber->red by "cost") -------------------
const QCOLOR = {
  "F32":"#e9f5ef","F16":"#9fe7c4","BF16":"#7ddfb8","Q8_0":"#4ade80","Q8_K":"#3fcf72",
  "Q6_K":"#9bd64a","Q5_K":"#d6c54a","Q5_0":"#e0b73e","Q4_K":"#ffb547","Q4_0":"#ff8c42",
  "Q3_K":"#ff6b4a","Q2_K":"#ff5d5d","IQ4_NL":"#ffa047","IQ4_XS":"#ffac5e"
};
const qc = t => QCOLOR[t] || "#6f8279";
const ARCHCOLOR = {qwen2:"#38e1d4",qwen3:"#4ade80",llama:"#ffb547",gemma3:"#b39dff",
  gemma4:"#c9a7ff",mistral3:"#ff8c42",glm4moelite:"#ff5d5d"};
const ac = a => ARCHCOLOR[a] || "#9fe7c4";

const hasN = n => n !== null && n !== undefined && n !== "" && Number.isFinite(+n);
const fmtN = n => { if(!hasN(n)) return "–"; n=+n; const u=["","K","M","B","T"]; let i=0;
  while(Math.abs(n)>=1000&&i<u.length-1){n/=1000;i++;} return (i?n.toFixed(1):n)+u[i]; };
const fmtB = n => { if(!hasN(n)) return "–"; n=+n; const u=["B","KiB","MiB","GiB","TiB"]; let i=0;
  while(Math.abs(n)>=1024&&i<u.length-1){n/=1024;i++;} return n.toFixed(1)+u[i]; };
// HTML-escape (GGUF metadata is untrusted). Defined early: used by card/scatter/ratios.
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));

// ---- i18n: German is the source; EN maps German UI strings to English -----
let LANG = (window.localStorage && localStorage.getItem("mm-lang")) || "de";
const EN = {
  // KPI + controls + generic
  "MODELLE":"MODELS","ARCHITEKTUREN":"ARCHITECTURES","Σ PARAMETER":"Σ PARAMETERS",
  "Σ ON-DISK":"Σ ON-DISK","MULTIMODAL":"MULTIMODAL","Modell":"Model","Metrik":"Metric",
  "alle":"all","schliessen":"close","ja":"yes","nein":"no","Bereich":"Range","andere":"other",
  "Modelle":"Models","Tokenizer":"Tokenizers","Lineage-Paare":"Lineage pairs","Anomalie-Sets":"Anomaly sets",
  "im Vergleichsreport":"in comparison report",
  "direkt aus Modellpfaden gelesen":"read directly from model paths",
  "heuristische Paar-Scores":"heuristic pair scores",
  "aus Gewichtsreports":"from weight reports",
  "keine Vergleichsdaten geladen":"no comparison data loaded",
  "keine Lineage-Paare":"no lineage pairs",
  "keine Architektur-/Quant-Paare":"no architecture/quant pairs",
  "keine Anomalien oder Diff-Erklärungen":"no anomalies or diff explanations",
  // source / precision badge + modal degradation section
  "Quelle":"source","exact":"exact","approx":"approx","inventory-only":"inventory-only",
  "keine Degradationen":"no degradations",
  "Quelle & Degradationen":"Source & degradations",
  // modal section headers
  "Steckbrief · abgeleitete Größen":"Profile · derived values",
  "Tokenizer & Special Tokens":"Tokenizer & special tokens",
  "Roher GGUF-Header · hex / ascii":"Raw GGUF header · hex / ascii",
  "Tensor-Internals · Layer × Komponente":"Tensor internals · layer × component",
  "Komplette Metadaten · alle":"Complete metadata · all",
  "Key-Values":"key-values",
  // derived / kvrow labels
  "bits / weight":"bits / weight","file_type":"file_type","dominanter Tensor-Quant":"dominant tensor quant",
  "head_dim":"head_dim","FFN-Verhältnis":"FFN ratio","RoPE freq_base":"RoPE freq_base",
  "Kontext gestreckt":"context extended","Sliding-Window":"sliding window","MoE Experten":"MoE experts",
  "Tied Embeddings":"tied embeddings","Vokabular":"vocabulary","Bias-Tensoren":"bias tensors",
  "GGUF-Version":"GGUF version","Alignment":"alignment","Daten-Bytes":"data bytes",
  "aktiv":"active","konsistent":"consistent","vgl. file_type":"cf. file_type",
  "Tokenizer-Modell":"tokenizer model","Pre-Tokenizer":"pre-tokenizer","Vokabulargröße":"vocab size",
  "BPE-Merges":"BPE merges","BOS-Token":"BOS token","EOS-Token":"EOS token",
  "PAD-Token":"PAD token","UNK-Token":"UNK token","Chat-Template":"chat template",
  "Token-Typen":"token types","reservierte/unbenutzte Slots":"reserved/unused slots",
  "funktionale Special-Tokens":"functional special tokens","Skripte (Stichprobe":"scripts (sample",
  "Magic":"magic","Version":"version","n_kv (Metadaten)":"n_kv (metadata)",
  "Header-Ende (Byte)":"header end (byte)","Daten-Start (Byte)":"data start (byte)",
  "Embedding-Dim":"embedding dim","Anisotropie":"anisotropy","Norm Ø / min / max":"norm avg / min / max",
  "Tokens mit ~0-Norm":"tokens with ~0 norm",
  // prose fragments in dynamic renders
  "Token-Stichprobe":"token sample","Reserviert (Sample):":"reserved (sample):",
  "Special (Sample):":"special (sample):","Globale Tensoren":"global tensors",
  "Niedrigste Normen (Glitch-Hinweise):":"lowest norms (glitch indicators):",
  "Am stärksten veränderte Tensoren (Top":"most-changed tensors (top",
  "keine Daten geladen (mit":"no data loaded (build with","bauen)":")",
  "erklärte Varianz":"explained variance","Tokens (Farbe=Norm)":"tokens (colour=norm)",
  // tokenizer forensics: expandable token lists
  "UNKNOWN-Tokens":"UNKNOWN tokens","reservierte / unbenutzte Slots":"reserved / unused slots",
  "funktionale Special-Tokens":"functional special tokens",
  "(erste":"(first","erklärt":"explained",
  "UNKNOWN-Liste: Forensik neu bauen (tokenizer_forensics.py)":
    "UNKNOWN list: rebuild forensics (tokenizer_forensics.py)",
  // run-log tab (dynamic render)
  "Zeit":"time","Schwere":"severity","Stufe":"stage","Code":"code","Meldung":"message",
  "alle Schweren":"all severities","alle Modelle":"all models","alle Stufen":"all stages",
  "keine Log-Einträge":"no log entries",
  "keine Einträge für diesen Filter":"no entries for this filter",
  "Filter:":"filter:",
};
// EN HTML for static, markup-bearing strings (keyed by data-i18n)
const I18N_HTML = {
  "subtitle":"Static model analysis for GGUF, HF safetensors and inventory formats. The dashboard loads only JSON reports: architecture, tokenizer, quantization, weight summaries and degradations — no forward pass.",
  "tour-start":"Tutorial","quiz-start":"Quiz",
  "tab-fleet":"Fleet","tab-tok":"Tokenizer","tab-weights":"Weights","tab-embdiff":"Embedding &amp; Diff","tab-compare":"Compare","tab-gloss":"Glossary","tab-log":"Log",
  "s1":"Fleet","s2":"Architecture topology","s3":"Tokenizer forensics","s4":"Weight statistics",
  "s5":"Spectral analysis (WeightWatcher)","s6":"Embedding geometry","s7":"Model diff ★","s8":"Glossary · what means what",
  "s9":"Run log · degradations","s10":"Static comparisons",
  "p-tw":"Depth vs. width","cap-tw":"x = parameters (log) · y = layers · radius = d_model · colour = architecture",
  "p-gqa":"GQA &amp; FFN ratio","cap-gqa":"bar = KV compression (head/kv) · dot = FFN expansion (ffn/d_model)",
  "c-log":`Structured run-log entries from <b>reports/run_log.json</b> (task 15) — every fallback, skip or error that degraded the analysis. <span class="hq" data-help="ONE truth: what is not here did not happen. info = ok, warn = degraded/approximated, error = stage failed.">?</span> Newest first.`,
  "c-tok":`Vocabulary overlap (Jaccard) across all models — bright = shared tokenizer lineage. <span class="hq" data-help="Jaccard = shared tokens ÷ union. 1.0 = identical vocabulary, often a finetune of the other.">?</span> Legend: <b style="color:var(--phos)">■</b> = identical (≥99.5%), number = Jaccard in % (blank = &lt;10%). Click a cell to open the model detail.`,
  "c-ws":`Computed from the <b>dequantized</b> weights (level 4) — no forward pass. <span class="hq" data-help="GGUF quants are reconstructed (approx). HF safetensors are read at their stored float precision and decoded to fp32.">?</span> Pick model and metric; heatmap = layer × component.`,
  "c-sp":`Singular values per weight matrix (level 5). <b>alpha</b> = estimated heavy-tail exponent; WeightWatcher treats roughly 2–6 as typical for well-correlated trained layers, &gt;6 as a warning sign. <span class="hq" data-help="This is a simple Hill estimate on dequantized 2D weights, not the full WeightWatcher analysis.">?</span>`,
  "c-emb":`Static token-embedding matrix (level 6): 2D PCA (point = token, colour = norm), norm histogram and lowest norms (stronger glitch-token indicators than name heuristics). <span class="hq" data-help="From dequantized token_embd. Tokens with ~0 norm may be under-trained or rarely seen (SolidGoldMagikarp-style).">?</span>`,
  "c-diff":`Per-tensor change between two models (level 7) — <b>where</b> did a finetune/abliteration change the model, without a forward pass. <span class="hq" data-help="delta = relative Frobenius difference, cosine = directional similarity. High delta / low cosine = strongly changed.">?</span>`,
  "c-compare":`Architecture, tokenizer, quantization, lineage and anomaly comparisons from the available reports. <span class="hq" data-help="Still static: no prompt, no forward pass. Scores are heuristics from measurable file/weight signals.">?</span>`,
  "c-gloss":`Every metric and term in plain words, incl. „more or less better?“. The same texts appear as <span class="hq" data-help="This is what a help hint looks like.">?</span> tooltips throughout the dashboard. Grouped by topic; typing filters live.`,
  "m-warn":`Source &amp; degradations <span class="hq" data-help="Source format/precision and every degradation hint (source.warnings) for this model — e.g. skipped tensors, missing lm_head, unmapped architecture.">?</span>`,
  "mc-warn":"Per-model degradations from the source block (multi-format pipeline). Empty = nothing degraded.",
  "mc-steck":"Efficiency, provenance and plausibility — derived header-only (level 1b).",
  "mc-tensor":"Layer topology + quant map from the tensor directory (level 2). Hover for details.",
  "mc-meta":"Complete header dump (level 1), grouped. Long strings (e.g. chat_template) expandable.",
  "m1":`Profile · derived values <span class="hq" data-help="Everything here is computed from the header only — no forward pass, no weights read.">?</span>`,
  "m2":`Tokenizer &amp; special tokens <span class="hq" data-help="How the model splits text into tokens and which control tokens it has. Determines chat format and compatibility.">?</span>`,
  "m3":`Raw GGUF header · hex / ascii <span class="hq" data-help="The first 64 bytes as on disk. Magic 'GGUF' (ASCII), then version, tensor and KV counts as little-endian integers.">?</span>`,
  "m4":`Tensor internals · layer × component <span class="hq" data-help="Each cell = a weight tensor coloured by quant type. Column = component (q/k/v/o, ffn, norms), row = block. Norm columns are usually F32 (bright). Biases are 1-D and hidden (counted separately).">?</span>`,
  "m5":`Complete metadata · all <span id="m-kvn"></span> key-values <span class="hq" data-help="Every single metadata entry from the GGUF header, nothing dropped. Large arrays (tokens/merges) as length + sample.">?</span>`,
};
const I18N_PH = { "gloss-ph":"search … (title + description)" };
const tr = s => (LANG === "en" && EN[s] != null) ? EN[s] : s;
function setLang(l){
  LANG = l;
  try{ localStorage.setItem("mm-lang", l); }catch(e){}
  applyLang();
  if(typeof TOUR_ON !== "undefined" && TOUR_ON) renderTutorialStep();
}
function applyLang(){
  // static strings carrying markup: capture the German innerHTML once, swap to EN
  document.querySelectorAll("[data-i18n]").forEach(el=>{
    if(el.dataset.de === undefined) el.dataset.de = el.innerHTML;
    el.innerHTML = (LANG === "en" && I18N_HTML[el.dataset.i18n] != null)
      ? I18N_HTML[el.dataset.i18n] : el.dataset.de;
  });
  // translatable placeholders (e.g. glossary search)
  document.querySelectorAll("[data-i18n-ph]").forEach(el=>{
    if(el.dataset.deph === undefined) el.dataset.deph = el.getAttribute("placeholder");
    el.setAttribute("placeholder", (LANG === "en" && I18N_PH[el.dataset.i18nPh] != null)
      ? I18N_PH[el.dataset.i18nPh] : el.dataset.deph);
  });
  const tg = document.getElementById("langtog"); if(tg) tg.textContent = LANG === "de" ? "EN" : "DE";
  const ac = document.querySelector('#filters [data-arch="all"]'); if(ac) ac.textContent = tr("alle");
  // re-render dynamic, language-dependent content
  renderKPIs(); renderFleet(); renderOverlap();
  statGrid(WSTATS,"ws-controls","ws-wrap","--weight-stats");
  statGrid(SPECTRAL,"sp-controls","sp-wrap","--spectral");
  statGrid(DIFF,"df-controls","df-wrap","--diff");
  embControls(); renderDiffExtra(); renderGlossary();
  renderCompare();
  renderLog(); renderLogBadge();
}

// ---- KPIs -----------------------------------------------------------------
const totP = MODELS.reduce((s,m)=>s+(+m.params||0),0);
const totB = MODELS.reduce((s,m)=>s+(+m.file_size||0),0);
const archs = [...new Set(MODELS.map(m=>m.arch))];
function renderKPIs(){
  const kpis=[
    ["MODELLE", MODELS.length],["ARCHITEKTUREN", archs.length],
    ["Σ PARAMETER", fmtN(totP)],["Σ ON-DISK", fmtB(totB)],
    ["MULTIMODAL", MODELS.filter(m=>m.vision||m.audio).length],
  ];
  document.getElementById("kpis").innerHTML = kpis.map(k=>
    `<div class="kpi"><div class="v">${k[1]}</div><div class="l">${tr(k[0])}</div></div>`).join("");
  document.getElementById("stamp-count").innerHTML =
    MODELS.length+" "+tr("MODELLE").toLowerCase()+" · "+archs.length+" "+tr("ARCHITEKTUREN").toLowerCase();
}

// ---- filters + fleet ------------------------------------------------------
let active = "all";
const fc = document.getElementById("filters");
["all",...archs].forEach(a=>{
  const c=document.createElement("div"); c.className="chip"+(a==="all"?" on":""); c.dataset.arch=a;
  c.textContent = a==="all"?tr("alle"):a; c.onclick=()=>{active=a;
    [...fc.children].forEach(x=>x.classList.remove("on")); c.classList.add("on"); renderFleet();};
  fc.appendChild(c);
});

// ---- source badge: "format · precision", colour-hinted by precision -------
// exact=green, approx=amber, inventory-only=grey. Missing/unknown source -> no
// badge (graceful). format/precision come from the model's source block.
function srcBadge(src){
  if(!src||!src.format||!src.precision) return "";
  const cls = src.precision==="exact" ? "exact"
            : src.precision==="approx" ? "approx" : "inv";
  // precision label is translatable; "inventory-only" stays as the source token
  const txt = `${esc(src.format)} · ${esc(tr(src.precision))}`;
  return `<div class="srcb ${cls}" title="${esc(tr("Quelle"))}: ${esc(src.format)} / ${esc(src.precision)}">${txt}</div>`;
}

function card(m,i){
  const qtot = Object.values(m.quant).reduce((a,b)=>a+b,0);
  const qents = Object.entries(m.quant).sort((a,b)=>b[1]-a[1]);
  const bar = qtot ? qents.map(([t,c])=>`<i style="width:${(c/qtot*100).toFixed(2)}%;background:${qc(t)}" title="${t} ×${c}"></i>`).join("") : "";
  const leg = qents.slice(0,5).map(([t,c])=>`<span><b style="background:${qc(t)}"></b>${t}</span>`).join("");
  const tags=[];
  // all values below come from untrusted GGUF metadata -> esc()
  if(m.finetune) tags.push(`<span class="tag ${/abliter/i.test(m.finetune)?'hot':'cy'}">${esc(m.finetune)}</span>`);
  if(m.vision) tags.push(`<span class="tag">vision</span>`);
  if(m.audio) tags.push(`<span class="tag">audio</span>`);
  if(m.moe) tags.push(`<span class="tag hot">MoE</span>`);
  const aliasLine = (m.aliases&&m.aliases.length?m.aliases.concat([m.label]):[m.label]).map(esc).join(' · ');
  return `<div class="card" data-i="${i}">
    <div class="hd">
      <div><div class="nm">${esc(m.name||m.label)}</div>
        <div class="pa">${aliasLine}</div></div>
      <div class="badges">
        <div class="badge" style="color:${ac(m.arch)};border-color:${ac(m.arch)}">${esc(m.arch)}</div>
        ${srcBadge(m.source)}
      </div>
    </div>
    <div class="body">
      <div class="specrow">
        <div class="spec"><div class="sv">${fmtN(m.params)}</div><div class="sl">params${hq('params')}</div></div>
        <div class="spec"><div class="sv">${m.n_layers??'–'}</div><div class="sl">layers${hq('layers')}</div></div>
        <div class="spec"><div class="sv">${m.d_model??'–'}</div><div class="sl">d_model${hq('d_model')}</div></div>
        <div class="spec"><div class="sv">${m.gqa??'–'}×</div><div class="sl">gqa${hq('heads')}</div></div>
      </div>
      <div class="qbar">${bar}</div>
      <div class="qleg">${leg}</div>
      <div class="tags">
        <span class="tag">${fmtB(m.file_size)}</span>
        <span class="tag">ctx ${fmtN(m.ctx)}</span>
        <span class="tag">vocab ${fmtN(m.vocab)}</span>
        ${tags.join("")}
      </div>
    </div></div>`;
}
function renderFleet(){
  const list = MODELS.map((m,i)=>[m,i]).filter(([m])=>active==="all"||m.arch===active);
  document.getElementById("fleet").innerHTML = list.map(([m,i])=>card(m,i)).join("");
  document.querySelectorAll(".card").forEach(c=>c.onclick=()=>openModal(+c.dataset.i));
}
// renderFleet() is called at the very end, after GLOSSARY/hq are defined

// ---- scatter: params(log x) vs layers(y), r=d_model -----------------------
function scatter(){
  const W=560,H=360,pl=44,pr=16,pt=16,pb=34;
  const xs=MODELS.map(m=>Math.log10(Math.max(1,+m.params)));
  const ys=MODELS.map(m=>+m.n_layers||0);
  const xmin=Math.min(...xs)-0.1,xmax=Math.max(...xs)+0.1;
  const ymax=Math.max(...ys)*1.1;
  const dm=MODELS.map(m=>+m.d_model||1);const dmax=Math.max(...dm);
  const X=v=>pl+(v-xmin)/(xmax-xmin)*(W-pl-pr);
  const Y=v=>H-pb-(v/ymax)*(H-pt-pb);
  let s=`<line class="axis" x1="${pl}" y1="${H-pb}" x2="${W-pr}" y2="${H-pb}"/>
         <line class="axis" x1="${pl}" y1="${pt}" x2="${pl}" y2="${H-pb}"/>`;
  for(let e=8;e<=11;e++){const lv=Math.log10(Math.pow(10,e));if(lv<xmin||lv>xmax)continue;
    s+=`<line class="gridln" x1="${X(lv)}" y1="${pt}" x2="${X(lv)}" y2="${H-pb}"/>
        <text class="axislbl" x="${X(lv)}" y="${H-pb+14}" text-anchor="middle">${fmtN(Math.pow(10,e))}</text>`;}
  [0,20,40,60,80].forEach(v=>{if(v>ymax)return;
    s+=`<line class="gridln" x1="${pl}" y1="${Y(v)}" x2="${W-pr}" y2="${Y(v)}"/>
        <text class="axislbl" x="${pl-6}" y="${Y(v)+3}" text-anchor="end">${v}</text>`;});
  MODELS.forEach((m,i)=>{const r=5+(+m.d_model||0)/dmax*14;
    s+=`<circle cx="${X(xs[i])}" cy="${Y(ys[i])}" r="${r}" fill="${ac(m.arch)}" fill-opacity=".22" stroke="${ac(m.arch)}" stroke-width="1.3" data-i="${i}" class="pt"/>`;});
  s+=`<text class="axislbl" x="${(pl+W-pr)/2}" y="${H-3}" text-anchor="middle">parameter →</text>`;
  document.getElementById("scatter").innerHTML=s;
  document.querySelectorAll("#scatter .pt").forEach(c=>{
    c.style.cursor="pointer";
    c.onmousemove=e=>showTip(e,MODELS[+c.dataset.i]);
    c.onmouseleave=hideTip; c.onclick=()=>openModal(+c.dataset.i);});
}
scatter();

// ---- ratios: GQA bars + FFN ratio dot -------------------------------------
function ratios(){
  const M=MODELS.filter(m=>m.gqa||m.ffn_ratio);
  const W=560, rowH=26, pt=10, pl=120, pr=120;
  const H=pt+M.length*rowH+10;
  const gmax=Math.max(...M.map(m=>m.gqa||1),1);
  const fmax=Math.max(...M.map(m=>m.ffn_ratio||1),1);
  const svg=document.getElementById("ratios"); svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  let s="";
  M.forEach((m,i)=>{const y=pt+i*rowH;
    const gw=(m.gqa||0)/gmax*(W-pl-pr);
    s+=`<text class="axislbl" x="${pl-6}" y="${y+13}" text-anchor="end" fill="#c8d6cf">${esc((m.name||m.label).slice(0,18))}</text>`;
    s+=`<rect x="${pl}" y="${y+4}" width="${gw}" height="13" fill="${ac(m.arch)}" fill-opacity=".5"/>`;
    s+=`<text class="axislbl" x="${pl+gw+5}" y="${y+14}">${m.gqa||'–'}×</text>`;
    if(m.ffn_ratio){const fx=W-pr+ (m.ffn_ratio/fmax)*(pr-30);
      s+=`<circle cx="${fx}" cy="${y+10}" r="4" fill="#38e1d4"/>
          <text class="axislbl" x="${fx+8}" y="${y+13}" fill="#38e1d4">${m.ffn_ratio}</text>`;}
  });
  svg.innerHTML=s;
}
ratios();

// ---- tooltip --------------------------------------------------------------
const tip=document.getElementById("tip");
function showTip(e,m){tip.style.display="block";tip.style.left=(e.clientX+12)+"px";
  tip.style.top=(e.clientY+12)+"px";
  tip.innerHTML=`<b>${esc(m.name)}</b> · ${esc(m.arch)}<br>${fmtN(m.params)}p · ${fmtN(m.n_layers)}L · d${fmtN(m.d_model)}`;}
function hideTip(){tip.style.display="none";}

// ---- glossary: every field explained, with "mehr/weniger?" guidance ------
const CATS=[
 ["arch","Architektur"],
 ["quant","Quantisierung"],
 ["tok","Tokenizer"],
 ["wstat","Gewichts-Statistik (Ebene 4)"],
 ["spec","Spektral-Analyse (Ebene 5)"],
 ["emb","Embedding-Geometrie (Ebene 6)"],
 ["diff","Modell-Diff & Lineage (Ebene 7)"],
 ["compare","Statische Vergleiche (Ebene 8)"],
 ["gguf","GGUF-Format & Grundbegriffe"],
];
const CATS_EN={arch:"Architecture",quant:"Quantization",tok:"Tokenizer",
 wstat:"Weight statistics (level 4)",spec:"Spectral analysis (level 5)",
 emb:"Embedding geometry (level 6)",diff:"Model diff & lineage (level 7)",
 compare:"Static comparisons (level 8)",
 gguf:"GGUF format & fundamentals"};
function catName(k){return (LANG==="en"&&CATS_EN[k])?CATS_EN[k]:((CATS.find(c=>c[0]===k)||[,k])[1]);}

const GLOSSARY={
 // ---- Architektur ----
 arch:{t:"Architektur",d:"Bauplan-Familie (Attention-Typ, Norm-Platzierung, RoPE). Bestimmt, mit welchen Tools das Modell läuft.",b:"Kein Skalar — eine Designwahl, nicht besser/schlechter.",c:"arch"},
 layers:{t:"Layer (block_count)",d:"Tiefe: Anzahl gestapelter Transformer-Blöcke. Jeder transformiert den Residual-Stream weiter.",b:"Mehr = mehr Reasoning-Kapazität, aber mehr Latenz & schwerer trainierbar.",c:"arch"},
 d_model:{t:"d_model (embedding_length)",d:"Breite: Vektor-Dimension pro Token (Residual-Stream).",b:"Breiter = mehr Kapazität pro Token, aber quadratisch wachsende Matrizen.",c:"arch"},
 heads:{t:"Heads (Q / KV)",d:"Query-Heads / Key-Value-Heads. Grouped-Query-Attention: mehrere Q-Heads teilen sich weniger K/V-Heads. Bei Hybrid-/Sliding-Window-Modellen (z.B. Gemma) variieren die KV-Heads pro Layer — angezeigt wird der Maximalwert (globale Layer).",b:"Höheres Q:KV = kleinerer KV-Cache; je nach Modell kann das einen Qualitäts-/Kapazitäts-Tradeoff bedeuten.",c:"arch"},
 head_dim:{t:"head_dim",d:"Dimension pro Attention-Head (key_length, sonst d_model/heads). Meist 64–128.",b:"Konvention, kein Qualitätsmaß.",c:"arch"},
 ffn:{t:"FFN (feed_forward_length)",d:"Innere Weite des MLP-Blocks (Up-Projection).",b:"Größer = mehr Per-Token-Kapazität & mehr Parameter.",c:"arch"},
 ffn_ratio:{t:"FFN-Verhältnis",d:"ffn ÷ d_model — Expansionsfaktor des MLP. Freie Designwahl; reale Modelle hier 3.0–6.0 (klassisches SwiGLU-⅔·4≈2.67 wird oft überschritten).",b:"",c:"arch"},
 ctx:{t:"Kontextlänge",d:"Maximale Sequenzlänge in Tokens, für die das Modell konfiguriert ist.",b:"Größer = mehr Kontext, aber Attention kostet quadratisch.",c:"arch"},
 kv_cache:{t:"KV-Cache @ctx",d:"Attention-Cache-Speicher bei voller Kontextlänge (fp16). GQA/MHA: 2·layers·kv_heads·head_dim·ctx·2B — eine OBERE Schranke. MLA (z.B. glm) cached nur einen komprimierten Latent (viel kleiner); Sliding-Window reduziert es ebenfalls.",b:"Kleiner = billigere lange Kontexte. MLA & SWA sind genau dafür da.",c:"arch"},
 rope_freq_base:{t:"RoPE freq_base",d:"Basisfrequenz (Theta) der Rotary-Position-Embeddings. Sie bestimmt die Frequenzskala der Rotationen; Langkontext-Modelle nutzen oft größere Werte und/oder explizites RoPE-Scaling.",b:"Höher allein garantiert keine gute Langkontext-Qualität — Kontextlänge, Scaling und Finetuning zählen mit.",c:"arch"},
 ctx_extended:{t:"Kontext gestreckt?",d:"Wurde der Kontext per RoPE-Scaling (YaRN/linear) über das Original hinaus verlängert?",b:"Ja = nachträglich gestreckt; Qualität am Ende oft schwächer.",c:"arch"},
 sliding_window:{t:"Sliding-Window",d:"Fenstergröße für lokale (statt globaler) Attention in manchen Layern, z.B. Gemma.",b:"Kleiner = billiger, aber kürzere lokale Reichweite.",c:"arch"},
 moe:{t:"Mixture-of-Experts",d:"Gesamt-Experten / aktiv pro Token. Ein Router wählt je Token wenige Experten-FFNs; nur ein Bruchteil der FFN-Gewichte rechnet mit.",b:"Viele Experten = große Kapazität bei moderatem Rechenaufwand pro Token.",c:"arch"},
 tied_embeddings:{t:"Tied Embeddings",d:"Teilen Eingabe-Embedding und Ausgabe-Projektion (LM-Head) dieselben Gewichte?",b:"Tied = spart Parameter; untied = etwas flexibler.",c:"arch"},
 rms_eps:{t:"RMSNorm-Epsilon",d:"Numerische Stabilitätskonstante im Nenner der RMS-Normalisierung (verhindert Division durch 0).",b:"Reines Implementierungsdetail.",c:"arch"},
 residual_stream:{t:"Residual-Stream",d:"Der durchlaufende Vektor (d_model breit) pro Token, den jeder Block liest und auf den er sein Ergebnis addiert (Residual-/Skip-Verbindung). Attention und FFN schreiben additiv hinein, statt zu überschreiben.",b:"Interpretierbarkeits-Sicht: Information fließt durch diesen Stream, Blöcke 'kommunizieren' über ihn.",c:"arch"},
 attention:{t:"Self-Attention",d:"Jeder Token bildet Query/Key/Value-Vektoren; die Gewichte softmax(Q·Kᵀ ÷ √head_dim) mischen die Value-Vektoren aller (vorhergehenden) Tokens. So bezieht sich jeder Token auf andere.",b:"Der Mechanismus, mit dem Tokens Kontext austauschen. Kosten wachsen quadratisch mit der Sequenzlänge.",c:"arch"},
 gqa:{t:"MHA / MQA / GQA",d:"Attention-Varianten nach KV-Teilung: MHA = jeder Q-Head hat eigene K/V; MQA = alle Q-Heads teilen ein einziges K/V (kleinster Cache); GQA = Mittelweg, Gruppen von Q-Heads teilen sich K/V.",b:"Weniger KV-Heads = kleinerer KV-Cache und weniger Speicherbandbreite bei der Generierung; GQA ist ein häufiger Kompromiss.",c:"arch"},
 rope:{t:"RoPE (Rotary Position Embedding)",d:"Positionsinfo wird kodiert, indem Query/Key-Vektoren paarweise um einen positionsabhängigen Winkel rotiert werden. Relative Positionen erscheinen dadurch im Attention-Skalarprodukt.",b:"Für längere Kontexte braucht es passende Skalierung/Training; freq_base ist nur ein Teil dieser Parametrisierung.",c:"arch"},
 swiglu:{t:"SwiGLU (FFN)",d:"Gattergesteuerter MLP: SwiGLU(x) = (Swish(x·W_gate) ⊙ x·W_up)·W_down — drei Matrizen (gate, up, down) statt zwei. Die innere Weite wird um ⅔ skaliert, damit die Parameterzahl wie beim klassischen 4·d-FFN bleibt.",b:"Heutiger Standard-MLP (Llama/Qwen/Gemma); empirisch besser als ReLU/GELU-FFN.",c:"arch"},
 rmsnorm:{t:"RMSNorm",d:"Root-Mean-Square-Normalisierung: teilt den Vektor durch die Wurzel seines mittleren Quadrats (RMS) und multipliziert mit einem gelernten Gain — KEINE Mittelwert-Subtraktion, kein Bias. Billiger als LayerNorm.",b:"Stabilisiert das Training; rms_eps verhindert die Division durch 0.",c:"arch"},
 block:{t:"Transformer-Block",d:"Die wiederholte Einheit aus Norm → Attention → Norm → FFN, jeweils mit Residual-Verbindung. block_count solcher Blöcke (blk.N) werden gestapelt.",b:"Tiefe = Anzahl Blöcke; jeder verfeinert die Repräsentation im Residual-Stream weiter.",c:"arch"},
 logits:{t:"Logits",d:"Die rohen, un-normalisierten Scores über das gesamte Vokabular am Ausgang (vor Softmax) — ein Wert je möglichem Folge-Token.",b:"Softmax(logits) ergibt die Wahrscheinlichkeiten, aus denen das nächste Token gezogen wird.",c:"arch"},
 lm_head:{t:"LM-Head (output)",d:"Ausgabe-Projektion (output.weight): bildet den finalen d_model-Vektor auf vocab Logits ab. Bei tied embeddings identisch mit der Eingabe-Embedding-Matrix.",b:"Untied = eigene Matrix (mehr Parameter, etwas flexibler); tied = teilt Gewichte mit token_embd.",c:"arch"},

 // ---- Quantisierung ----
 quantization:{t:"Quantisierung (Prinzip)",d:"Gewichte mit weniger Bits als fp16/fp32 speichern (z.B. 4 statt 16). Werte werden blockweise auf wenige Stufen gerundet und über gespeicherte Skalen wieder rekonstruiert.",b:"Weniger Bits = kleiner/schneller, aber Genauigkeitsverlust. Norms/Embeddings bleiben oft hochpräzise.",c:"quant"},
 quant:{t:"Quant-Verteilung",d:"Verteilung der Tensor-Typen in dieser Datei. F32/F16/BF16 = unquantisierte Float-Speicherung (mit unterschiedlicher Float-Präzision), Q*/IQ* = blockweise quantisierte Gewichte.",b:"Zeigt, WELCHE Tensoren hoch-/niedrigpräzise liegen (Norms/Embeddings vs. Gewichte) — nicht nur das Datei-Mittel (bits/weight).",c:"quant"},
 bits_per_weight:{t:"bits/weight",d:"GANZDATEI-Mittel: Daten-Bytes×8 ÷ alle Parameter — inkl. unquantisierter Norms/Embeddings sowie möglicher Vision-/Audio-Tower. Bei großen Text-Modellen liegt es oft nahe am nominalen Quant; bei kleinen oder multimodalen Modellen kann es deutlich höher sein.",b:"Niedriger = kleiner. Kein reines Gewichts-Quant-Maß — Abweichung vom file_type ist NICHT automatisch ein Mislabel.",c:"quant"},
 file_type:{t:"file_type",d:"Vom Ersteller deklarierter Quant-Typ (Enum). Der dominante Tensor-Quant daneben dient als grober Plausibilitäts-Abgleich (nicht als strenger Mislabel-Beweis — Embeddings/Norms weichen bewusst ab).",b:"",c:"quant"},
 kquants:{t:"K-Quants (Q*_K)",d:"llama.cpp-Quant-Familie mit block-/superblockweiser Speicherung: Gewichtsstufen plus Skalen/Minima werden kompakt kodiert; das genaue Layout unterscheidet sich je Quant-Typ. Q2_K…Q6_K geben grob die Bitbreite an.",b:"Meist besserer Größen-/Qualitätskompromiss als ältere Q4_0/Q5_0-Rezepte, aber modell- und rezeptabhängig.",c:"quant"},
 quant_mix:{t:"_S / _M / _L Suffix",d:"Gemischte Quant-Rezepte (small/medium/large). _M hebt einen Teil der sensiblen Tensoren (attn_v, ffn_down) auf Q6_K statt Q4_K; output & Embedding sind ohnehin höher quantisiert. _S lässt attn_v/ffn_down auf Q4_K, _L quantisiert großzügiger.",b:"_M ist meist der beste Kompromiss aus Größe und Qualität.",c:"quant"},
 dequant:{t:"Dequantisierung",d:"Rück-Rechnung quantisierter Bytes in fp32 (Skala × Stufe + Min) pro Block. Die Ebenen 4–7 müssen erst dequantisieren, bevor sie auf den Gewichten rechnen.",b:"Verlustbehaftet rekonstruiert — daher sind die Gewichts-Statistiken 'approx', nicht bit-exakt wie auf HF-Gewichten.",c:"quant"},
 imatrix:{t:"Importance-Matrix (imatrix)",d:"Aus Aktivierungs-Statistiken eines Kalibrier-Textes gewonnen; gewichtet den Quantisierungs-Fehler so, dass wichtige Gewichte präziser bleiben. Vor allem bei niedrigen Bitbreiten (IQ-Quants) hilfreich.",b:"Bessere Qualität bei gleicher Größe; kostet einen einmaligen Kalibrierlauf.",c:"quant"},
 perplexity:{t:"Perplexity",d:"exp(mittlerer negativer Log-Likelihood) auf einem Testtext — wie 'überrascht' das Modell vom nächsten Token ist. Niedriger = besseres Sprachmodell.",b:"Standard-Maß, um Quantisierungs-Qualität zu messen (Δ Perplexity gegen fp16).",c:"quant"},

 // ---- Tokenizer ----
 tokenization:{t:"Tokenisierung",d:"Zerlegen von Text in Tokens (Teilwörter/Bytes), die das Modell als Eingabe-IDs sieht. Bestimmt, wie effizient und in welchen Sprachen das Modell arbeitet.",b:"Komplett statisch prüfbar — alles steckt im Tokenizer, kein Forward-Pass nötig.",c:"tok"},
 vocab:{t:"Vokabular",d:"Anzahl Tokens im Tokenizer.",b:"Größer = mehr Sprachen/Effizienz, aber größere Embedding-Matrix.",c:"tok"},
 merges:{t:"BPE-Merges",d:"Anzahl Byte-Pair-Encoding-Merge-Regeln des Tokenizers.",b:"",c:"tok"},
 bpe:{t:"BPE (Byte-Pair Encoding)",d:"Startet bei Einzel-Bytes/Zeichen und fügt iterativ das häufigste benachbarte Paar zu einem neuen Token zusammen (merges = die gelernten Regeln). GPT-2/Qwen nutzen byte-level BPE.",b:"Häufige Wörter = ein Token, seltene = mehrere; bestimmt die Vokab-Effizienz.",c:"tok"},
 sentencepiece:{t:"SentencePiece",d:"Sprach-unabhängiger Tokenizer (Kudo/Google): behandelt Text als rohen Unicode-Strom inkl. Leerzeichen (als ▁ markiert), trainiert per BPE oder Unigram. Llama/Gemma-Linie.",b:"Braucht keine Vor-Segmentierung und ist verlustfrei umkehrbar.",c:"tok"},
 byte_fallback:{t:"Byte-Fallback",d:"Unbekannte Zeichen werden in ihre UTF-8-Bytes (als <0xXX>-Tokens) zerlegt, statt auf UNK zu fallen — so ist jeder Text kodierbar.",b:"Verhindert echte out-of-vocabulary-Fehler; kostet bei seltenen Zeichen mehr Tokens.",c:"tok"},
 special:{t:"Special Tokens",d:"Steuer-Token-IDs: BOS (Begin), EOS (End), PAD (Padding), UNK (Unknown). add_bos/eos = werden sie automatisch angehängt.",b:"BOS/EOS müssen korrekt gesetzt sein, sonst startet/stoppt die Generierung falsch. add_bos doppelt = häufiger Prompt-Fehler.",c:"tok"},
 special_tok:{t:"Funktionale Special-Tokens",d:"Intentionale Steuer-Tokens (<|im_start|>, [INST], <s> …). CONTROL/USER_DEFINED-Typ oder <|…|>-Muster.",b:"Keine Glitches — werden aktiv genutzt (Chat-Format, Tools).",c:"tok"},
 reserved:{t:"Reservierte / unbenutzte Slots",d:"Platzhalter-Tokens (<unusedN>, [PADn], UNUSED-Typ), die im Vokab existieren, aber meist nicht echt trainiert sind. Statischer Kandidat für 'Glitch'-Verhalten.",b:"Echte Glitch-Tokens (untrainiert) brauchen Embedding-Normen (Ebene 6) — das hier ist nur ein Namens-/Typ-Heuristik-Kandidat.",c:"tok"},
 token_type:{t:"Token-Typen",d:"GGUF-Klassifikation jedes Tokens: NORMAL, CONTROL, USER_DEFINED, UNUSED, BYTE, UNKNOWN.",b:"Die Verteilung trennt echtes Vokabular (NORMAL) von Steuer-/UNUSED-Tokens — Basis für Special- und Glitch-Erkennung.",c:"tok"},
 scripts:{t:"Skript-Abdeckung",d:"Welche Schriftsysteme das Vokabular abdeckt (Latein, CJK, Kyrillisch …), gezählt über eine Stichprobe der Tokens (gpt2-Byte-Kodierung wird rück-decodiert).",b:"Mehr Skripte = mehrsprachiger; verteilt aber das Vokab-Budget.",c:"tok"},
 overlap:{t:"Vokab-Overlap (Jaccard)",d:"Anteil gemeinsamer Tokens zweier Tokenizer: |A∩B| ÷ |A∪B|. 1.0 = identisches Vokabular.",b:"Hoher Overlap = gemeinsame Tokenizer-Herkunft (z.B. ein Finetune des anderen).",c:"tok"},
 chat_template:{t:"Chat-Template",d:"Jinja-Vorlage im GGUF (tokenizer.chat_template), die Chat-Nachrichten (System/User/Assistant) ins exakte Prompt-Format des Modells gießt — inklusive der Special-Tokens (<|im_start|> etc.).",b:"Bestimmt, ob ein Chat-Modell korrekt antwortet; ein falsches Template = degradierte Qualität. Nur Instruct/Chat-Modelle haben eins.",c:"tok"},
 tokens:{t:"Token-Liste (ggml.tokens)",d:"Das vollständige Vokabular als Array (tokenizer.ggml.tokens): Position = Token-ID, Wert = der String. Daneben liegt tokenizer.ggml.token_type (der Typ je Token).",b:"Quelle für Vokabulargröße, Token-Typen, Special-/Reserved-Erkennung und den Overlap-Vergleich.",c:"tok"},
 tok_model:{t:"Tokenizer-Modell",d:"GGUF-Feld tokenizer.ggml.model: welche Tokenizer-Familie das Vokabular trägt — 'gpt2'/'bpe' (byte-level BPE), 'llama' (SentencePiece). Bestimmt, wie ggml die Token-Liste interpretiert.",b:"Grobe Herkunfts-Angabe; die Details liefern bpe / sentencepiece.",c:"tok"},
 pretok:{t:"Pre-Tokenizer",d:"Regex-Schema (tokenizer.ggml.pre), das Text VOR dem BPE-Merging in Stücke schneidet (Ziffern, Leerzeichen, Satzzeichen) — z.B. 'llama-bpe', 'qwen2', 'gpt-2'.",b:"Muss exakt zum Training passen; ein falsches Pre-Schema = subtil andere Token-IDs trotz gleichem Vokabular.",c:"tok"},

 // ---- Gewichts-Statistik (Ebene 4) ----
 std:{t:"Standardabweichung",d:"Streuung der Gewichtswerte eines Tensors. Norm-Layer (RMSNorm) haben hohe std; Projektionen niedrige.",b:"Sehr kleine std = evtl. toter/wenig genutzter Layer.",c:"wstat"},
 l2:{t:"L2-Norm",d:"Euklidische Norm aller Gewichte des Tensors (Gesamt-'Energie'), entspricht der Frobenius-Norm der Matrix.",b:"",c:"wstat"},
 frobenius:{t:"Frobenius-Norm",d:"‖W‖_F = √(Σ aller Elemente²) — die 'Länge' der Gewichtsmatrix, betrachtet als flacher Vektor.",b:"Grundgröße für L2, Stable-Rank und das Diff-delta.",c:"wstat"},
 sparsity:{t:"Sparsity",d:"Anteil der Gewichte nahe 0 (|w|<1e-6).",b:"Hoch = viele effektiv inaktive Gewichte (auf dequantisierten Werten nur grob).",c:"wstat"},
 kurtosis:{t:"Kurtosis (exzess)",d:"Wie schwer die Verteilungs-Tails sind (0 = normalverteilt). Hoch = wenige große Ausreißer-Gewichte.",b:"Hohe Kurtosis in Projektionen kann auf Outlier-Features hindeuten.",c:"wstat"},
 outlier:{t:"Outlier-Channels",d:"Anteil Eingangs-Kanäle (Spalten) mit L2-Norm > 6× Median — Vorboten der 'massive activations'.",b:"Konzentrierte Outlier-Channels sind für Quantisierung kritisch.",c:"wstat"},
 massive_activations:{t:"Massive Activations",d:"Wenige versteckte Dimensionen, die im Forward-Pass extrem große Aktivierungswerte annehmen (Sun et al. 2024). Statisch nur indirekt sichtbar — als Outlier-Channels in den Gewichten.",b:"Kritisch für Quantisierung: ein einzelner Kanal kann die Skala dominieren.",c:"wstat"},

 // ---- Spektral-Analyse (Ebene 5) ----
 alpha:{t:"alpha (Heavy-Tail)",d:"Geschätzter Power-Law-Exponent des Gewichts-Spektrums (hier: Hill-Schätzer). WeightWatcher nutzt alpha als datenfreie Qualitätsdiagnose; grob gelten 2–6 als plausibler Bereich für gut korrelierte Layer, >6 als Warnsignal.",b:"Kein Ground-Truth-Qualitätslabel. Diese Implementierung ist einfacher als WeightWatcher und auf dequantisierten GGUF-Gewichten nur eine grobe Indikation.",c:"spec"},
 heavy_tail:{t:"Heavy-Tailed Self-Regularization",d:"Theorie/Empirie (Martin & Mahoney): trainierte Gewichtsmatrizen zeigen häufig eine heavy-tailed Eigenwert-/Singulärwert-Verteilung. alpha fasst die Steilheit dieses Tails zusammen.",b:"Kann ohne Trainings-/Testdaten mit Qualität korrelieren, ersetzt aber keine Evaluation.",c:"spec"},
 svd:{t:"SVD (Singulärwertzerlegung)",d:"W = U·Σ·Vᵀ zerlegt eine Matrix in orthogonale Richtungen (U,V) und ihre Stärken (Singulärwerte σ in Σ). Grundlage von alpha, Stable-Rank und effektivem Rang.",b:"Auf CPU teuer (≈O(n³)) — daher bei großen Modellen im Hintergrund.",c:"spec"},
 singular_values:{t:"Singulärwerte (σ)",d:"Die σ aus der SVD: wie viel 'Energie' die Matrix entlang jeder Hauptrichtung trägt. Ihre Verteilung verrät Rang, Heavy-Tail (alpha) und effektive Dimensionalität.",b:"",c:"spec"},
 stable_rank:{t:"Stable Rank",d:"‖W‖_F² ÷ σ_max² ∈ [1, Rang]. 1 = rang-1-artig (ein dominanter Modus), hoch = viele Moden genutzt.",b:"",c:"spec"},
 spectral_entropy:{t:"Effektiver Rang",d:"exp(Shannon-Entropie der normierten Singulärwert-Verteilung) — wie viele Dimensionen das Gewicht effektiv nutzt.",b:"Niedrig relativ zur Matrixgröße = Kapazität ungenutzt.",c:"spec"},
 weightwatcher:{t:"WeightWatcher",d:"Open-Source-Diagnosemethode (Martin & Mahoney), die Spektralmetriken wie alpha aus Gewichtsmatrizen berechnet — ohne Trainings- oder Testdaten.",b:"modelstrata implementiert nur eine kleine, einfache SVD/Hill-Variante für statische Exploration; nicht die vollständige WeightWatcher-Pipeline.",c:"spec"},

 // ---- Embedding-Geometrie (Ebene 6) ----
 embedding:{t:"Token-Embedding-Matrix",d:"token_embd (vocab × d_model): die gelernte Vektor-Darstellung jedes Tokens — die Repräsentation, mit der ein Token in den Residual-Stream eintritt.",b:"Statisch analysierbar: Geometrie, Normen, Glitch-Hinweise (Ebene 6).",c:"emb"},
 pca:{t:"PCA (Hauptkomponenten)",d:"Projiziert die hochdimensionalen Embeddings auf die 2 Richtungen größter Varianz, um sie als 2D-Streudiagramm sichtbar zu machen. evr = erklärte Varianz je Achse.",b:"Reine Visualisierung — nahe Punkte ≈ ähnliche Embeddings.",c:"emb"},
 anisotropy:{t:"Anisotropie",d:"Mittlerer Cosine zufälliger Token-Paare im Embedding-Raum. ~0 = gut verteilt; hoch = Embeddings drängen sich in einem schmalen Kegel.",b:"Sehr hohe Anisotropie kann Repräsentations-Degeneration anzeigen.",c:"emb"},
 glitch:{t:"Glitch-Token",d:"Tokens mit anomaler (oft ~0) Embedding-Norm; solche Tokens gelten häufig als Hinweis auf untertrainierte oder selten gesehene Token-Embeddings und können ungewöhnliches Verhalten auslösen (SolidGoldMagikarp, Rumbelow & Watkins 2023). Hinweise = niedrigste Normen.",b:"Ein stärkerer Hinweis kommt aus Embedding-Normen (Ebene 6); Name/Typ allein ist kein Beweis.",c:"emb"},
 cosine_sim:{t:"Cosine-Similarity",d:"Winkel-Ähnlichkeit zweier Vektoren = (a·b) ÷ (‖a‖·‖b‖) ∈ [−1, 1]. 1 = gleiche Richtung, 0 = orthogonal. Misst Richtung statt Länge.",b:"Basis für Anisotropie (Embeddings) und das Diff-cosine (Gewichte).",c:"emb"},

 // ---- Modell-Diff & Lineage (Ebene 7) ----
 delta:{t:"delta (Diff)",d:"Relative Frobenius-Differenz ‖W_b−W_a‖ ÷ ‖W_a‖ zweier Modelle. 0 = identisch.",b:"Hoch = der Tensor wurde stark verändert (Finetune/Abliteration).",c:"diff"},
 cosine:{t:"cosine (Diff)",d:"Richtungs-Ähnlichkeit zweier Gewichts-Tensoren. 1 = gleiche Richtung, <1 = gedreht.",b:"Robuster gegen reine Skalierung/Quant-Rauschen als delta.",c:"diff"},
 finetune:{t:"Finetuning",d:"Weitertrainieren eines vortrainierten Basis-Modells auf engeren Daten (Domäne, Stil, Instruktionen). Verändert Gewichte meist nur leicht — sichtbar als niedriges delta / hohes cosine.",b:"Erbt Tokenizer & Architektur des Basis-Modells (Vokab-Overlap 1.0 ist ein starker Hinweis).",c:"diff"},
 abliteration:{t:"Abliteration",d:"Gezieltes Abschwächen einer identifizierten Refusal-Richtung in den Gewichten/Aktivierungen (Arditi et al. 2024): per Orthogonalisierung wird ihr Beitrag reduziert — ohne Re-Training. Kann Refusal-Verhalten und damit Sicherheitsverhalten verändern.",b:"Im Diff kann das als lokalisierte Änderung in attention/ffn-Projektionen erscheinen; ethisch heikel.",c:"diff"},
 distillation:{t:"Distillation",d:"Ein kleineres 'Student'-Modell lernt, die Ausgaben (Logits/Antworten) eines größeren 'Teacher' nachzuahmen. z.B. DeepSeek-R1-Distill = Qwen/Llama, trainiert auf R1-Ausgaben.",b:"Komprimiert Fähigkeiten in kleinere Modelle; die Architektur bleibt die des Students.",c:"diff"},
 lora:{t:"LoRA (Low-Rank Adaptation)",d:"Friert die Basis-Gewichte ein und lernt nur eine niedrig-rangige Korrektur ΔW = B·A (wenige Parameter). Beim Mergen wird ΔW zu den Basis-Gewichten addiert.",b:"Effizientes Finetuning; gemergte LoRAs zeigen im Diff oft lokalisierte, niedrig-rangige Änderungen.",c:"diff"},
 base_instruct:{t:"Base- vs. Instruct-Modell",d:"Base = nur auf Next-Token vortrainiert (roher Text-Vervollständiger). Instruct/Chat = danach per SFT + RLHF/DPO nachtrainiert, um Anweisungen zu folgen.",b:"Gleiche Architektur, anderes Verhalten; Chat-Modelle bringen Chat-Template & Steuer-Tokens mit.",c:"diff"},

 // ---- Statische Vergleiche (Ebene 8) ----
 jaccard:{t:"Jaccard-Ähnlichkeit",d:"Mengen-Ähnlichkeit: |A∩B| ÷ |A∪B|. 1.0 = gleiche Menge, 0.0 = keine gemeinsamen Elemente. modelstrata nutzt sie z.B. für Vokabular-, Tensornamen-, Rollen-, Quant- und Template-Marker-Overlap.",b:"Gut für 'wie viel teilen zwei Modelle?', aber ohne Reihenfolge/Training zu kennen.",c:"compare"},
 health_score:{t:"Health-Score",d:"Heuristische Zusammenfassung der statischen Checks pro Modell. Fehler, Warnungen und Hinweise aus Config↔Tensor-, Tokenizer↔Embedding-, Chat-Template-, Quant-, Metadata-, MoE- und Multimodal-Diagnosen senken den Score.",b:"Kein Qualitäts- oder Safety-Label; zeigt, wo sich ein menschlicher Review lohnt.",c:"compare"},
 lineage_score:{t:"Lineage-Score",d:"Heuristische Nähe zweier Modelle aus statischen Signalen: Architekturgleichheit, Tensor-/Rollen-Overlap, Tokenizer-Overlap, Prompt-Kompatibilität und Quant-Profil. Hohe Werte sprechen für gleiche Familie, Base/Finetune oder eng verwandte Releases.",b:"Beweist keine Abstammung; Modellkarten und Hashes bleiben die Quelle der Wahrheit.",c:"compare"},
 same_id_ratio:{t:"same-id Ratio",d:"Anteil gemeinsamer Tokens, die in zwei Tokenizern dieselbe Token-ID haben. Gemeinsamer Token-String allein reicht nicht: andere IDs ändern Prompt- und Embedding-Verhalten.",b:"Hoch = Tokenizer vermutlich kompatibel; niedrig = Vorsicht bei Adapter-/Prompt-Übertragung.",c:"compare"},
 same_shape_ratio:{t:"same-shape Ratio",d:"Anteil gemeinsam benannter Tensoren mit identischer Shape. Gleicher Name mit anderer Shape deutet auf andere Breite, andere Expertengröße oder inkompatible Varianten hin.",b:"Wichtig für Diffs, LoRA-Merges und Gewichtsübertragung.",c:"compare"},
 marker_jaccard:{t:"Marker-Jaccard",d:"Jaccard-Overlap der im Chat-Template erkannten Rollen-/Formatmarker, z.B. system/user/assistant, [INST] oder ChatML-Marker. Misst Prompt-Format-Nähe ohne Inferenz.",b:"Heuristik: Templates können logisch unterschiedlich sein, obwohl Marker ähnlich aussehen.",c:"compare"},
 config_tensor_checks:{t:"Config↔Tensor-Invarianten",d:"Statische Plausibilitätschecks zwischen Metadaten und Tensor-Verzeichnis: Layer-Lücken, erwartete Rollen pro Block und Shape-Regeln für Attention/FFN/Output.",b:"Findet kaputte, teilkonvertierte oder ungewöhnlich gemappte Modelle; manche Spezialarchitekturen brauchen menschliche Einordnung.",c:"compare"},
 tokenizer_embedding_checks:{t:"Tokenizer↔Embedding-Konsistenz",d:"Vergleicht Vokabulargröße und Special-Token-IDs mit Embedding-/Output-Head-Shapes. Ein Vokab größer als die Embedding-Zeilen oder Special-IDs außerhalb des Bereichs ist ein echter Konsistenzfehler.",b:"Sehr nützlich nach Tokenizer-Edits, Konvertierungen oder Adapter-Merges.",c:"compare"},
 chat_lint:{t:"Chat-Template-Lint",d:"Statische Warnungen für Prompt-Templates: fehlende User-/Assistant-Marker, mögliche doppelte BOS-Erzeugung, schwache Tool-Marker oder unbekannte Template-Familie.",b:"Lint ist kein Ausführungstest; er zeigt typische Prompt-Fehlerquellen.",c:"compare"},
 quant_diagnostics:{t:"Quant-Diagnostik",d:"Prüft, ob sensible Rollen wie Embeddings, Output, Norms, attn_v oder ffn_down aggressiv quantisiert sind oder global ungewöhnliche Quant-Muster auftreten.",b:"Niedrige Bits sind nicht automatisch schlecht; die Rolle des Tensors zählt.",c:"compare"},
 metadata_audit:{t:"Metadata-Audit",d:"Prüft, ob wichtige Metadaten wie Name, Architektur, Lizenz, Sprachen oder Herkunft fehlen oder schwach sind.",b:"Fehlende Metadaten ändern nicht die Gewichte, erschweren aber Reproduzierbarkeit und sichere Nutzung.",c:"compare"},
 moe_diagnostics:{t:"MoE-Diagnostik",d:"Prüft Mixture-of-Experts-Tensoren: Router/Gate, Expert-IDs, geteilte Experten und Lücken in Expertennummern.",b:"Hilft, kaputte MoE-Konvertierungen oder nur teilweise gemappte Experten zu erkennen.",c:"compare"},
 multimodal_diagnostics:{t:"Multimodal-Diagnostik",d:"Prüft Vision-/Audio-Tower, Projektoren und multimodale Flags gegen die gefundenen Tensorrollen.",b:"Ein Textmodell ohne Tower ist normal; ein multimodales Modell ohne passende Tower-Tensoren ist verdächtig.",c:"compare"},

 // ---- GGUF-Format & Grundbegriffe ----
 gguf:{t:"GGUF-Format",d:"Dateiformat von llama.cpp/ggml für ein Modell in EINER Datei: Header + Metadaten-Key-Values + alle Tensoren. Nachfolger des alten GGML-Dateiformats.",b:"Selbstbeschreibend: Architektur & Tokenizer stecken in den Metadaten (kein externes Config nötig).",c:"gguf"},
 ggml:{t:"GGML",d:"Die C-Tensor-Bibliothek hinter llama.cpp (Georgi Gerganov) — das 'GG' in GGUF/GGML. Stellt die Quant-Typen (Q4_K …) und die CPU/GPU-Kernels bereit.",b:"",c:"gguf"},
 tensor:{t:"Tensor",d:"Mehrdimensionales Zahlen-Array — die eigentlichen Gewichte (z.B. blk.0.attn_q.weight als [out,in]-Matrix). Das Tensor-Verzeichnis listet Name, Form, Quant-Typ & Offset.",b:"",c:"gguf"},
 params:{t:"Parameter",d:"Gesamtzahl Gewichte = Summe aller Tensor-Elemente.",b:"Mehr Kapazität, aber nicht automatisch besser — Training zählt.",c:"gguf"},
 file_size:{t:"Dateigröße",d:"Belegter Speicher on-disk.",b:"",c:"gguf"},
 data_bytes:{t:"Daten-Bytes",d:"Reine Tensor-Datenmenge = Datei minus Header & Verzeichnis.",b:"",c:"gguf"},
 bias_count:{t:"Bias-Tensoren",d:"Anzahl 1-D Bias-Tensoren (in der Heatmap ausgeblendet). Qwen z.B. hat Q/K/V-Bias.",b:"",c:"gguf"},
 gguf_version:{t:"GGUF-Version",d:"Version des GGUF-Containerformats.",b:"",c:"gguf"},
 alignment:{t:"Alignment",d:"Byte-Ausrichtung der Tensor-Daten (default 32).",b:"",c:"gguf"},
 magic:{t:"Magic-Bytes",d:"Die ersten 4 Bytes jeder GGUF-Datei: ASCII 'GGUF' (0x47 47 55 46) — die Signatur, an der ein Parser das Format erkennt.",b:"",c:"gguf"},
 little_endian:{t:"Little-Endian",d:"Byte-Reihenfolge mit niederwertigstem Byte zuerst. GGUF speichert Ganzzahlen (Versions-, Tensor-, KV-Anzahl) little-endian; im Hex-Dump wirken sie daher 'rückwärts'.",b:"",c:"gguf"},
 forward_pass:{t:"Forward-Pass",d:"Ein Durchlauf Eingabe → Ausgabe durch das Modell (das eigentliche 'Rechnen-Lassen'). Dieses Projekt macht bewusst KEINEN Forward-Pass — alles wird statisch aus Datei & Gewichten abgeleitet.",b:"Statisch = kein GPU/RAM für Inferenz, keine Daten, reproduzierbar.",c:"gguf"},
 inference:{t:"Inferenz",d:"Das Anwenden eines fertig trainierten Modells zur Vorhersage (Token für Token generieren). Gegenstück zum Training.",b:"Black-box-Verhalten (Phase 1) braucht Inferenz; die statischen Ebenen hier nicht.",c:"gguf"},
};
const GLOSSARY_EN={
 // ---- Architecture ----
 arch:{t:"Architecture",d:"Blueprint family (attention type, norm placement, RoPE). Determines which tools run the model.",b:"Not a scalar — a design choice, not better/worse.",c:"arch"},
 layers:{t:"Layers (block_count)",d:"Depth: number of stacked transformer blocks. Each one transforms the residual stream further.",b:"More = more reasoning capacity, but more latency & harder to train.",c:"arch"},
 d_model:{t:"d_model (embedding_length)",d:"Width: vector dimension per token (the residual stream).",b:"Wider = more capacity per token, but quadratically larger matrices.",c:"arch"},
 heads:{t:"Heads (Q / KV)",d:"Query heads / key-value heads. Grouped-query attention: several Q heads share fewer K/V heads. In hybrid/sliding-window models (e.g. Gemma) the KV heads vary per layer — the max (global layers) is shown.",b:"Higher Q:KV = smaller KV cache; depending on the model, this can be a quality/capacity tradeoff.",c:"arch"},
 head_dim:{t:"head_dim",d:"Dimension per attention head (key_length, else d_model/heads). Usually 64–128.",b:"Convention, not a quality measure.",c:"arch"},
 ffn:{t:"FFN (feed_forward_length)",d:"Inner width of the MLP block (up-projection).",b:"Larger = more per-token capacity & more parameters.",c:"arch"},
 ffn_ratio:{t:"FFN ratio",d:"ffn ÷ d_model — MLP expansion factor. Free design choice; real models here 3.0–6.0 (classic SwiGLU ⅔·4≈2.67 is often exceeded).",b:"",c:"arch"},
 ctx:{t:"Context length",d:"Maximum sequence length in tokens the model is configured for.",b:"Larger = more context, but attention costs scale quadratically.",c:"arch"},
 kv_cache:{t:"KV cache @ctx",d:"Attention-cache memory at full context (fp16). GQA/MHA: 2·layers·kv_heads·head_dim·ctx·2B — an UPPER bound. MLA (e.g. glm) caches only a compressed latent (much smaller); sliding window reduces it too.",b:"Smaller = cheaper long contexts. MLA & SWA exist exactly for this.",c:"arch"},
 rope_freq_base:{t:"RoPE freq_base",d:"Base frequency (theta) of the rotary position embeddings. It sets the rotation frequency scale; long-context models often use larger values and/or explicit RoPE scaling.",b:"Higher alone does not guarantee good long-context quality — context length, scaling and finetuning matter too.",c:"arch"},
 ctx_extended:{t:"Context extended?",d:"Was context stretched beyond the original via RoPE scaling (YaRN/linear)?",b:"Yes = stretched after the fact; quality at the far end is often weaker.",c:"arch"},
 sliding_window:{t:"Sliding window",d:"Window size for local (instead of global) attention in some layers, e.g. Gemma.",b:"Smaller = cheaper, but shorter local reach.",c:"arch"},
 moe:{t:"Mixture-of-Experts",d:"Total experts / active per token. A router picks a few expert FFNs per token; only a fraction of the FFN weights compute.",b:"Many experts = large capacity at moderate compute per token.",c:"arch"},
 tied_embeddings:{t:"Tied embeddings",d:"Do the input embedding and output projection (LM head) share the same weights?",b:"Tied = saves parameters; untied = a bit more flexible.",c:"arch"},
 rms_eps:{t:"RMSNorm epsilon",d:"Numerical stability constant in the denominator of RMS normalization (prevents division by 0).",b:"Pure implementation detail.",c:"arch"},
 residual_stream:{t:"Residual stream",d:"The running vector (d_model wide) per token that every block reads from and adds its result back into (residual/skip connection). Attention and FFN write additively rather than overwriting.",b:"Interpretability view: information flows through this stream; blocks 'communicate' via it.",c:"arch"},
 attention:{t:"Self-attention",d:"Each token forms query/key/value vectors; the weights softmax(Q·Kᵀ ÷ √head_dim) mix the value vectors of all (preceding) tokens. That is how a token attends to others.",b:"The mechanism by which tokens exchange context. Cost grows quadratically with sequence length.",c:"arch"},
 gqa:{t:"MHA / MQA / GQA",d:"Attention variants by KV sharing: MHA = each Q head has its own K/V; MQA = all Q heads share a single K/V (smallest cache); GQA = middle ground, groups of Q heads share K/V.",b:"Fewer KV heads = smaller KV cache and lower memory bandwidth during generation; GQA is a common compromise.",c:"arch"},
 rope:{t:"RoPE (rotary position embedding)",d:"Position is encoded by rotating query/key vectors pairwise by a position-dependent angle. Relative position information then appears in the attention dot product.",b:"Longer contexts require suitable scaling/training; freq_base is only one part of that parametrization.",c:"arch"},
 swiglu:{t:"SwiGLU (FFN)",d:"Gated MLP: SwiGLU(x) = (Swish(x·W_gate) ⊙ x·W_up)·W_down — three matrices (gate, up, down) instead of two. The inner width is scaled by ⅔ to keep the parameter count of a classic 4·d FFN.",b:"Today's standard MLP (Llama/Qwen/Gemma); empirically better than ReLU/GELU FFN.",c:"arch"},
 rmsnorm:{t:"RMSNorm",d:"Root-mean-square normalization: divides the vector by the root of its mean square (RMS) and multiplies by a learned gain — NO mean subtraction, no bias. Cheaper than LayerNorm.",b:"Stabilizes training; rms_eps prevents division by 0.",c:"arch"},
 block:{t:"Transformer block",d:"The repeated unit of norm → attention → norm → FFN, each with a residual connection. block_count such blocks (blk.N) are stacked.",b:"Depth = number of blocks; each refines the representation in the residual stream.",c:"arch"},
 logits:{t:"Logits",d:"The raw, un-normalized scores over the whole vocabulary at the output (before softmax) — one value per possible next token.",b:"Softmax(logits) gives the probabilities the next token is sampled from.",c:"arch"},
 lm_head:{t:"LM head (output)",d:"Output projection (output.weight): maps the final d_model vector to vocab logits. With tied embeddings it is identical to the input embedding matrix.",b:"Untied = its own matrix (more parameters, a bit more flexible); tied = shares weights with token_embd.",c:"arch"},

 // ---- Quantization ----
 quantization:{t:"Quantization (principle)",d:"Storing weights with fewer bits than fp16/fp32 (e.g. 4 instead of 16). Values are rounded block-wise to a few levels and reconstructed via stored scales.",b:"Fewer bits = smaller/faster, but accuracy loss. Norms/embeddings often stay high-precision.",c:"quant"},
 quant:{t:"Quant distribution",d:"Distribution of tensor types in this file. F32/F16/BF16 = unquantized float storage (with different float precision); Q*/IQ* = block-quantized weights.",b:"Shows WHICH tensors are high/low precision (norms/embeddings vs. weights) — not just the file average (bits/weight).",c:"quant"},
 bits_per_weight:{t:"bits/weight",d:"WHOLE-FILE average: data bytes×8 ÷ all parameters — including unquantized norms/embeddings and possible vision/audio towers. For large text models it is often near the nominal quant; for small or multimodal models it can be much higher.",b:"Lower = smaller. Not a pure weight-quant measure — deviation from file_type is NOT automatically a mislabel.",c:"quant"},
 file_type:{t:"file_type",d:"Quant type declared by the author (enum). The dominant tensor quant next to it is a rough plausibility check (not strict mislabel proof — embeddings/norms deviate on purpose).",b:"",c:"quant"},
 kquants:{t:"K-quants (Q*_K)",d:"llama.cpp quant family with block/superblock storage: weight levels plus scales/mins are encoded compactly; the exact layout differs by quant type. Q2_K…Q6_K roughly indicate bit width.",b:"Often a better size/quality tradeoff than older Q4_0/Q5_0 recipes, but model- and recipe-dependent.",c:"quant"},
 quant_mix:{t:"_S / _M / _L suffix",d:"Mixed quant recipes (small/medium/large). _M bumps some sensitive tensors (attn_v, ffn_down) to Q6_K instead of Q4_K; output & embedding are higher-precision anyway. _S keeps attn_v/ffn_down at Q4_K, _L quantizes more generously.",b:"_M is usually the best size/quality compromise.",c:"quant"},
 dequant:{t:"Dequantization",d:"Reconstructing fp32 from quantized bytes (scale × level + min) per block. Levels 4–7 must dequantize before computing on the weights.",b:"Lossily reconstructed — so the weight statistics are 'approx', not bit-exact like on HF weights.",c:"quant"},
 imatrix:{t:"Importance matrix (imatrix)",d:"Derived from activation statistics on a calibration text; weights the quantization error so important weights stay more precise. Especially helpful at low bit widths (IQ quants).",b:"Better quality at the same size; costs a one-off calibration run.",c:"quant"},
 perplexity:{t:"Perplexity",d:"exp(mean negative log-likelihood) on a test text — how 'surprised' the model is by the next token. Lower = better language model.",b:"Standard metric to measure quantization quality (Δ perplexity vs fp16).",c:"quant"},

 // ---- Tokenizer ----
 tokenization:{t:"Tokenization",d:"Splitting text into tokens (sub-words/bytes) the model sees as input IDs. Determines how efficiently and in which languages the model works.",b:"Fully checkable statically — it all lives in the tokenizer, no forward pass needed.",c:"tok"},
 vocab:{t:"Vocabulary",d:"Number of tokens in the tokenizer.",b:"Larger = more languages/efficiency, but a bigger embedding matrix.",c:"tok"},
 merges:{t:"BPE merges",d:"Number of byte-pair-encoding merge rules of the tokenizer.",b:"",c:"tok"},
 bpe:{t:"BPE (byte-pair encoding)",d:"Starts from single bytes/characters and iteratively merges the most frequent adjacent pair into a new token (merges = the learned rules). GPT-2/Qwen use byte-level BPE.",b:"Common words = one token, rare ones = several; sets vocab efficiency.",c:"tok"},
 sentencepiece:{t:"SentencePiece",d:"Language-independent tokenizer (Kudo/Google): treats text as a raw unicode stream incl. spaces (marked as ▁), trained via BPE or Unigram. Llama/Gemma lineage.",b:"Needs no pre-segmentation and is losslessly reversible.",c:"tok"},
 byte_fallback:{t:"Byte fallback",d:"Unknown characters are decomposed into their UTF-8 bytes (as <0xXX> tokens) instead of falling back to UNK — so any text is encodable.",b:"Prevents real out-of-vocabulary errors; costs more tokens for rare characters.",c:"tok"},
 special:{t:"Special tokens",d:"Control token IDs: BOS (begin), EOS (end), PAD (padding), UNK (unknown). add_bos/eos = whether they are appended automatically.",b:"BOS/EOS must be set correctly or generation starts/stops wrong. add_bos applied twice = a common prompt bug.",c:"tok"},
 special_tok:{t:"Functional special tokens",d:"Intentional control tokens (<|im_start|>, [INST], <s> …). CONTROL/USER_DEFINED type or <|…|> pattern.",b:"Normally not glitches — actively used by chat formats, tools or templates.",c:"tok"},
 reserved:{t:"Reserved / unused slots",d:"Placeholder tokens (<unusedN>, [PADn], UNUSED type) that exist in the vocab but may be reserved, unused or weakly trained. Static candidate for unusual token behavior.",b:"A stronger signal needs embedding norms (level 6); this is only a name/type heuristic.",c:"tok"},
 token_type:{t:"Token types",d:"GGUF classification of each token: NORMAL, CONTROL, USER_DEFINED, UNUSED, BYTE, UNKNOWN.",b:"The distribution separates real vocabulary (NORMAL) from control/UNUSED tokens — the basis for special & glitch detection.",c:"tok"},
 scripts:{t:"Script coverage",d:"Which writing systems the vocabulary covers (Latin, CJK, Cyrillic …), counted over a sample of tokens (gpt2 byte encoding is decoded back).",b:"More scripts = more multilingual; but spreads the vocab budget.",c:"tok"},
 overlap:{t:"Vocab overlap (Jaccard)",d:"Share of common tokens of two tokenizers: |A∩B| ÷ |A∪B|. 1.0 = identical vocabulary.",b:"High overlap = shared tokenizer lineage (e.g. one is a finetune of the other).",c:"tok"},
 chat_template:{t:"Chat template",d:"Jinja template in the GGUF (tokenizer.chat_template) that renders chat messages (system/user/assistant) into the model's exact prompt format — including the special tokens (<|im_start|> etc.).",b:"Decides whether a chat model answers correctly; a wrong template = degraded quality. Only instruct/chat models ship one.",c:"tok"},
 tokens:{t:"Token list (ggml.tokens)",d:"The full vocabulary as an array (tokenizer.ggml.tokens): position = token ID, value = the string. Alongside it sits tokenizer.ggml.token_type (the type per token).",b:"Source for vocab size, token types, special/reserved detection and the overlap comparison.",c:"tok"},
 tok_model:{t:"Tokenizer model",d:"GGUF field tokenizer.ggml.model: which tokenizer family carries the vocab — 'gpt2'/'bpe' (byte-level BPE), 'llama' (SentencePiece). Determines how ggml interprets the token list.",b:"A rough provenance hint; bpe / sentencepiece give the details.",c:"tok"},
 pretok:{t:"Pre-tokenizer",d:"Regex scheme (tokenizer.ggml.pre) that splits text into chunks BEFORE BPE merging (digits, whitespace, punctuation) — e.g. 'llama-bpe', 'qwen2', 'gpt-2'.",b:"Must match training exactly; a wrong pre-scheme = subtly different token IDs despite the same vocabulary.",c:"tok"},

 // ---- Weight statistics (level 4) ----
 std:{t:"Standard deviation",d:"Spread of a tensor's weight values. Norm layers (RMSNorm) have high std; projections low.",b:"Very small std = possibly a dead/under-used layer.",c:"wstat"},
 l2:{t:"L2 norm",d:"Euclidean norm of all the tensor's weights (total 'energy'), equal to the matrix's Frobenius norm.",b:"",c:"wstat"},
 frobenius:{t:"Frobenius norm",d:"‖W‖_F = √(Σ of all elements²) — the 'length' of the weight matrix viewed as a flat vector.",b:"Base quantity for L2, stable rank and the diff delta.",c:"wstat"},
 sparsity:{t:"Sparsity",d:"Fraction of weights near 0 (|w|<1e-6).",b:"High = many effectively inactive weights (only rough on dequantized values).",c:"wstat"},
 kurtosis:{t:"Kurtosis (excess)",d:"How heavy the distribution tails are (0 = normal). High = a few large outlier weights.",b:"High kurtosis in projections can indicate outlier features.",c:"wstat"},
 outlier:{t:"Outlier channels",d:"Fraction of input channels (columns) with L2 norm > 6× median — precursors of 'massive activations'.",b:"Concentrated outlier channels are critical for quantization.",c:"wstat"},
 massive_activations:{t:"Massive activations",d:"A few hidden dimensions that take on extremely large activation values during the forward pass (Sun et al. 2024). Statically only visible indirectly — as outlier channels in the weights.",b:"Critical for quantization: a single channel can dominate the scale.",c:"wstat"},

 // ---- Spectral analysis (level 5) ----
 alpha:{t:"alpha (heavy-tail)",d:"Estimated power-law exponent of the weight spectrum (here: Hill estimator). WeightWatcher uses alpha as a data-free quality diagnostic; roughly 2–6 is a plausible range for well-correlated layers, while >6 is a warning sign.",b:"Not a ground-truth quality label. This implementation is simpler than WeightWatcher and only a rough indication on dequantized GGUF weights.",c:"spec"},
 heavy_tail:{t:"Heavy-tailed self-regularization",d:"Theory/empirics (Martin & Mahoney): trained weight matrices often show heavy-tailed eigenvalue/singular-value distributions. alpha summarizes the slope of that tail.",b:"Can correlate with quality without training/test data, but does not replace evaluation.",c:"spec"},
 svd:{t:"SVD (singular value decomposition)",d:"W = U·Σ·Vᵀ decomposes a matrix into orthogonal directions (U,V) and their strengths (singular values σ in Σ). Basis for alpha, stable rank and effective rank.",b:"Expensive on CPU (≈O(n³)) — hence run in the background for large models.",c:"spec"},
 singular_values:{t:"Singular values (σ)",d:"The σ from the SVD: how much 'energy' the matrix carries along each principal direction. Their distribution reveals rank, heavy tail (alpha) and effective dimensionality.",b:"",c:"spec"},
 stable_rank:{t:"Stable rank",d:"‖W‖_F² ÷ σ_max² ∈ [1, rank]. 1 = rank-1-like (one dominant mode), high = many modes used.",b:"",c:"spec"},
 spectral_entropy:{t:"Effective rank",d:"exp(Shannon entropy of the normalized singular-value distribution) — how many dimensions the weight effectively uses.",b:"Low relative to matrix size = capacity unused.",c:"spec"},
 weightwatcher:{t:"WeightWatcher",d:"Open-source diagnostic method (Martin & Mahoney) that computes spectral metrics such as alpha from weight matrices — without training or test data.",b:"modelstrata implements only a small, simple SVD/Hill variant for static exploration; not the full WeightWatcher pipeline.",c:"spec"},

 // ---- Embedding geometry (level 6) ----
 embedding:{t:"Token embedding matrix",d:"token_embd (vocab × d_model): the learned vector representation of each token — the representation a token enters the residual stream with.",b:"Statically analyzable: geometry, norms, glitch indicators (level 6).",c:"emb"},
 pca:{t:"PCA (principal components)",d:"Projects the high-dimensional embeddings onto the 2 directions of greatest variance to show them as a 2D scatter. evr = explained variance per axis.",b:"Pure visualization — nearby points ≈ similar embeddings.",c:"emb"},
 anisotropy:{t:"Anisotropy",d:"Mean cosine of random token pairs in embedding space. ~0 = well spread; high = embeddings crowd into a narrow cone.",b:"Very high anisotropy can indicate representation degeneration.",c:"emb"},
 glitch:{t:"Glitch token",d:"Tokens with anomalous (often ~0) embedding norms; such tokens are often suspected to have under-trained or rarely seen embeddings and can trigger unusual behavior (SolidGoldMagikarp, Rumbelow & Watkins 2023). Indicators = lowest norms.",b:"A stronger signal comes from embedding norms (level 6); name/type alone is not proof.",c:"emb"},
 cosine_sim:{t:"Cosine similarity",d:"Angular similarity of two vectors = (a·b) ÷ (‖a‖·‖b‖) ∈ [−1, 1]. 1 = same direction, 0 = orthogonal. Measures direction, not length.",b:"Basis for anisotropy (embeddings) and the diff cosine (weights).",c:"emb"},

 // ---- Model diff & lineage (level 7) ----
 delta:{t:"delta (diff)",d:"Relative Frobenius difference ‖W_b−W_a‖ ÷ ‖W_a‖ of two models. 0 = identical.",b:"High = the tensor was strongly changed (finetune/abliteration).",c:"diff"},
 cosine:{t:"cosine (diff)",d:"Directional similarity of two weight tensors. 1 = same direction, <1 = rotated.",b:"More robust against pure scaling/quant noise than delta.",c:"diff"},
 finetune:{t:"Fine-tuning",d:"Continued training of a pretrained base model on narrower data (domain, style, instructions). Usually changes weights only slightly — visible as low delta / high cosine.",b:"Inherits the base model's tokenizer & architecture (vocab overlap 1.0 is a strong hint).",c:"diff"},
 abliteration:{t:"Abliteration",d:"Targeted weakening of an identified refusal direction in weights/activations (Arditi et al. 2024): orthogonalization reduces its contribution without re-training. Can change refusal and safety behavior.",b:"May appear in the diff as localized changes in attention/ffn projections; ethically sensitive.",c:"diff"},
 distillation:{t:"Distillation",d:"A smaller 'student' model learns to mimic the outputs (logits/answers) of a larger 'teacher'. e.g. DeepSeek-R1-Distill = Qwen/Llama trained on R1 outputs.",b:"Compresses capabilities into smaller models; the architecture stays the student's.",c:"diff"},
 lora:{t:"LoRA (low-rank adaptation)",d:"Freezes the base weights and learns only a low-rank correction ΔW = B·A (few parameters). On merging, ΔW is added to the base weights.",b:"Efficient fine-tuning; merged LoRAs often show localized, low-rank changes in the diff.",c:"diff"},
 base_instruct:{t:"Base vs. instruct model",d:"Base = only next-token pretrained (raw text completer). Instruct/chat = then post-trained via SFT + RLHF/DPO to follow instructions.",b:"Same architecture, different behavior; chat models ship a chat template & control tokens.",c:"diff"},

 // ---- Static comparisons (level 8) ----
 jaccard:{t:"Jaccard similarity",d:"Set similarity: |A∩B| ÷ |A∪B|. 1.0 = same set, 0.0 = no shared elements. modelstrata uses it for vocabulary, tensor names, roles, quant profiles and template-marker overlap.",b:"Good for 'how much do two models share?', but it does not know ordering or training history.",c:"compare"},
 health_score:{t:"Health score",d:"Heuristic summary of static checks per model. Errors, warnings and infos from config↔tensor, tokenizer↔embedding, chat-template, quant, metadata, MoE and multimodal diagnostics lower the score.",b:"Not a quality or safety label; it points to where human review is worthwhile.",c:"compare"},
 lineage_score:{t:"Lineage score",d:"Heuristic closeness of two models from static signals: architecture match, tensor/role overlap, tokenizer overlap, prompt compatibility and quant profile. High values suggest same family, base/finetune or closely related releases.",b:"Does not prove ancestry; model cards and hashes remain the source of truth.",c:"compare"},
 same_id_ratio:{t:"same-id ratio",d:"Share of common tokens that have the same token ID in two tokenizers. A shared token string alone is not enough: different IDs change prompt and embedding behavior.",b:"High = tokenizer likely compatible; low = be careful with adapters or prompt transfer.",c:"compare"},
 same_shape_ratio:{t:"same-shape ratio",d:"Share of commonly named tensors with identical shape. Same name but different shape points to different width, expert size or incompatible variants.",b:"Important for diffs, LoRA merges and weight transfer.",c:"compare"},
 marker_jaccard:{t:"Marker Jaccard",d:"Jaccard overlap of role/format markers detected in chat templates, e.g. system/user/assistant, [INST] or ChatML markers. Measures prompt-format closeness without inference.",b:"Heuristic: templates can behave differently even when markers look similar.",c:"compare"},
 config_tensor_checks:{t:"Config↔tensor invariants",d:"Static plausibility checks between metadata and tensor directory: layer gaps, expected roles per block and shape rules for attention/FFN/output.",b:"Finds broken, partially converted or unusually mapped models; some special architectures need human review.",c:"compare"},
 tokenizer_embedding_checks:{t:"Tokenizer↔embedding consistency",d:"Compares vocabulary size and special-token IDs with embedding/output-head shapes. A vocab larger than embedding rows or special IDs outside the row range is a real consistency error.",b:"Very useful after tokenizer edits, conversions or adapter merges.",c:"compare"},
 chat_lint:{t:"Chat-template lint",d:"Static warnings for prompt templates: missing user/assistant markers, possible double BOS insertion, weak tool markers or unknown template family.",b:"Lint is not an execution test; it surfaces common prompt failure modes.",c:"compare"},
 quant_diagnostics:{t:"Quant diagnostics",d:"Checks whether sensitive roles such as embeddings, output, norms, attn_v or ffn_down are aggressively quantized, and whether global quant patterns look unusual.",b:"Low bits are not automatically bad; tensor role matters.",c:"compare"},
 metadata_audit:{t:"Metadata audit",d:"Checks whether important metadata such as name, architecture, license, languages or provenance is missing or weak.",b:"Missing metadata does not change weights, but hurts reproducibility and safe use.",c:"compare"},
 moe_diagnostics:{t:"MoE diagnostics",d:"Checks Mixture-of-Experts tensors: router/gate, expert IDs, shared experts and gaps in expert numbering.",b:"Helps spot broken MoE conversions or only partially mapped experts.",c:"compare"},
 multimodal_diagnostics:{t:"Multimodal diagnostics",d:"Checks vision/audio towers, projectors and multimodal flags against the tensor roles found in the model.",b:"A text model without towers is normal; a multimodal model without matching tower tensors is suspicious.",c:"compare"},

 // ---- GGUF format & fundamentals ----
 gguf:{t:"GGUF format",d:"llama.cpp/ggml file format for a model in ONE file: header + metadata key-values + all tensors. Successor of the old GGML file format.",b:"Self-describing: architecture & tokenizer live in the metadata (no external config needed).",c:"gguf"},
 ggml:{t:"GGML",d:"The C tensor library behind llama.cpp (Georgi Gerganov) — the 'GG' in GGUF/GGML. Provides the quant types (Q4_K …) and the CPU/GPU kernels.",b:"",c:"gguf"},
 tensor:{t:"Tensor",d:"Multi-dimensional number array — the actual weights (e.g. blk.0.attn_q.weight as an [out,in] matrix). The tensor directory lists name, shape, quant type & offset.",b:"",c:"gguf"},
 params:{t:"Parameters",d:"Total weights = sum of all tensor elements.",b:"More capacity, but not automatically better — training matters.",c:"gguf"},
 file_size:{t:"File size",d:"Bytes occupied on disk.",b:"",c:"gguf"},
 data_bytes:{t:"Data bytes",d:"Pure tensor data = file minus header & directory.",b:"",c:"gguf"},
 bias_count:{t:"Bias tensors",d:"Number of 1-D bias tensors (hidden in the heatmap). Qwen e.g. has Q/K/V bias.",b:"",c:"gguf"},
 gguf_version:{t:"GGUF version",d:"Version of the GGUF container format.",b:"",c:"gguf"},
 alignment:{t:"Alignment",d:"Byte alignment of the tensor data (default 32).",b:"",c:"gguf"},
 magic:{t:"Magic bytes",d:"The first 4 bytes of every GGUF file: ASCII 'GGUF' (0x47 47 55 46) — the signature a parser recognizes the format by.",b:"",c:"gguf"},
 little_endian:{t:"Little-endian",d:"Byte order with the least-significant byte first. GGUF stores integers (version, tensor, KV counts) little-endian; they look 'reversed' in the hex dump.",b:"",c:"gguf"},
 forward_pass:{t:"Forward pass",d:"One run input → output through the model (the actual 'running' of it). This project deliberately does NO forward pass — everything is derived statically from file & weights.",b:"Static = no GPU/RAM for inference, no data, reproducible.",c:"gguf"},
 inference:{t:"Inference",d:"Applying a trained model to predict (generating token by token). The counterpart to training.",b:"Black-box behavior (phase 1) needs inference; the static levels here do not.",c:"gguf"},
};
function G(k){return (LANG==="en" && typeof GLOSSARY_EN!=="undefined" && GLOSSARY_EN[k]) ? GLOSSARY_EN[k] : GLOSSARY[k];}
function help(key){const g=G(key);return g?` data-help="${(g.d+(g.b?' — '+g.b:'')).replace(/"/g,'&quot;')}"`:"";}
function hq(key){return GLOSSARY[key]?`<span class="hq"${help(key)}>?</span>`:"";}

// ---- global help-tooltip wiring (any [data-help]) -------------------------
document.addEventListener("mousemove",e=>{
  const t=e.target.closest("[data-help]");
  if(t){tip.style.display="block";tip.style.whiteSpace="normal";tip.style.maxWidth="320px";
    tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY+14)+"px";
    tip.innerHTML=`<span class="help">${t.getAttribute("data-help")}</span>`;}
});
document.addEventListener("mouseout",e=>{if(e.target.closest&&e.target.closest("[data-help]"))hideTip();});

// ---- value renderers ------------------------------------------------------
function kvrow(key,label,val,cls,raw){
  // values are HTML-escaped by default (GGUF metadata is untrusted); pass
  // raw=true only for values we build as markup ourselves.
  const v=(val===null||val===undefined||val==="")?"–":(raw?val:esc(val));
  return `<div class="kv"><span class="k">${tr(label)}${hq(key)}</span><span class="v ${cls||''}">${v}</span></div>`;
}
function fmtMeta(v){
  if(v&&typeof v==="object"&&v._array)
    return `<span class="arr">⟨array · ${v.len} × type${v.elem_type}⟩</span> ${esc(JSON.stringify(v.sample).slice(0,120))}…`;
  const s=String(v);
  if(s.length>90) return `<details><summary>⟨string · ${s.length} ${LANG==="en"?"chars":"Zeichen"}⟩</summary><pre>${esc(s)}</pre></details>`;
  return esc(s);
}

// ---- modal ----------------------------------------------------------------
// ---- known special-token catalog (purpose / meaning) ---------------------
// Exact-match first, then patterns. Best-effort annotations for common model
// families in this fleet (Qwen/ChatML, Llama-2/3, Mistral, Gemma, Phi,
// DeepSeek-R1, Qwen-Coder FIM). Used to annotate functional special tokens.
const SPECIAL_TOKEN_INFO = {
 "<|im_start|>":{de:"ChatML: <b>öffnet</b> eine Chat-Nachricht; direkt danach folgt die Rolle (system/user/assistant). Qwen & OpenAI-ChatML.",en:"ChatML: <b>opens</b> a chat message; the role (system/user/assistant) follows immediately. Qwen & OpenAI ChatML."},
 "<|im_end|>":{de:"ChatML: <b>schließt</b> eine Chat-Nachricht. Dient meist auch als EOS im Chat-Format.",en:"ChatML: <b>closes</b> a chat message. Usually doubles as the chat-format EOS."},
 "<|endoftext|>":{de:"GPT-2/Qwen: Dokument-Trenner und Ende-des-Texts. Oft zugleich EOS/PAD.",en:"GPT-2/Qwen: document separator and end-of-text. Often also EOS/PAD."},
 "<s>":{de:"BOS — Anfang einer Sequenz (SentencePiece/Llama-Linie).",en:"BOS — beginning of a sequence (SentencePiece/Llama lineage)."},
 "</s>":{de:"EOS — Ende einer Sequenz; das Stopp-Signal der Generierung.",en:"EOS — end of a sequence; the generation stop signal."},
 "<unk>":{de:"Unknown — Platzhalter für nicht kodierbare Eingaben (ohne Byte-Fallback).",en:"Unknown — placeholder for inputs that cannot be encoded (without byte fallback)."},
 "<pad>":{de:"Padding — füllt kürzere Sequenzen im Batch auf gleiche Länge auf.",en:"Padding — pads shorter sequences in a batch to equal length."},
 "<bos>":{de:"BOS — explizites Sequenz-Anfangs-Token.",en:"BOS — explicit beginning-of-sequence token."},
 "<eos>":{de:"EOS — explizites Sequenz-Ende-Token.",en:"EOS — explicit end-of-sequence token."},
 "<|begin_of_text|>":{de:"Llama-3 BOS — Anfang des gesamten Prompts.",en:"Llama-3 BOS — start of the whole prompt."},
 "<|end_of_text|>":{de:"Llama-3 Ende-des-Texts / EOS.",en:"Llama-3 end-of-text / EOS."},
 "<|eot_id|>":{de:"Llama-3 <b>End-of-Turn</b> — beendet EINE Nachricht in einem Mehr-Runden-Chat.",en:"Llama-3 <b>end-of-turn</b> — ends ONE message in a multi-turn chat."},
 "<|start_header_id|>":{de:"Llama-3: öffnet den Rollen-Header einer Nachricht (z.B. …user…).",en:"Llama-3: opens a message's role header (e.g. …user…)."},
 "<|end_header_id|>":{de:"Llama-3: schließt den Rollen-Header; danach folgt der Nachrichtentext.",en:"Llama-3: closes the role header; the message body follows."},
 "<|python_tag|>":{de:"Llama-3: markiert einen Tool-/Code-Aufruf (Function Calling).",en:"Llama-3: marks a tool/code invocation (function calling)."},
 "[INST]":{de:"Llama-2/Mistral: <b>öffnet</b> die Nutzer-Instruktion.",en:"Llama-2/Mistral: <b>opens</b> the user instruction."},
 "[/INST]":{de:"Llama-2/Mistral: <b>schließt</b> die Nutzer-Instruktion; danach antwortet das Modell.",en:"Llama-2/Mistral: <b>closes</b> the user instruction; the model replies after."},
 "<<SYS>>":{de:"Llama-2: öffnet den System-Prompt (innerhalb des ersten [INST]).",en:"Llama-2: opens the system prompt (inside the first [INST])."},
 "<</SYS>>":{de:"Llama-2: schließt den System-Prompt.",en:"Llama-2: closes the system prompt."},
 "<start_of_turn>":{de:"Gemma: öffnet einen Gesprächszug (gefolgt von user/model).",en:"Gemma: opens a conversation turn (followed by user/model)."},
 "<end_of_turn>":{de:"Gemma: schließt einen Gesprächszug.",en:"Gemma: closes a conversation turn."},
 "<|system|>":{de:"Rollen-Tag: leitet die System-Anweisung ein (Zephyr/Phi-Linie).",en:"Role tag: introduces the system instruction (Zephyr/Phi lineage)."},
 "<|user|>":{de:"Rollen-Tag: leitet die Nutzer-Nachricht ein.",en:"Role tag: introduces the user message."},
 "<|assistant|>":{de:"Rollen-Tag: leitet die Modell-Antwort ein.",en:"Role tag: introduces the assistant reply."},
 "<|end|>":{de:"Phi-3: Ende einer Nachricht.",en:"Phi-3: end of a message."},
 "<|fim_prefix|>":{de:"Fill-in-the-Middle: markiert den Code <b>vor</b> der Lücke (Infilling).",en:"Fill-in-the-middle: marks the code <b>before</b> the gap (infilling)."},
 "<|fim_suffix|>":{de:"Fill-in-the-Middle: markiert den Code <b>nach</b> der Lücke.",en:"Fill-in-the-middle: marks the code <b>after</b> the gap."},
 "<|fim_middle|>":{de:"Fill-in-the-Middle: hier <b>schreibt</b> das Modell die fehlende Mitte.",en:"Fill-in-the-middle: where the model <b>writes</b> the missing middle."},
 "<|fim_pad|>":{de:"Fill-in-the-Middle: Padding-Token für das FIM-Format.",en:"Fill-in-the-middle: padding token for the FIM format."},
 "<|repo_name|>":{de:"Qwen-Coder: trennt Repository-Namen im Repo-Level-Training.",en:"Qwen-Coder: separates repository names in repo-level training."},
 "<|file_sep|>":{de:"Qwen-Coder: trennt einzelne Dateien im Repo-Level-Training.",en:"Qwen-Coder: separates individual files in repo-level training."},
 "<|think|>":{de:"Reasoning-/Denk-Block (modell-spezifische Variante von &lt;think&gt;).",en:"Reasoning/think block (model-specific variant of &lt;think&gt;)."},
 "<think>":{de:"DeepSeek-R1: öffnet einen Reasoning-/Denk-Block.",en:"DeepSeek-R1: opens a reasoning/think block."},
 "</think>":{de:"DeepSeek-R1: schließt den Reasoning-Block; danach folgt die finale Antwort.",en:"DeepSeek-R1: closes the reasoning block; the final answer follows."},
 "<tool_call>":{de:"Function-Calling: öffnet einen Werkzeug-Aufruf (Hermes/Qwen).",en:"Function calling: opens a tool call (Hermes/Qwen)."},
 "</tool_call>":{de:"Function-Calling: schließt den Werkzeug-Aufruf.",en:"Function calling: closes the tool call."},
 "<｜begin▁of▁sentence｜>":{de:"DeepSeek BOS — Anfang der Sequenz (Voll-Breite-Zeichen).",en:"DeepSeek BOS — start of sequence (full-width characters)."},
 "<｜end▁of▁sentence｜>":{de:"DeepSeek EOS — Ende der Sequenz.",en:"DeepSeek EOS — end of sequence."},
 "<｜User｜>":{de:"DeepSeek Rollen-Tag: Nutzer-Nachricht.",en:"DeepSeek role tag: user message."},
 "<｜Assistant｜>":{de:"DeepSeek Rollen-Tag: Modell-Antwort.",en:"DeepSeek role tag: assistant reply."},
 "<|User|>":{de:"DeepSeek Rollen-Tag (ASCII-Pipe-Variante): Nutzer-Nachricht.",en:"DeepSeek role tag (ASCII-pipe variant): user message."},
 "<|Assistant|>":{de:"DeepSeek Rollen-Tag (ASCII-Pipe-Variante): Modell-Antwort.",en:"DeepSeek role tag (ASCII-pipe variant): assistant reply."},
 "<|EOT|>":{de:"DeepSeek-Coder End-of-Turn — Stopp-Signal im Instruktions-Format.",en:"DeepSeek-Coder end-of-turn — stop signal in the instruction format."},
 "<｜fim▁begin｜>":{de:"DeepSeek-Coder FIM (Voll-Breite): Code <b>vor</b> der Lücke.",en:"DeepSeek-Coder FIM (full-width): code <b>before</b> the gap."},
 "<｜fim▁hole｜>":{de:"DeepSeek-Coder FIM: die zu füllende <b>Lücke</b>.",en:"DeepSeek-Coder FIM: the <b>gap</b> to fill."},
 "<｜fim▁end｜>":{de:"DeepSeek-Coder FIM: Code <b>nach</b> der Lücke.",en:"DeepSeek-Coder FIM: code <b>after</b> the gap."},
 // --- Qwen2-VL / Qwen2.5 vision + visual grounding (inherited by Qwen2.5-Coder, Qwen3, R1-Distill-Qwen) ---
 "<|vision_start|>":{de:"Qwen2-VL: <b>öffnet</b> einen eingebetteten Bild-/Video-Token-Block im Prompt.",en:"Qwen2-VL: <b>opens</b> an embedded image/video token block in the prompt."},
 "<|vision_end|>":{de:"Qwen2-VL: <b>schließt</b> den Bild-/Video-Token-Block.",en:"Qwen2-VL: <b>closes</b> the image/video token block."},
 "<|vision_pad|>":{de:"Qwen2-VL: Padding innerhalb des Vision-Token-Blocks.",en:"Qwen2-VL: padding inside the vision token block."},
 "<|image_pad|>":{de:"Qwen2-VL: Platzhalter je Bild-Patch (wird durch Bild-Features ersetzt).",en:"Qwen2-VL: per-image-patch placeholder (replaced by image features)."},
 "<|video_pad|>":{de:"Qwen2-VL: Platzhalter je Video-Frame-Patch.",en:"Qwen2-VL: per-video-frame-patch placeholder."},
 "<|object_ref_start|>":{de:"Qwen2-VL Grounding: <b>öffnet</b> eine Objekt-Referenz (Text, der ein Bildobjekt benennt).",en:"Qwen2-VL grounding: <b>opens</b> an object reference (text naming an image object)."},
 "<|object_ref_end|>":{de:"Qwen2-VL Grounding: <b>schließt</b> die Objekt-Referenz.",en:"Qwen2-VL grounding: <b>closes</b> the object reference."},
 "<|box_start|>":{de:"Qwen2-VL Grounding: <b>öffnet</b> Bounding-Box-Koordinaten.",en:"Qwen2-VL grounding: <b>opens</b> bounding-box coordinates."},
 "<|box_end|>":{de:"Qwen2-VL Grounding: <b>schließt</b> die Bounding-Box-Koordinaten.",en:"Qwen2-VL grounding: <b>closes</b> the bounding-box coordinates."},
 "<|quad_start|>":{de:"Qwen2-VL Grounding: <b>öffnet</b> Viereck-/Polygon-Koordinaten (gedrehte Boxen).",en:"Qwen2-VL grounding: <b>opens</b> quadrilateral/polygon coordinates (rotated boxes)."},
 "<|quad_end|>":{de:"Qwen2-VL Grounding: <b>schließt</b> die Viereck-Koordinaten.",en:"Qwen2-VL grounding: <b>closes</b> the quadrilateral coordinates."},
 "<tool_response>":{de:"Function-Calling: <b>öffnet</b> das zurückgegebene Tool-Ergebnis (Qwen/Hermes/GLM).",en:"Function calling: <b>opens</b> the returned tool result (Qwen/Hermes/GLM)."},
 "</tool_response>":{de:"Function-Calling: <b>schließt</b> das Tool-Ergebnis.",en:"Function calling: <b>closes</b> the tool result."},
 // --- Llama-3.1 extras ---
 "<|eom_id|>":{de:"Llama-3.1 End-of-Message — beendet eine Nachricht, wenn weitere Schritte folgen (z.B. Tool-Aufruf), anders als <|eot_id|>.",en:"Llama-3.1 end-of-message — ends a message when more steps follow (e.g. a tool call), unlike <|eot_id|>."},
 "<|finetune_right_pad_id|>":{de:"Llama-3.1 Padding-Token fürs Finetuning (rechtsseitiges Auffüllen von Batches).",en:"Llama-3.1 padding token for finetuning (right-padding batches)."},
 // --- Mistral v3 (tekken) / Pixtral / Magistral / Voxtral ---
 "[TOOL_CALLS]":{de:"Mistral: leitet die vom Modell erzeugten Tool-/Function-Calls ein.",en:"Mistral: introduces the model-generated tool/function calls."},
 "[AVAILABLE_TOOLS]":{de:"Mistral: <b>öffnet</b> die Liste verfügbarer Tools (JSON-Schemas).",en:"Mistral: <b>opens</b> the list of available tools (JSON schemas)."},
 "[/AVAILABLE_TOOLS]":{de:"Mistral: <b>schließt</b> die Tool-Liste.",en:"Mistral: <b>closes</b> the tool list."},
 "[TOOL_RESULTS]":{de:"Mistral: <b>öffnet</b> die zurückgegebenen Tool-Ergebnisse.",en:"Mistral: <b>opens</b> the returned tool results."},
 "[/TOOL_RESULTS]":{de:"Mistral: <b>schließt</b> die Tool-Ergebnisse.",en:"Mistral: <b>closes</b> the tool results."},
 "[TOOL_CONTENT]":{de:"Mistral: Inhalt eines einzelnen Tool-Ergebnisses.",en:"Mistral: content of a single tool result."},
 "[ARGS]":{de:"Mistral: Argumente eines Function-Calls.",en:"Mistral: arguments of a function call."},
 "[CALL_ID]":{de:"Mistral: ID, die einen Tool-Aufruf mit seinem Ergebnis verknüpft.",en:"Mistral: id linking a tool call to its result."},
 "[SYSTEM_PROMPT]":{de:"Mistral v3: <b>öffnet</b> den System-Prompt.",en:"Mistral v3: <b>opens</b> the system prompt."},
 "[/SYSTEM_PROMPT]":{de:"Mistral v3: <b>schließt</b> den System-Prompt.",en:"Mistral v3: <b>closes</b> the system prompt."},
 "[THINK]":{de:"Magistral (Mistral-Reasoning): <b>öffnet</b> den Denk-/Reasoning-Block.",en:"Magistral (Mistral reasoning): <b>opens</b> the reasoning block."},
 "[/THINK]":{de:"Magistral (Mistral-Reasoning): <b>schließt</b> den Reasoning-Block.",en:"Magistral (Mistral reasoning): <b>closes</b> the reasoning block."},
 "[PREFIX]":{de:"Mistral-Code FIM: Code <b>vor</b> der Lücke.",en:"Mistral code FIM: code <b>before</b> the gap."},
 "[MIDDLE]":{de:"Mistral-Code FIM: die zu füllende <b>Mitte</b>.",en:"Mistral code FIM: the <b>middle</b> to fill."},
 "[SUFFIX]":{de:"Mistral-Code FIM: Code <b>nach</b> der Lücke.",en:"Mistral code FIM: code <b>after</b> the gap."},
 "[IMG]":{de:"Pixtral (Mistral-Vision): Platzhalter für einen Bild-Patch.",en:"Pixtral (Mistral vision): placeholder for an image patch."},
 "[IMG_BREAK]":{de:"Pixtral: Zeilenumbruch zwischen Bild-Patch-Reihen.",en:"Pixtral: row break between image-patch rows."},
 "[IMG_END]":{de:"Pixtral: Ende des Bildblocks.",en:"Pixtral: end of the image block."},
 "[AUDIO]":{de:"Voxtral (Mistral-Audio): Platzhalter für Audio-Eingabe.",en:"Voxtral (Mistral audio): placeholder for audio input."},
 "[BEGIN_AUDIO]":{de:"Voxtral: Beginn des Audio-Blocks.",en:"Voxtral: start of the audio block."},
 // --- GLM-4 / GLM-4V ---
 "[gMASK]":{de:"GLM: General-Mask — startet die autoregressive Blank-Infilling-Generierung (GLM-Pretraining-Erbe).",en:"GLM: general mask — starts autoregressive blank-infilling generation (GLM pretraining heritage)."},
 "[sMASK]":{de:"GLM: Sentence-Mask — maskiert eine ganze Span/Satz.",en:"GLM: sentence mask — masks a whole span/sentence."},
 "[MASK]":{de:"GLM: Token-Maske (Blank-Infilling).",en:"GLM: token mask (blank infilling)."},
 "<sop>":{de:"GLM: Start-of-Piece — Beginn der Generierung nach [gMASK].",en:"GLM: start-of-piece — generation begins here after [gMASK]."},
 "<eop>":{de:"GLM: End-of-Piece.",en:"GLM: end-of-piece."},
 "<|observation|>":{de:"GLM-4: Rollen-Tag für Tool-/Funktions-Ergebnisse (neben user/assistant/system).",en:"GLM-4: role tag for tool/function results (alongside user/assistant/system)."},
 "<arg_key>":{de:"GLM-4 Function-Calling: <b>öffnet</b> einen Argument-Schlüssel.",en:"GLM-4 function calling: <b>opens</b> an argument key."},
 "</arg_key>":{de:"GLM-4 Function-Calling: schließt den Argument-Schlüssel.",en:"GLM-4 function calling: closes the argument key."},
 "<arg_value>":{de:"GLM-4 Function-Calling: <b>öffnet</b> einen Argument-Wert.",en:"GLM-4 function calling: <b>opens</b> an argument value."},
 "</arg_value>":{de:"GLM-4 Function-Calling: schließt den Argument-Wert.",en:"GLM-4 function calling: closes the argument value."},
 "<|code_prefix|>":{de:"GLM-Code FIM: Code <b>vor</b> der Lücke.",en:"GLM code FIM: code <b>before</b> the gap."},
 "<|code_middle|>":{de:"GLM-Code FIM: die zu füllende <b>Mitte</b>.",en:"GLM code FIM: the <b>middle</b> to fill."},
 "<|code_suffix|>":{de:"GLM-Code FIM: Code <b>nach</b> der Lücke.",en:"GLM code FIM: code <b>after</b> the gap."},
 "<|begin_of_image|>":{de:"GLM-4V: <b>öffnet</b> eine Bild-Eingabe.",en:"GLM-4V: <b>opens</b> an image input."},
 "<|end_of_image|>":{de:"GLM-4V: schließt die Bild-Eingabe.",en:"GLM-4V: closes the image input."},
 "<|begin_of_video|>":{de:"GLM-4V: <b>öffnet</b> eine Video-Eingabe.",en:"GLM-4V: <b>opens</b> a video input."},
 "<|end_of_video|>":{de:"GLM-4V: schließt die Video-Eingabe.",en:"GLM-4V: closes the video input."},
 "<|begin_of_audio|>":{de:"GLM-4V: <b>öffnet</b> eine Audio-Eingabe.",en:"GLM-4V: <b>opens</b> an audio input."},
 "<|end_of_audio|>":{de:"GLM-4V: schließt die Audio-Eingabe.",en:"GLM-4V: closes the audio input."},
 "<|begin_of_transcription|>":{de:"GLM-4V: <b>öffnet</b> einen Transkriptions-Block.",en:"GLM-4V: <b>opens</b> a transcription block."},
 "<|end_of_transcription|>":{de:"GLM-4V: schließt den Transkriptions-Block.",en:"GLM-4V: closes the transcription block."},
 "<|begin_of_box|>":{de:"GLM-4V Grounding: <b>öffnet</b> Bounding-Box-Koordinaten.",en:"GLM-4V grounding: <b>opens</b> bounding-box coordinates."},
 "<|end_of_box|>":{de:"GLM-4V Grounding: schließt die Bounding-Box-Koordinaten.",en:"GLM-4V grounding: closes the bounding-box coordinates."},
 "/nothink":{de:"GLM-4.x: schaltet den Reasoning-/Think-Modus für diesen Turn ab.",en:"GLM-4.x: disables reasoning/think mode for this turn."},
 // --- Gemma multimodal ---
 "<mask>":{de:"Gemma: Masken-Token.",en:"Gemma: mask token."},
 "[multimodal]":{de:"Gemma: Marker für eine multimodale Eingabe.",en:"Gemma: marker for a multimodal input."},
 "<|image|>":{de:"Multimodal: Platzhalter für eine Bild-Eingabe.",en:"Multimodal: placeholder for an image input."},
 "<|video|>":{de:"Multimodal: Platzhalter für eine Video-Eingabe.",en:"Multimodal: placeholder for a video input."},
 "<|audio|>":{de:"Multimodal: Platzhalter für eine Audio-Eingabe.",en:"Multimodal: placeholder for an audio input."},
};
const SPECIAL_TOKEN_PATTERNS = [
 [/^<\|reserved_special_token_\d+\|>$/,{de:"Llama-3 reservierter Platzhalter — Slot für künftige/eigene Special-Tokens; mit Embedding-Normen prüfen, ob er schwach trainiert wirkt.",en:"Llama-3 reserved placeholder — slot for future/custom special tokens; check embedding norms before treating it as weakly trained."}],
 [/^<unused\d+>$/,{de:"Reservierter, unbenutzter Slot (Gemma/T5-Stil) — Platzhalter; mit Embedding-Normen prüfen, ob er schwach trainiert wirkt.",en:"Reserved, unused slot (Gemma/T5 style) — placeholder; check embedding norms before treating it as weakly trained."}],
 [/^<extra_id_\d+>$/,{de:"T5-Sentinel — Masken-Platzhalter aus dem Span-Corruption-Training.",en:"T5 sentinel — mask placeholder from span-corruption training."}],
 [/^\[PAD\d*\]$/,{de:"Padding-Platzhalter-Slot.",en:"Padding placeholder slot."}],
 [/^<0x[0-9A-Fa-f]{2}>$/,{de:"Byte-Fallback-Token — ein einzelnes rohes UTF-8-Byte (für nicht im Vokab enthaltene Zeichen).",en:"Byte-fallback token — a single raw UTF-8 byte (for characters not in the vocab)."}],
 [/^<dummy\d+>$/,{de:"DeepSeek reservierter Platzhalter-Slot; mit Embedding-Normen prüfen, ob er schwach trainiert wirkt.",en:"DeepSeek reserved placeholder slot; check embedding norms before treating it as weakly trained."}],
 [/^\[control_\d+\]$/,{de:"Mistral v3 reservierter Control-Slot — Platzhalter für künftige Steuer-Tokens.",en:"Mistral v3 reserved control slot — placeholder for future control tokens."}],
 [/^<SPECIAL_\d+>$/,{de:"Mistral/tekken reservierter Special-Slot — Platzhalter.",en:"Mistral/tekken reserved special slot — placeholder."}],
 [/^\n+$/,{de:"Whitespace-Token: eine Folge von N Zeilenumbrüchen als EIN Token (häufig in Code-Tokenizern, z.B. Gemma).",en:"Whitespace token: a run of N newlines as ONE token (common in code tokenizers, e.g. Gemma)."}],
 [/^ +$/,{de:"Whitespace-Token: eine Folge von N Leerzeichen als EIN Token (Einrückung; häufig in Code-Tokenizern).",en:"Whitespace token: a run of N spaces as ONE token (indentation; common in code tokenizers)."}],
 // Split-pipe control delimiters seen in some non-standard models (e.g. a
 // "gemma4" ollama build with no chat_template): the pipe sits on ONE side —
 // <|name> opens a section, <name|> closes it; both ship typed CONTROL. Label
 // the structure honestly without over-claiming the exact protocol.
 [/^<\|[a-z][a-z0-9_]*>$/,{de:"Steuer-Token (Öffner) — modell-spezifisches Split-Pipe-Format: öffnet einen Abschnitt (z.B. tool/channel/turn). Gegenstück: &lt;name|&gt;.",en:"Control token (opener) — model-specific split-pipe format: opens a section (e.g. tool/channel/turn). Counterpart: &lt;name|&gt;."}],
 [/^<[a-z][a-z0-9_]*\|>$/,{de:"Steuer-Token (Schließer) — Gegenstück zu &lt;|name&gt;.",en:"Control token (closer) — counterpart of &lt;|name&gt;."}],
];
function tokExplain(tok){
  const e = SPECIAL_TOKEN_INFO[tok];
  if(e) return LANG==="en"?e.en:e.de;
  for(const [rx,info] of SPECIAL_TOKEN_PATTERNS) if(rx.test(tok)) return LANG==="en"?info.en:info.de;
  return "";
}
// build an expandable <details> list of tokens, annotating known ones
function tokDetails(title, toks, truncated, cls){
  if(!toks||!toks.length) return "";
  const known = toks.filter(t=>tokExplain(t));
  const rows = toks.map(t=>{
    const ex = tokExplain(t);
    return `<div class="tokrow ${cls||''}"><code>${esc(t)}</code>`+
      (ex?`<span class="ex">${ex}</span>`:"")+`</div>`;
  }).join("");
  const cap = truncated ? ` ${tr("(erste")} ${toks.length})` : ` (${toks.length})`;
  const note = known.length ? ` · <span style="color:var(--cyan)">${known.length} ${tr("erklärt")}</span>` : "";
  return `<details class="tokdex"><summary>${esc(title)}<span class="cnt">${cap}${note}</span></summary>`+
    `<div class="body">${rows}</div></details>`;
}

function openModal(i){ openModelModal(MODELS[i]); }
function openModelModal(m){
  const g=m.grid;
  document.getElementById("m-nm").textContent=m.name||m.label;
  document.getElementById("m-sub").textContent=
    `${m.arch} · ${m.label} · ${fmtN(m.params)} params · ${m.n_tensors} tensors · ${fmtB(m.file_size)}`;

  // source block: format · precision line + per-model degradation warnings.
  // Section is hidden when the model has no source block at all (graceful).
  const sec=document.getElementById("m-warn-sec"), wbox=document.getElementById("m-warnings");
  const src=m.source;
  if(src){
    sec.style.display="";
    const warns=(src.warnings||[]);
    let wh = `<div class="kv2">${[
      kvrow("","format", src.format),
      kvrow("","precision", tr(src.precision||"–")),
      kvrow("","arch", src.arch),
      kvrow("","mapped", src.mapped?tr("ja"):tr("nein"), src.mapped?"good":"warn"),
    ].join("")}</div>`;
    if(warns.length){
      wh += `<div style="margin-top:10px">`+warns.map(w=>
        `<div class="kv"><span class="v warn" style="text-align:left;word-break:break-word">⚠ ${esc(w)}</span></div>`
      ).join("")+`</div>`;
    } else {
      wh += `<div style="margin-top:8px;color:var(--phos);font-size:11px">✓ ${tr("keine Degradationen")}</div>`;
    }
    wbox.innerHTML=wh;
  } else {
    sec.style.display="none";
    wbox.innerHTML="";
  }

  // headline specs (with inline help on labels)
  const heads = (hasN(m.n_head) && hasN(m.n_head_kv)) ? `${m.n_head}/${m.n_head_kv}` : "–";
  const specs=[["arch",m.arch,"arch"],["layers",m.n_layers,"layers"],["d_model",m.d_model,"d_model"],
    ["heads",heads,"heads"],["ffn",fmtN(m.ffn),"ffn"],["ctx",fmtN(m.ctx),"ctx"]];
  document.getElementById("m-specs").innerHTML=specs.map(s=>
    `<div class="spec"><div class="sv">${s[1]==null?'–':esc(s[1])}</div><div class="sl">${s[0]}${hq(s[2])}</div></div>`).join("");

  // derived (level 1b)
  const ftConsistent = m.file_type_label && m.dom_quant &&
    m.file_type_label.indexOf(m.dom_quant)===0;
  const ftCls = m.dom_quant ? (ftConsistent?"good":"warn") : "";
  const fileType = m.file_type_label ? `${m.file_type_label} (#${m.file_type})`
    : (hasN(m.file_type) ? "#"+m.file_type : null);
  const dr=[
    kvrow("bits_per_weight","bits / weight", m.bits_per_weight?m.bits_per_weight.toFixed(2):null),
    kvrow("file_type","file_type", fileType),
    kvrow("quant","dominanter Tensor-Quant", m.dom_quant?`${m.dom_quant} ${ftConsistent?"✓ konsistent":"⚠ vgl. file_type"}`:null, ftCls),
    kvrow("head_dim","head_dim", m.head_dim),
    kvrow("ffn_ratio","FFN-Verhältnis", m.ffn_ratio?m.ffn_ratio+"×":null),
    kvrow("kv_cache","KV-Cache @"+fmtN(m.ctx), m.kv_cache?`${fmtB(m.kv_cache)}${m.kv_cache_kind?" · "+m.kv_cache_kind:""}`:null),
    kvrow("rope_freq_base","RoPE freq_base", m.rope_freq_base),
    kvrow("ctx_extended","Kontext gestreckt", m.ctx_extended?`ja (${m.rope_scaling_type||''} ×${m.rope_scaling_factor||'?'}, orig ${fmtN(m.rope_orig_ctx)})`:"nein", m.ctx_extended?"warn":""),
    kvrow("sliding_window","Sliding-Window", m.sliding_window),
    kvrow("moe","MoE Experten", m.moe?`${m.moe.expert_used_count}/${m.moe.expert_count} aktiv${m.moe.expert_shared_count?` (+${m.moe.expert_shared_count} shared)`:''}`:"nein", m.moe?"good":""),
    kvrow("tied_embeddings","Tied Embeddings", m.tied_embeddings?"ja":"nein"),
    kvrow("vocab","Vokabular", fmtN(m.vocab)),
    kvrow("rms_eps","RMSNorm-ε", m.rms_eps),
    kvrow("bias_count","Bias-Tensoren", m.bias_count),
    kvrow("gguf_version","GGUF-Version", m.gguf_version),
    kvrow("alignment","Alignment", hasN(m.alignment)?m.alignment+" B":null),
    kvrow("data_bytes","Daten-Bytes", m.data_bytes?fmtB(m.data_bytes):null),
  ];
  document.getElementById("m-derived").innerHTML=dr.join("");

  // tokenizer & special tokens (level 1 + 3)
  const sp=m.special||{};
  document.getElementById("m-tok").innerHTML=[
    kvrow("tok_model","Tokenizer-Modell", m.tokenizer_model),
    kvrow("pretok","Pre-Tokenizer", sp.pre),
    kvrow("vocab","Vokabulargröße", fmtN(m.vocab)),
    kvrow("merges","BPE-Merges", m.merges_n?fmtN(m.merges_n):null),
    kvrow("special","BOS-Token", sp.bos),
    kvrow("special","EOS-Token", sp.eos),
    kvrow("special","PAD-Token", sp.pad),
    kvrow("special","UNK-Token", sp.unk),
    kvrow("","add_bos / add_eos", `${sp.add_bos} / ${sp.add_eos}`),
    kvrow("chat_template","Chat-Template", m.chat_template_len?`${m.chat_template_len} ${LANG==="en"?"chars":"Zeichen"}`:"–"),
  ].join("");
  const toks=(m.meta||{})["tokenizer.ggml.tokens"];
  let tokHtml = (toks&&toks._array)
    ? `<b style="color:var(--ink)">${tr("Token-Stichprobe")}</b>${hq('tokens')} (von ${fmtN(toks.len)}): `+
      toks.sample.map(t=>`<span style="color:var(--cyan)">${esc(JSON.stringify(t))}</span>`).join(" ")
    : "";
  // ---- tokenizer forensics (Ebene 3) ----
  const fo=forOf(m);
  if(fo){
    const tc=Object.entries(fo.type_counts||{}).sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`${k} <span style="color:var(--ink)">${fmtN(v)}</span>`).join(" · ");
    const scr=Object.entries(fo.script_hist||{}).slice(0,8)
      .map(([k,v])=>`${k} <span style="color:var(--ink)">${v}</span>`).join(" · ");
    tokHtml += `
      <div style="margin-top:12px;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px 18px">
        ${kvrow("token_type","Token-Typen", tc, "", true)}
        ${kvrow("reserved","reservierte/unbenutzte Slots", `${fmtN(fo.reserved_count)}`, fo.reserved_count>1000?"warn":"")}
        ${kvrow("special_tok","funktionale Special-Tokens", fmtN(fo.special_count))}
        ${kvrow("","UNKNOWN-Tokens", fmtN(fo.unknown_count!=null?fo.unknown_count:((fo.type_counts||{}).UNKNOWN||0)), (fo.unknown_count||((fo.type_counts||{}).UNKNOWN))?"warn":"")}
        ${kvrow("scripts","Skripte (Stichprobe "+fmtN(fo.script_sampled)+")", scr, "", true)}
      </div>`;
    // complete, expandable lists (fall back to *_sample on older reports)
    const special = fo.special && fo.special.length ? fo.special : (fo.special_sample||[]);
    const reserved = fo.reserved && fo.reserved.length ? fo.reserved : (fo.reserved_sample||[]);
    const unknown = fo.unknown || [];
    tokHtml += tokDetails(tr("funktionale Special-Tokens"), special, fo.special_truncated, "");
    if(unknown.length)
      tokHtml += tokDetails(tr("UNKNOWN-Tokens"), unknown, fo.unknown_truncated, "unk");
    else if(fo.unknown_count===undefined && (fo.type_counts||{}).UNKNOWN)
      tokHtml += `<div style="margin-top:6px;font-size:10px;color:var(--faint)">${tr("UNKNOWN-Liste: Forensik neu bauen (tokenizer_forensics.py)")}</div>`;
    tokHtml += tokDetails(tr("reservierte / unbenutzte Slots"), reserved, fo.reserved_truncated, "res");
  }
  document.getElementById("m-toksample").innerHTML=tokHtml;

  // raw header hex/ascii (level 1)
  document.getElementById("m-hex").innerHTML=renderHex(m);
  document.getElementById("m-layout").innerHTML=[
    kvrow("","Magic", '47 47 55 46 = "GGUF"'),
    kvrow("gguf_version","Version", m.gguf_version),
    kvrow("","n_tensors", m.n_tensors),
    kvrow("","n_kv (Metadaten)", m.meta?Object.keys(m.meta).length:"?"),
    kvrow("","Header-Ende (Byte)", m.header_end),
    kvrow("alignment","Alignment", m.alignment),
    kvrow("","Daten-Start (Byte)", m.data_start),
  ].join("");

  // heatmap (level 2)
  const roles=g.roles, L=g.layers;
  let th=`<tr><th></th>`+roles.map(r=>`<th class="colh">${r}</th>`).join("")+`</tr>`;
  let rows="";
  for(let l=0;l<L;l++){
    rows+=`<tr><th>blk.${l}</th>`+roles.map(r=>{
      const t=(g.cells[r]||{})[l];
      if(!t) return `<td style="background:#0a0e0d"></td>`;
      return `<td style="background:${qc(t)}" data-t="${t}" data-r="${r}" data-l="${l}"></td>`;
    }).join("")+`</tr>`;
  }
  document.getElementById("m-hm").innerHTML=th+rows;
  document.querySelectorAll("#m-hm td[data-t]").forEach(td=>{
    td.onmousemove=e=>{tip.style.display="block";tip.style.whiteSpace="nowrap";tip.style.maxWidth="none";
      tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
      tip.innerHTML=`blk.${td.dataset.l}.${td.dataset.r}<br><b>${td.dataset.t}</b>`;};
    td.onmouseleave=hideTip;
  });
  const present=[...new Set(roles.flatMap(r=>Object.values(g.cells[r]||{})))];
  document.getElementById("m-leg").innerHTML=present.sort().map(t=>
    `<div class="g"><i style="background:${qc(t)}"></i>${t}</div>`).join("");
  const gl=Object.entries(g.globals);
  document.getElementById("m-glob").innerHTML=
    `<b style="color:var(--ink)">${tr("Globale Tensoren")} (${gl.length}):</b> `+
    gl.map(([n,t])=>`${esc(n)} <span style="color:${qc(t)}">[${t}]</span>`).join(" · ");

  // complete metadata (level 1, grouped)
  const meta=m.meta||{};
  document.getElementById("m-kvn").textContent=Object.keys(meta).length;
  const groups={"general":[],[m.arch]:[],"tokenizer":[],"andere":[]};
  Object.keys(meta).sort().forEach(k=>{
    const grp = k.startsWith("general.")?"general"
      : k.startsWith(m.arch+".")?m.arch
      : k.startsWith("tokenizer")?"tokenizer":"andere";
    groups[grp].push(k);
  });
  let mh="";
  for(const [gname,keys] of Object.entries(groups)){
    if(!keys.length) continue;
    mh+=`<div class="metagrp"><div class="gh">${esc(tr(gname))} · ${keys.length}</div>`+
      keys.map(k=>`<div class="mrow"><span class="mk">${esc(k)}</span><span class="mv">${fmtMeta(meta[k])}</span></div>`).join("")+`</div>`;
  }
  document.getElementById("m-meta").innerHTML=mh;

  document.getElementById("modal").classList.add("on");
}

// format the raw 64-byte header as offset | hex | ascii with field annotations
function renderHex(m){
  const hex=m.header_hex||"", asc=m.header_ascii||"";
  let out="";
  for(let off=0;off<hex.length/2;off+=16){
    const bytes=[];
    for(let j=0;j<16 && (off+j)*2<hex.length;j++) bytes.push(hex.substr((off+j)*2,2));
    const hexpart=bytes.map((b,j)=>{
      const idx=off+j;
      // annotate the structural header fields
      // colour the four structural header fields distinctly
      const cls = idx<4?"f-magic":idx<8?"f-ver":idx<16?"f-nt":idx<24?"f-nkv":"hxb";
      return `<span class="${cls}">${b}</span>`;
    }).join(" ");
    const ascpart=asc.substr(off,16).padEnd(16," ");
    out+=`<span class="off">${off.toString(16).padStart(4,"0")}</span>  ${hexpart.padEnd(16*3-1," ")}  <span class="asc">${esc(ascpart)}</span>\n`;
  }
  out+=`\n<span class="f-magic">0–3</span> magic "GGUF"   <span class="f-ver">4–7</span> version   <span class="f-nt">8–15</span> n_tensors   <span class="f-nkv">16–23</span> n_kv  <span class="off">(alle little-endian)</span>`;
  return out;
}
document.getElementById("m-close").onclick=()=>document.getElementById("modal").classList.remove("on");
document.getElementById("modal").onclick=e=>{if(e.target.id==="modal")e.target.classList.remove("on");};
document.addEventListener("keydown",e=>{if(e.key==="Escape")document.getElementById("modal").classList.remove("on");});

document.getElementById("foot").innerHTML=
  `self-contained report viewer · no model files or forward-pass executed in the browser · `+
  `click a model for source, tokenizer, tensor and metadata details · ${MODELS.length} unique models`;

// ---- render glossary tab: grouped by category, live search ----------------
function glHl(s,q){ s=esc(s); if(!q) return s;
  const r=q.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");
  try{ return s.replace(new RegExp("("+r+")","ig"),"<mark>$1</mark>"); }catch(e){ return s; }
}
function renderGlossary(){
  const box=document.getElementById("gloss");
  const inp=document.getElementById("gloss-search");
  const q=((inp&&inp.value)||"").trim().toLowerCase();
  const keys=Object.keys(GLOSSARY), total=keys.length;
  let shown=0, html="";
  for(const [ck] of CATS){
    const inCat=keys.filter(k=>(GLOSSARY[k].c||"gguf")===ck);
    const hit=inCat.filter(k=>{const g=G(k);
      return !q || (g.t+" "+g.d+" "+(g.b||"")).toLowerCase().includes(q);});
    if(!hit.length) continue;
    shown+=hit.length;
    html+=`<div class="gcat"><div class="gcat-h"><span class="gc-n">${esc(catName(ck))}</span>`+
      `<span class="gc-c">${hit.length}</span><span class="line"></span></div><div class="gloss">`+
      hit.map(k=>{const g=G(k);
        return `<div class="gcard"><div class="gt">${glHl(g.t,q)}</div><div class="gd">${glHl(g.d,q)}</div>`+
          (g.b?`<span class="gb">▸ ${glHl(g.b,q)}</span>`:"")+`</div>`;}).join("")+`</div></div>`;
  }
  box.innerHTML = html || `<div class="gloss-empty">${LANG==="en"?"No matches.":"Keine Treffer."}</div>`;
  const cnt=document.getElementById("gloss-count");
  if(cnt) cnt.textContent = q ? `${shown} / ${total}` : `${total} ${LANG==="en"?"terms":"Begriffe"}`;
}
// ---- Task 16: run-log tab -------------------------------------------------
// counts computed ONLY from the source array (RUNLOG) -> data integrity:
// rendered "X info · Y warn · Z error" == counts of the underlying JSON.
const SEVS=["info","warn","error"];
const SEVCOLOR={info:"var(--phos)",warn:"var(--amber)",error:"var(--red)"};
function logCounts(){const c={info:0,warn:0,error:0};
  for(const e of RUNLOG){ if(c[e.severity]!=null) c[e.severity]++; } return c; }
// best-effort model -> fleet index (match name/label/alias). '*' = run-wide, no jump.
function modelIndex(name){
  if(!name||name==="*") return -1;
  for(let i=0;i<MODELS.length;i++){const m=MODELS[i];
    if(m.name===name||m.label===name||(m.aliases||[]).includes(name)) return i;}
  return -1;
}
let LOG_SEV="all", LOG_MODEL="all", LOG_STAGE="all";
function renderLogBadge(){
  const b=document.getElementById("log-badge"); if(!b) return;
  const c=logCounts(); const we=c.warn+c.error;
  if(we<=0){ b.style.display="none"; b.textContent=""; return; }
  b.style.display="inline-flex";
  b.className="logbadge"+(c.error>0?" err":"");
  b.textContent=(c.error>0?"✕ ":"⚠ ")+we;
  b.title=`${c.warn} warn · ${c.error} error`;
}
function renderLog(){
  const wrap=document.getElementById("log-wrap"); if(!wrap) return;
  const counts=logCounts();
  // counts header (from source) — language-neutral severity words are i18n-safe
  document.getElementById("log-counts").innerHTML = SEVS.map(s=>
    `<span class="c"><span class="dot" style="background:${SEVCOLOR[s]}"></span>`+
    `<b style="color:${SEVCOLOR[s]}">${counts[s]}</b> ${s}</span>`).join('<span style="color:var(--faint)">·</span>');
  // filter <select>s: severity, model, stage (built from the data)
  const models=[...new Set(RUNLOG.map(e=>e.model))].sort();
  const stages=[...new Set(RUNLOG.map(e=>e.stage))].sort();
  const sel=(id,cur,allLbl,opts)=>`<select id="${id}"><option value="all">${esc(tr(allLbl))}</option>`+
    opts.map(o=>`<option value="${esc(o)}"${o===cur?" selected":""}>${esc(o)}</option>`).join("")+`</select>`;
  document.getElementById("log-filters").innerHTML =
    `<span class="lf-l">${esc(tr("Filter:"))}</span>`+
    sel("lf-sev",LOG_SEV,"alle Schweren",SEVS)+
    sel("lf-model",LOG_MODEL,"alle Modelle",models)+
    sel("lf-stage",LOG_STAGE,"alle Stufen",stages);
  document.getElementById("lf-sev").onchange=e=>{LOG_SEV=e.target.value;renderLogRows();};
  document.getElementById("lf-model").onchange=e=>{LOG_MODEL=e.target.value;renderLogRows();};
  document.getElementById("lf-stage").onchange=e=>{LOG_STAGE=e.target.value;renderLogRows();};
  renderLogRows();
}
function renderLogRows(){
  const wrap=document.getElementById("log-wrap");
  if(!RUNLOG.length){ wrap.innerHTML=`<div class="log-empty">${esc(tr("keine Log-Einträge"))}</div>`; return; }
  // newest-first: reverse the source order (entries appended chronologically)
  const rows=RUNLOG.map((e,i)=>[e,i]).reverse().filter(([e])=>
    (LOG_SEV==="all"||e.severity===LOG_SEV)&&
    (LOG_MODEL==="all"||e.model===LOG_MODEL)&&
    (LOG_STAGE==="all"||e.stage===LOG_STAGE));
  if(!rows.length){ wrap.innerHTML=`<div class="log-empty">${esc(tr("keine Einträge für diesen Filter"))}</div>`; return; }
  // column headers — DE source words; tr() swaps to EN when LANG==="en"
  const colhdr=`<thead><tr>`+
    ["Zeit","Schwere","Modell","Stufe","Code","Meldung"].map(c=>`<th>${esc(tr(c))}</th>`).join("")+
    `</tr></thead>`;
  const body=`<tbody>`+rows.map(([e])=>{
    const mi=modelIndex(e.model);
    const modelCell=(mi>=0)
      ? `<span class="lmodel jump" data-mi="${mi}">${esc(e.model)}</span>`
      : `<span class="lmodel">${esc(e.model)}</span>`;
    return `<tr class="sev-${esc(e.severity)}">`+
      `<td class="lts">${esc(e.ts||"")}</td>`+
      `<td><span class="sevb ${esc(e.severity)}">${esc(e.severity)}</span></td>`+
      `<td>${modelCell}</td>`+
      `<td class="lstage">${esc(e.stage||"")}</td>`+
      `<td class="lcode">${esc(e.code||"")}</td>`+
      `<td class="lmsg">${esc(e.msg||"")}</td></tr>`;
  }).join("")+`</tbody>`;
  wrap.innerHTML=`<table class="logtbl">${colhdr}${body}</table>`;
  // best-effort: clicking a real model jumps to its card detail (reuses modal)
  wrap.querySelectorAll(".lmodel.jump").forEach(el=>el.onclick=()=>{
    const i=+el.dataset.mi; if(i>=0&&typeof openModal==="function") openModal(i);
  });
}

// ---- static cross-model comparisons --------------------------------------
function cmpEmpty(el, msg){
  el.innerHTML=`<div class="log-empty">${esc(tr(msg))}</div>`;
}
function pct(x){return hasN(x)?Math.round(+x*100)+"%":"–";}
function renderCompare(){
  const sum=document.getElementById("cmp-summary");
  const lin=document.getElementById("cmp-lineage");
  const aq=document.getElementById("cmp-archquant");
  const ad=document.getElementById("cmp-anomdiff");
  if(!sum||!lin||!aq||!ad) return;
  if(!COMPARE||!COMPARE.schema){
    sum.innerHTML=""; cmpEmpty(lin,"keine Vergleichsdaten geladen"); aq.innerHTML=""; ad.innerHTML=""; return;
  }
  const cov=COMPARE.coverage||{};
  sum.innerHTML=[
    ["Modelle",cov.model_count||0,"im Vergleichsreport"],
    ["Tokenizer",cov.tokenizer_models||0,"direkt aus Modellpfaden gelesen"],
    ["Lineage-Paare",(COMPARE.lineage_pairs||[]).length,"heuristische Paar-Scores"],
    ["Anomalie-Sets",((COMPARE.anomalies||{}).weight_stats||[]).length+((COMPARE.anomalies||{}).spectral||[]).length,"aus Gewichtsreports"],
  ].map(([t,v,d])=>`<div class="cmp-card"><div class="ct">${esc(tr(t))}</div><div class="cv">${esc(v)}</div><div class="cd">${esc(tr(d))}</div></div>`).join("");

  const pairs=(COMPARE.lineage_pairs||[]).slice(0,12);
  if(!pairs.length) cmpEmpty(lin,"keine Lineage-Paare");
  else lin.innerHTML=pairs.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} → ${esc(p.b)} <span class="score">${pct(p.score)}</span></div>`+
    `<div class="meta">${esc(p.class||"")}</div><div class="cmp-tags">`+
    Object.entries(p.signals||{}).map(([k,v])=>`<span>${esc(k)} ${pct(v)}</span>`).join("")+
    `</div></div>`).join("");

  const arch=(COMPARE.architecture_pairs||[]).slice(0,8);
  const health=(COMPARE.health_summary||[]).slice(0,8);
  const ctxPairs=(COMPARE.context_pairs||[]).slice(0,8);
  const quant=(COMPARE.quant_pairs||[]).slice(0,8);
  const toks=(COMPARE.tokenizer_pairs||[]).slice(0,8);
  const tensors=(COMPARE.tensor_pairs||[]).slice(0,8);
  const clusters=(COMPARE.architecture_clusters||[]).filter(c=>(c.models||[]).length>1).slice(0,6);
  if(!health.length&&!arch.length&&!ctxPairs.length&&!quant.length&&!toks.length&&!tensors.length&&!clusters.length) cmpEmpty(aq,"keine Architektur-/Quant-/Tokenizer-Paare");
  else aq.innerHTML=[
    ...health.map(h=>`<div class="cmp-row"><div class="pair">${esc(h.model)} <span class="${h.errors?'hot':h.warnings?'warn':'score'}">Health ${esc(h.score)}</span></div>`+
      `<div class="meta">${esc(h.class||"")} · errors ${esc(h.errors)} · warnings ${esc(h.warnings)} · infos ${esc(h.infos)}</div>`+
      `<div class="cmp-tags">${(h.top_issues||[]).slice(0,6).map(i=>`<span>${esc(i.section)}:${esc(i.code)}</span>`).join("")}</div></div>`),
    ...clusters.map(c=>`<div class="cmp-row"><div class="pair">Cluster <span class="score">${esc(c.size)} Modelle</span></div>`+
      `<div class="meta">${esc((c.models||[]).join(" · "))}</div><div class="cmp-tags">`+
      `<span>${esc((c.signature||{}).arch||"?")} L=${esc((c.signature||{}).layers)} d=${esc((c.signature||{}).d_model)}</span>`+
      `<span>formats ${(c.formats||[]).map(esc).join(", ")}</span>`+
      `${c.ctx_range?`<span>ctx ${esc(c.ctx_range[0])}..${esc(c.ctx_range[1])}</span>`:""}</div></div>`),
    ...arch.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="${p.compat_core?'score':'warn'}">${p.compat_core?'core-kompatibel':'core-diff'}</span></div>`+
      `<div class="meta">${esc(Object.keys(p.diffs||{}).slice(0,8).join(", ")||"keine Struktur-Diffs in den verglichenen Feldern")}</div></div>`),
    ...ctxPairs.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="${p.score>=0.9?'score':'warn'}">Context ${pct(p.score)}</span></div>`+
      `<div class="meta">ctx ${esc((p.ctx||[])[0])} ↔ ${esc((p.ctx||[])[1])} · RoPE ${pct(p.rope_compat)} · KV ${pct(p.kv_cache_ratio)}${(p.notes||[]).length?` · ${(p.notes||[]).map(esc).join(" · ")}`:""}</div>`+
      `<div class="cmp-tags">${Object.keys(p.diffs||{}).slice(0,6).map(k=>`<span>${esc(k)}</span>`).join("")}</div></div>`),
    ...quant.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="score">Quant ${pct(p.breakdown_jaccard)}</span></div>`+
      `<div class="meta">${Object.keys(p.role_quant_diffs||{}).length} Rollen mit anderer Mehrheits-Quantisierung</div></div>`),
    ...tensors.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="score">Tensor ${pct(p.name_jaccard)}</span></div>`+
      `<div class="meta">roles ${pct(p.role_jaccard)} · shared ${esc(p.shared_names)} · same-shape ${pct(p.same_shape_shared_ratio)} · shape mismatches ${esc(p.shape_mismatch_count)}</div></div>`),
    ...toks.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="score">Vokab ${pct(p.vocab_jaccard)}</span></div>`+
      `<div class="meta">same-id ${pct(p.same_id_shared_ratio)} · shared tokens ${esc(p.shared_tokens)}${hasN(p.merge_jaccard)?` · BPE merges ${pct(p.merge_jaccard)} (${esc(p.shared_merges)} shared)`:""}</div></div>`)
  ].join("");

  const anoms=[...(((COMPARE.anomalies||{}).weight_stats)||[]),...(((COMPARE.anomalies||{}).spectral)||[])]
    .filter(x=>(x.anomalies||[]).length).slice(0,8);
  const diffs=(COMPARE.diff_explanations||[]).slice(0,4);
  const chatPairs=(COMPARE.chat_template_pairs||[]).slice(0,6);
  const chats=(COMPARE.chat_templates||[]).slice(0,4);
  const embeds=(COMPARE.embedding_summaries||[]).filter(e=>(e.seed_neighbors||[]).length||e.output_head_compare).slice(0,4);
  const moes=(COMPARE.moe_analysis||[]).slice(0,4);
  const mm=(COMPARE.multimodal_inventory||[]).slice(0,4);
  if(!anoms.length&&!diffs.length&&!chatPairs.length&&!chats.length&&!embeds.length&&!moes.length&&!mm.length) cmpEmpty(ad,"keine Anomalien oder Diff-Erklärungen");
  else ad.innerHTML=[
    ...anoms.map(x=>`<div class="cmp-row"><div class="pair">${esc(x.model)} <span class="warn">${x.anomalies.length} Anomalien</span></div>`+
      `<div class="cmp-tags">${x.anomalies.slice(0,6).map(a=>`<span>${esc(a.metric)} blk.${a.layer}.${esc(a.role)} z=${esc(a.robust_z)}</span>`).join("")}</div></div>`),
    ...diffs.map(d=>`<div class="cmp-row"><div class="pair">${esc(d.label||"diff")} <span class="score">${esc(d.matched||0)} matched</span></div>`+
      `<div class="meta">${esc((d.notes||[]).join(" · "))}</div><div class="cmp-tags">${(d.top_roles||[]).slice(0,6).map(r=>`<span>${esc(r.role)} Δ=${esc(r.mean_delta)}</span>`).join("")}</div></div>`),
    ...chatPairs.map(p=>`<div class="cmp-row"><div class="pair">${esc(p.a)} ↔ ${esc(p.b)} <span class="${p.score>=0.8?'score':'warn'}">prompt ${pct(p.score)}</span></div>`+
      `<div class="meta">${esc(p.class||"")} · markers ${pct(p.marker_jaccard)} · special IDs ${pct(p.special_id_same_ratio)}${(p.missing_template||[]).length?` · missing template: ${(p.missing_template||[]).map(esc).join(", ")}`:""}</div>`+
      `<div class="cmp-tags">${(p.shared_markers||[]).map(k=>`<span>${esc(k)}</span>`).join("")}${Object.keys(p.special_mismatches||{}).map(k=>`<span>${esc(k)} mismatch</span>`).join("")}</div></div>`),
    ...chats.map(c=>`<div class="cmp-row"><div class="pair">${esc(c.model)} <span class="score">chat template</span></div>`+
      `<div class="cmp-tags">${Object.entries(c.markers||{}).filter(([,v])=>v).map(([k])=>`<span>${esc(k)}</span>`).join("")}</div></div>`),
    ...embeds.map(e=>`<div class="cmp-row"><div class="pair">${esc(e.model)} <span class="score">embedding</span></div>`+
      `<div class="meta">near-zero=${esc(e.near_zero)} · anisotropy=${esc(e.anisotropy)}${e.output_head_compare?` · output Δ=${esc(e.output_head_compare.delta)}`:""}</div>`+
      `<div class="cmp-tags">${(e.seed_neighbors||[]).slice(0,4).map(s=>`<span>${esc(s.seed)} → ${(s.neighbors||[]).slice(0,3).map(n=>esc(n.tok)).join(", ")}</span>`).join("")}</div></div>`),
    ...moes.map(m=>`<div class="cmp-row"><div class="pair">${esc(m.model)} <span class="score">MoE</span></div>`+
      `<div class="meta">${esc((m.expert_roles||[]).length)} expert/shared roles measured</div></div>`),
    ...mm.map(m=>`<div class="cmp-row"><div class="pair">${esc(m.model)} <span class="warn">multimodal inventory</span></div>`+
      `<div class="meta">vision=${esc(m.vision)} · audio=${esc(m.audio)} · tower roles=${esc((m.tower_roles||[]).length)}</div></div>`)
  ].join("");
}

// initial paint of everything in the persisted language (applyLang calls all
// the render fns: KPIs, fleet, overlap, statGrid×3, embControls, diff, glossary)
applyLang();

// ---- tabs: switch panes, persist, focus search on glossary ----------------
let TAB = (window.localStorage && localStorage.getItem("mm-tab")) || "fleet";
function setTab(t){
  if(!document.querySelector('.tabpane[data-tab="'+t+'"]')) t="fleet";
  TAB=t; try{ localStorage.setItem("mm-tab",t); }catch(e){}
  document.querySelectorAll(".tabpane").forEach(p=>p.classList.toggle("on",p.dataset.tab===t));
  document.querySelectorAll("#tabs .tab").forEach(el=>el.classList.toggle("on",el.dataset.tab===t));
}
document.querySelectorAll("#tabs .tab").forEach(el=>el.onclick=()=>setTab(el.dataset.tab));
const _gs=document.getElementById("gloss-search"); if(_gs) _gs.addEventListener("input",renderGlossary);
setTab(TAB);

// ---- Tutorial mode: guided learning layer over a fictitious model ----------
const TUTORIAL_MODEL = {
  label:"strata-tutor-7b.gguf", path:"/tutorial/strata-tutor-7b.gguf",
  arch:"llama", source:{format:"gguf",precision:"approx",arch:"llama",mapped:true,warnings:["Tutorial-Datensatz: fiktiv, nicht aus einer echten Modelldatei gelesen."]},
  name:"Strata-Tutor-7B", basename:"Strata-Tutor", finetune:"instruct-tutorial",
  size_label:"7B", license:"fictional", aliases:["tutorial:fiktives-modell"],
  params:7200000000, file_size:4200000000, n_tensors:291, n_layers:32,
  d_model:4096, n_head:32, n_head_kv:8, gqa:4, ffn:11008,
  ffn_ratio:2.69, ctx:8192, vocab:32000, quant:{"Q4_K":210,"Q6_K":48,"F16":33},
  chat_template_len:420, tokenizer_model:"llama", vision:false, audio:false,
  head_dim:128, key_len:null, val_len:null, kv_cache:1073741824,
  kv_cache_kind:"GQA/MHA, obere Schranke", rope_freq_base:10000,
  ctx_extended:false, rope_scaling_type:null, rope_scaling_factor:null,
  rope_orig_ctx:null, sliding_window:null, moe:null, tied_embeddings:true,
  rms_eps:0.00001, special:{bos:1,eos:2,pad:null,unk:0,add_bos:true,add_eos:false,pre:"llama-bpe"},
  merges_n:28000, quant_version:2, file_type:15, file_type_label:"Q4_K_M",
  dom_quant:"Q4_K", bits_per_weight:4.67, data_bytes:4010000000,
  data_start:32768, header_end:31024, alignment:32, gguf_version:3,
  header_hex:"4747554603000000230100000000000078000000000000002000000000000000",
  header_ascii:"GGUF....#.......x....... .......",
  bias_count:0,
  meta:{
    "general.architecture":"llama",
    "general.name":"Strata-Tutor-7B",
    "general.basename":"Strata-Tutor",
    "general.size_label":"7B",
    "general.finetune":"instruct-tutorial",
    "llama.block_count":32,
    "llama.embedding_length":4096,
    "llama.attention.head_count":32,
    "llama.attention.head_count_kv":8,
    "llama.feed_forward_length":11008,
    "llama.context_length":8192,
    "llama.rope.freq_base":10000,
    "llama.attention.layer_norm_rms_epsilon":0.00001,
    "tokenizer.ggml.model":"llama",
    "tokenizer.ggml.pre":"llama-bpe",
    "tokenizer.ggml.bos_token_id":1,
    "tokenizer.ggml.eos_token_id":2,
    "tokenizer.ggml.unknown_token_id":0,
    "tokenizer.ggml.add_bos_token":true,
    "tokenizer.ggml.add_eos_token":false,
    "tokenizer.ggml.tokens":{_array:true,len:32000,elem_type:8,sample:["<unk>","<s>","</s>","▁The","▁model","<unused42>"]}
  },
  grid:{
    roles:["attn_norm","attn_q","attn_k","attn_v","attn_output","ffn_norm","ffn_gate","ffn_up","ffn_down"],
    layers:6,
    cells:{
      attn_norm:{0:"F16",1:"F16",2:"F16",3:"F16",4:"F16",5:"F16"},
      attn_q:{0:"Q4_K",1:"Q4_K",2:"Q4_K",3:"Q4_K",4:"Q4_K",5:"Q4_K"},
      attn_k:{0:"Q4_K",1:"Q4_K",2:"Q4_K",3:"Q4_K",4:"Q4_K",5:"Q4_K"},
      attn_v:{0:"Q6_K",1:"Q6_K",2:"Q6_K",3:"Q6_K",4:"Q6_K",5:"Q6_K"},
      attn_output:{0:"Q4_K",1:"Q4_K",2:"Q4_K",3:"Q4_K",4:"Q4_K",5:"Q4_K"},
      ffn_norm:{0:"F16",1:"F16",2:"F16",3:"F16",4:"F16",5:"F16"},
      ffn_gate:{0:"Q4_K",1:"Q4_K",2:"Q4_K",3:"Q4_K",4:"Q4_K",5:"Q4_K"},
      ffn_up:{0:"Q4_K",1:"Q4_K",2:"Q4_K",3:"Q4_K",4:"Q4_K",5:"Q4_K"},
      ffn_down:{0:"Q6_K",1:"Q6_K",2:"Q6_K",3:"Q6_K",4:"Q6_K",5:"Q6_K"}
    },
    globals:{"token_embd.weight":"F16","output.weight":"F16","output_norm.weight":"F16"},
    role_params:{attn_q:100663296,attn_k:25165824,attn_v:25165824,attn_output:100663296,ffn_gate:270532608,ffn_up:270532608,ffn_down:270532608}
  }
};
const TUTORIAL_STEPS = [
  {tab:"fleet",target:"#tutorial-start",
   title:{de:"Tutorial Mode",en:"Tutorial mode"},
   text:{de:"Wir gehen Schritt fuer Schritt durch modelstrata. Als roter Faden dient das fiktive Modell Strata-Tutor-7B: klein genug zum Erklaeren, aber mit allen typischen Bausteinen eines GGUF-Reports.",
         en:"We will walk through modelstrata step by step. The thread is the fictitious Strata-Tutor-7B model: small enough to explain, but with the typical pieces of a GGUF report."}},
  {tab:"fleet",target:"#kpis",
   title:{de:"1 · Flotte lesen",en:"1 · Read the fleet"},
   text:{de:"Oben siehst du die grobe Lage: wie viele Modelle, wie viele Architekturen, wie viele Parameter und wie viel Speicher. Beim Tutor-Modell waere das ein einzelner 7B-Llama-artiger Report.",
         en:"The top row gives the rough situation: model count, architectures, parameters and disk size. For the tutor model this would be one 7B Llama-style report."}},
  {tab:"fleet",target:"#fleet",
   title:{de:"2 · Modellkarte verstehen",en:"2 · Understand a model card"},
   text:{de:"Eine Karte fasst Architektur, Quelle, Praezision, Layer, Breite, GQA, Kontext und Quantisierung zusammen. Klicke auf echte Karten fuer Details; im Tutorial oeffnet der naechste Schritt das fiktive Detailfenster.",
         en:"A card summarizes architecture, source, precision, layers, width, GQA, context and quantization. Click real cards for details; the next tutorial step opens the fictitious detail view."},
   quiz:{q:{de:"Mini-Check: Bedeutet 'gguf · approx', dass der Header ungenau ist?",
            en:"Mini-check: Does 'gguf · approx' mean the header is imprecise?"},
         a:{de:"Nein. Approx bezieht sich auf rekonstruierte quantisierte Gewichtswerte; Header und Tensorverzeichnis werden direkt gelesen.",
            en:"No. Approx refers to reconstructed quantized weight values; the header and tensor directory are read directly."}}},
  {demo:true,target:"#modal .sheet",
   title:{de:"3 · Detailfenster",en:"3 · Detail view"},
   text:{de:"Das Detailfenster zeigt, was aus Header und Tensorverzeichnis abgeleitet wird: KV-Cache-Obergrenze, RoPE, Tokenizer, Special-Tokens, Hex-Header, Layer-Heatmap und komplette Metadaten.",
         en:"The detail view shows what is derived from the header and tensor directory: KV-cache upper bound, RoPE, tokenizer, special tokens, hex header, layer heatmap and complete metadata."}},
  {tab:"fleet",target:"#scatter",
   title:{de:"4 · Architektur-Topologie",en:"4 · Architecture topology"},
   text:{de:"Die Scatterplots helfen beim Vergleich: Parameter gegen Tiefe/Breite und GQA gegen FFN-Verhaeltnis. Das ist kein Ranking, sondern eine Strukturkarte.",
         en:"The scatter plots help comparison: parameters against depth/width and GQA against FFN ratio. This is not a ranking, but a structure map."}},
  {tab:"tok",target:"#overlap-wrap",
   title:{de:"5 · Tokenizer-Forensik",en:"5 · Tokenizer forensics"},
   text:{de:"Der Jaccard-Overlap zeigt gemeinsame Tokenizer-Herkunft. Beim Tutor-Modell wuerde identisches Vokabular stark fuer gleiche Familie oder Finetune-Verwandtschaft sprechen, aber nicht allein fuer identische Gewichte.",
         en:"Jaccard overlap shows shared tokenizer lineage. For the tutor model, identical vocabulary would strongly suggest the same family or finetune lineage, but not identical weights by itself."},
   quiz:{q:{de:"Mini-Check: Beweist Jaccard 1.0, dass zwei Modelle identisch sind?",
            en:"Mini-check: Does Jaccard 1.0 prove that two models are identical?"},
         a:{de:"Nein. Es beweist nur identisches Vokabular; Gewichte, Finetuning und Verhalten koennen trotzdem verschieden sein.",
            en:"No. It only proves identical vocabulary; weights, fine-tuning and behavior can still differ."}}},
  {tab:"weights",target:"#ws-controls",
   title:{de:"6 · Gewichts-Statistik",en:"6 · Weight statistics"},
   text:{de:"Ebene 4 berechnet Metriken wie Standardabweichung, L2, Sparsity und Outlier-Channels aus statischen Gewichten. Bei GGUF sind quantisierte Gewichte rekonstruiert, daher als approx markiert.",
         en:"Level 4 computes metrics such as standard deviation, L2, sparsity and outlier channels from static weights. For GGUF, quantized weights are reconstructed, hence marked approx."}},
  {tab:"weights",target:"#sp-controls",
   title:{de:"7 · Spektral-Analyse",en:"7 · Spectral analysis"},
   text:{de:"Ebene 5 nutzt Singulaerwerte und einfache Heavy-Tail-Schaetzungen. Das ist ein diagnostischer Hinweis auf Layer-Struktur, kein Ground-Truth-Qualitaetslabel.",
         en:"Level 5 uses singular values and simple heavy-tail estimates. This is a diagnostic signal about layer structure, not a ground-truth quality label."},
   quiz:{q:{de:"Mini-Check: Darf alpha allein ein Modell als 'gut' labeln?",
            en:"Mini-check: Can alpha alone label a model as 'good'?"},
         a:{de:"Nein. Alpha ist eine datenfreie Diagnose und muss mit Evaluation, Format/Praezision und Kontext gelesen werden.",
            en:"No. Alpha is a data-free diagnostic and must be read with evaluation, format/precision and context."}}},
  {tab:"embdiff",target:"#emb-grid",
   title:{de:"8 · Embedding-Geometrie",en:"8 · Embedding geometry"},
   text:{de:"Ebene 6 visualisiert Token-Embeddings: PCA, Norm-Histogramm und niedrigste Normen. Beim Tutor-Modell waeren niedrigste Normen nur Hinweise auf auffaellige Tokens, keine Beweise fuer Glitches.",
         en:"Level 6 visualizes token embeddings: PCA, norm histogram and lowest norms. For the tutor model, lowest norms would be indicators of unusual tokens, not proof of glitches."},
   quiz:{q:{de:"Mini-Check: Reicht der Token-Name '<unused42>' als Glitch-Beweis?",
            en:"Mini-check: Is the token name '<unused42>' proof of a glitch?"},
         a:{de:"Nein. Der Name ist nur eine Heuristik; ein staerkerer Hinweis kommt aus Embedding-Normen und spaeter aus Verhaltenstests.",
            en:"No. The name is only a heuristic; stronger evidence comes from embedding norms and later behavior tests."}}},
  {tab:"embdiff",target:"#df-controls",
   title:{de:"9 · Modell-Diff",en:"9 · Model diff"},
   text:{de:"Ebene 7 vergleicht zwei Modelle tensorweise. Hohe Delta-Werte oder niedrige Cosines zeigen, wo ein Finetune oder eine Abliteration Gewichte staerker veraendert hat.",
         en:"Level 7 compares two models tensor by tensor. High delta values or low cosines show where a finetune or abliteration changed weights more strongly."}},
  {tab:"gloss",target:"#gloss-search",
   title:{de:"10 · Glossar als Lernschicht",en:"10 · Glossary as learning layer"},
   text:{de:"Das Glossar erklaert die Begriffe ohne externe Daten. Dieselben Erklaerungen erscheinen an vielen Fragezeichen im Dashboard.",
         en:"The glossary explains terms without external data. The same explanations appear at many question marks in the dashboard."}},
  {tab:"log",target:"#log-wrap",
   title:{de:"11 · Log und Degradationen",en:"11 · Log and degradations"},
   text:{de:"Der Log-Tab ist die Ehrlichkeits-Schicht: skipped tensors, inventory-only Fallbacks, Approximationen und Fehler stehen hier statt stillschweigend zu verschwinden.",
         en:"The log tab is the honesty layer: skipped tensors, inventory-only fallbacks, approximations and errors appear here instead of silently disappearing."}},
  {tab:"fleet",target:"#tutorial-start",
   title:{de:"Fertig",en:"Done"},
   text:{de:"Damit kennst du den empfohlenen Lesepfad: erst Flotte, dann Detail, dann Tokenizer, Gewichte, Embeddings/Diff, Glossar und Log. Der Tutorial-Modus veraendert keine echten Reports.",
         en:"You now know the recommended reading path: fleet, detail, tokenizer, weights, embeddings/diff, glossary and log. Tutorial mode does not modify real reports."}}
];
function qa(q,a){return {q:{de:q,en:q},a:{de:a,en:a}};}
const QUIZ_BANK = [
  qa("Was ist ein Tensor in modelstrata?","Ein Tensor ist ein mehrdimensionales Zahlen-Array, hier meist eine Gewichtsmatrix wie blk.0.attn_q.weight mit Shape, Typ und Offset."),
  qa("Was bedeutet GGUF?","GGUF ist ein selbstbeschreibendes llama.cpp/ggml-Modellformat mit Header, Metadaten-Key-Values, Tokenizer-Infos und Tensoren in einer Datei."),
  qa("Warum ist GGUF bei Gewichtsmetriken oft 'approx'?","Quantisierte GGUF-Gewichte werden fuer Analysen erst nach fp32 rekonstruiert; diese Rekonstruktion ist verlustbehaftet."),
  qa("Sind GGUF-Header bei 'approx' ungenau?","Nein. Header und Tensorverzeichnis werden direkt gelesen; 'approx' betrifft rekonstruierte quantisierte Gewichtswerte."),
  qa("Was bedeutet 'safetensors · exact'?","Die gespeicherten Float-Gewichte werden direkt gelesen, z.B. fp32/fp16/bf16 nach fp32 fuer Analysen, ohne Quant-Rekonstruktion."),
  qa("Was bedeutet 'inventory-only'?","modelstrata kann Metadaten oder Tensorinventar anzeigen, berechnet aber keine Gewichtsmetriken aus den Werten."),
  qa("Warum werden PyTorch-Pickles nicht ausgefuehrt?","Aus Sicherheitsgruenden: Pickle kann Code ausfuehren; modelstrata scannt nur statisch, z.B. per Opcode-Disassembly."),
  qa("Was ist d_model?","d_model ist die Breite des Residual-Streams bzw. die Vektordimension pro Token."),
  qa("Was bedeutet block_count oder Layerzahl?","Das ist die Anzahl gestapelter Transformer-Bloecke; mehr Layer bedeuten mehr Tiefe und meist mehr Rechenaufwand."),
  qa("Was ist ein Attention-Head?","Ein Head ist ein paralleler Attention-Kanal mit eigenen Query/Key/Value-Projektionen bzw. geteilten K/V bei GQA/MQA."),
  qa("Was ist GQA?","Grouped-Query-Attention: mehrere Query-Heads teilen sich weniger Key/Value-Heads, was den KV-Cache kleiner macht."),
  qa("Was ist MQA?","Multi-Query-Attention: alle Query-Heads teilen ein einziges Key/Value-Head-Paar; das minimiert den KV-Cache."),
  qa("Was ist MHA?","Multi-Head-Attention: jeder Query-Head hat eigene Key/Value-Heads; klassischer, aber speicherintensiver."),
  qa("Was ist der KV-Cache?","Gespeicherte Key/Value-Aktivierungen frueherer Tokens bei der Generierung; er spart Rechenarbeit, kostet aber Speicher."),
  qa("Warum ist der angezeigte KV-Cache eine obere Schranke?","Die Formel nimmt oft volle globale Attention und fp16 an; MLA oder Sliding-Window koennen real weniger brauchen."),
  qa("Was ist RoPE?","Rotary Position Embedding kodiert Positionen durch Rotationen in Query/Key-Vektoren, sodass relative Positionen im Attention-Produkt sichtbar werden."),
  qa("Garantiert eine hohe RoPE freq_base gute Langkontextleistung?","Nein. Kontextlaenge, RoPE-Scaling, Training und Evaluation zaehlen ebenfalls."),
  qa("Was bedeutet Kontextlaenge?","Die konfigurierte maximale Sequenzlaenge in Tokens, die das Modell adressieren soll."),
  qa("Was ist Sliding-Window-Attention?","Attention wird auf ein lokales Fenster begrenzt, was lange Kontexte billiger macht, aber globale Reichweite reduziert."),
  qa("Was ist MLA?","Multi-head Latent Attention komprimiert Key/Value-Informationen in latente Repraesentationen und kann den Cache stark reduzieren."),
  qa("Was ist ein FFN?","Feed-Forward Network bzw. MLP-Teil eines Transformer-Blocks; verarbeitet jeden Token positionsweise."),
  qa("Was ist SwiGLU?","Ein gegateter MLP mit gate/up/down-Projektionen; heute verbreitet in Llama/Qwen/Gemma-artigen Modellen."),
  qa("Was ist RMSNorm?","Root-Mean-Square-Normalisierung ohne Mittelwertabzug; sie stabilisiert Aktivierungen mit einem gelernten Gain."),
  qa("Was bedeutet tied embeddings?","Input-Embedding und Output-Projektion teilen dieselbe Matrix, was Parameter spart."),
  qa("Was ist der LM-Head?","Die Ausgabeprojektion, die den finalen d_model-Vektor auf Vokabular-Logits abbildet."),
  qa("Was sind Logits?","Rohscores vor Softmax, ein Wert je moeglichem naechstem Token."),
  qa("Was ist der Residual-Stream?","Der laufende Token-Vektor, in den Attention und FFN additiv schreiben und den der naechste Block weiterverarbeitet."),
  qa("Was bedeutet Quantisierung?","Gewichte werden mit weniger Bits gespeichert, z.B. 4 statt 16, und spaeter aus Skalen/Stufen rekonstruiert."),
  qa("Was bedeutet bits/weight?","Ganzdatei-Mittel: Datenbytes mal 8 geteilt durch Parameterzahl; es enthaelt auch unquantisierte Norms, Embeddings oder Tower."),
  qa("Ist Abweichung zwischen file_type und bits/weight automatisch ein Fehler?","Nein. Embeddings, Norms und multimodale Tower koennen absichtlich hoeherpraezise sein."),
  qa("Was sind K-Quants wie Q4_K oder Q6_K?","llama.cpp-Quantisierungsfamilien mit block-/superblockweiser Speicherung und unterschiedlichen Bitbreiten."),
  qa("Was bedeutet Q4_K_M?","Ein gemischtes Quant-Rezept; sensible Tensoren koennen mit hoeherer Praezision gespeichert sein als einfache Q4-Anteile."),
  qa("Warum sind Norms oft F32/F16?","Norm-Gewichte sind klein und sensitiv; hohe Praezision kostet wenig Speicher und hilft Stabilitaet."),
  qa("Warum sind Embeddings oft hoeherpraezise?","Embeddings und Output-Head sind gross, aber sensitiv fuer Tokenverhalten; viele Quant-Rezepte behandeln sie vorsichtiger."),
  qa("Was ist Dequantisierung?","Rueckrechnung quantisierter Bytes in Floatwerte, meist fp32, damit Statistiken berechnet werden koennen."),
  qa("Was ist eine Importance-Matrix?","Kalibrierbasierte Gewichtung fuer Quantisierung, damit wichtigere Gewichte bei niedrigen Bits genauer bleiben."),
  qa("Was ist Perplexity?","exp(mittlerer negativer Log-Likelihood) auf einem Testtext; niedriger ist nur bei gleichem Testset/Setup meist bessere Next-Token-Vorhersage."),
  qa("Was ist Tokenisierung?","Zerlegung von Text in Token-IDs, die das Modell als Eingabe verarbeitet."),
  qa("Was ist ein Vokabular im Tokenizer?","Die Menge aller Token-Strings bzw. Token-IDs, die der Tokenizer kennt."),
  qa("Was sind BPE-Merges?","Gelernte Byte-Pair-Encoding-Regeln, die haeufige Paare zu laengeren Tokens zusammenfuehren."),
  qa("Was ist SentencePiece?","Ein sprachunabhaengiger Tokenizer, der Text inklusive Leerzeichen als Unicode-Strom behandelt."),
  qa("Was ist Byte-Fallback?","Bei aktivem Byte-Fallback werden sonst unbekannte Zeichen in UTF-8-Byte-Tokens zerlegt, damit jeder Text kodierbar bleibt."),
  qa("Was sind Special Tokens?","Steuer-Tokens wie BOS, EOS, PAD oder UNK, die Anfang, Ende, Padding oder unbekannte Eingaben markieren."),
  qa("Was ist BOS?","Begin-of-sequence: ein Token, das den Anfang einer Sequenz oder eines Prompts markiert."),
  qa("Was ist EOS?","End-of-sequence: ein Token, das Ende oder Stoppsignal einer Sequenz markiert."),
  qa("Was ist PAD?","Padding-Token, um Sequenzen in einem Batch auf gleiche Laenge zu bringen."),
  qa("Was ist UNK?","Unknown-Token fuer nicht kodierbare Eingaben, falls kein Byte-Fallback greift."),
  qa("Was sind reservierte Tokens?","Platzhalter wie <unusedN>, die im Vokab existieren und je nach Modell reserviert, unbenutzt oder schwach trainiert sein koennen."),
  qa("Sind reservierte Tokens automatisch Glitch-Tokens?","Nein. Sie sind Kandidaten; staerkere Hinweise liefern Embedding-Normen und Verhaltenstests."),
  qa("Was ist ein Glitch-Token?","Ein Token mit auffaelligem Verhalten; statisch sind anomale Embedding-Normen oder reserved/unused-Typen nur Hinweise, keine Beweise."),
  qa("Warum ist ein Token-Name kein Beweis fuer einen Glitch?","Namen wie <unused42> sind Heuristiken; Belege brauchen Normen, Kontext und spaeter Verhalten."),
  qa("Was ist Skript-Abdeckung?","Welche Schriftsysteme im Vokab sichtbar sind, z.B. Latein, CJK, Kyrillisch oder Kana."),
  qa("Was misst Jaccard?","Mengen-Overlap: |A geschnitten B| geteilt durch |A vereinigt B|; 1.0 bedeutet gleiche Menge, 0.0 keine gemeinsamen Elemente."),
  qa("Was bedeutet Vokab-Jaccard 1.0?","Die Token-Strings sind identisch, aber Gewichte, IDs, Finetuning oder Verhalten koennen trotzdem verschieden sein."),
  qa("Was ist same-id Ratio?","Anteil gemeinsamer Token, die in zwei Tokenizern dieselbe ID haben."),
  qa("Warum ist same-id wichtig?","Gleiche Token-Strings mit anderen IDs koennen Prompts und Embeddings inkompatibel machen."),
  qa("Was ist ein Chat-Template?","Eine Jinja-Vorlage, die System/User/Assistant-Nachrichten in das genaue Promptformat des Modells rendert."),
  qa("Warum ist ein falsches Chat-Template schlimm?","Das Modell sieht dann ein anderes Format als im Training und antwortet oft schlechter oder falsch."),
  qa("Was ist Chat-Template-Lint?","Statische Warnungen fuer typische Template-Probleme wie fehlende Rollenmarker oder doppelte BOS-Erzeugung."),
  qa("Was bedeutet moegliche doppelte BOS-Erzeugung?","Tokenizer und Template koennten beide ein BOS einfuegen; der Prompt startet dann mit doppeltem Anfangstoken."),
  qa("Was ist Standardabweichung bei Gewichten?","Streuung der Werte in einem Tensor; extrem kleine oder grosse Werte koennen Hinweise geben."),
  qa("Was ist die L2-Norm?","Euklidische Laenge aller Tensorwerte, also die Wurzel aus der Summe der Quadrate."),
  qa("Was ist die Frobenius-Norm?","Die Wurzel aus der Summe der Quadrate aller Matrixeintraege, also die Laenge der Matrix als flacher Vektor."),
  qa("Was ist Sparsity?","Anteil der Gewichte nahe Null; auf dequantisierten Werten ist das nur eine grobe Naeherung."),
  qa("Was ist Kurtosis?","Mass fuer schwere Verteilungstails; hohe Werte koennen auf wenige starke Ausreisser hindeuten."),
  qa("Was sind Outlier-Channels?","Eingangskanaele mit deutlich groesserer Norm als der Median; sie koennen fuer Quantisierung kritisch sein."),
  qa("Was sind Massive Activations?","Wenige Aktivierungsdimensionen werden im Forward-Pass sehr gross; statisch sieht man nur indirekte Hinweise."),
  qa("Was ist SVD?","Singulaerwertzerlegung: W = U Sigma V^T, Grundlage fuer Spektralmetriken."),
  qa("Was sind Singulaerwerte?","Staerken der Matrix entlang orthogonaler Richtungen; ihre Verteilung zeigt Rang und Spektrum."),
  qa("Was ist Stable Rank?","Frobenius-Norm quadratisch geteilt durch groessten Singulaerwert quadratisch; zeigt effektive Modenzahl grob an."),
  qa("Was ist effektiver Rang?","Aus der Entropie der Singulaerwertverteilung abgeleitete Anzahl effektiv genutzter Dimensionen."),
  qa("Was ist alpha in der Spektralanalyse?","Ein geschaetzter Heavy-Tail-Exponent des Gewichtsspektrums, hier als einfache Diagnose."),
  qa("Darf alpha allein Modellqualitaet beweisen?","Nein. Alpha ist datenfrei und heuristisch; Evaluation und Kontext bleiben notwendig."),
  qa("Was ist WeightWatcher?","Eine Diagnosemethode fuer Spektralmetriken aus Gewichtsmatrizen ohne Trainings- oder Testdaten."),
  qa("Implementiert modelstrata die volle WeightWatcher-Pipeline?","Nein. Es nutzt eine kleine SVD/Hill-Variante zur statischen Exploration."),
  qa("Was ist PCA?","Projektion hochdimensionaler Daten auf Richtungen groesster Varianz, hier fuer 2D-Embedding-Plots."),
  qa("Was bedeutet evr bei PCA?","Explained variance ratio: Anteil der Varianz, den eine PCA-Achse erklaert."),
  qa("Was ist Embedding-Anisotropie?","Mass dafuer, ob Token-Embeddings in einem engen Kegel liegen statt gut verteilt zu sein."),
  qa("Was bedeutet Cosine-Similarity?","Winkel-Aehnlichkeit zweier Vektoren; 1 = gleiche Richtung, 0 = orthogonal, -1 = entgegengesetzte Richtung."),
  qa("Was ist Model-Diff delta?","Relative Frobenius-Differenz zweier gleichnamiger Tensoren."),
  qa("Was bedeutet niedriger Diff-Cosine?","Die Richtung eines Gewichtstensors unterscheidet sich staerker zwischen zwei Modellen; Quantisierung oder Formatunterschiede muessen mitbedacht werden."),
  qa("Was ist Finetuning?","Weitertrainieren eines Basismodells auf speziellerem Daten- oder Instruktionsmaterial."),
  qa("Was ist ein Base-Modell?","Ein meist nur auf Next-Token-Vorhersage vortrainiertes Modell ohne Chat-Posttraining."),
  qa("Was ist ein Instruct-Modell?","Ein Modell, das nach dem Pretraining auf Anweisungsfolgen und Chatverhalten post-trainiert wurde."),
  qa("Was ist Distillation?","Ein Student-Modell lernt, Ausgaben, Wahrscheinlichkeitsverteilungen oder Antwortmuster eines Teacher-Modells nachzuahmen."),
  qa("Was ist LoRA?","Low-Rank Adaptation: kleine trainierte Rangkorrekturen werden statt voller Gewichte gelernt und ggf. gemerged."),
  qa("Was ist Abliteration?","Gezieltes Abschwaechen einer identifizierten Richtung, z.B. Refusal-Richtung, in Gewichten oder Aktivierungen."),
  qa("Warum ist Abliteration sicherheitsrelevant?","Sie kann Refusal- und Safety-Verhalten veraendern; statische Diffs zeigen nur Hinweise, keine vollstaendige Risikoanalyse."),
  qa("Was ist ein Health-Score?","Heuristische Zusammenfassung statischer Fehler, Warnungen und Hinweise pro Modell."),
  qa("Ist der Health-Score ein Qualitaetslabel?","Nein. Er priorisiert Review; er beweist weder Modellqualitaet noch Sicherheit."),
  qa("Was sind Config-Tensor-Invarianten?","Plausibilitaetsregeln zwischen Metadaten und Tensoren, z.B. Layer-Luecken, Rollen und Shapes."),
  qa("Was ist Tokenizer-Embedding-Konsistenz?","Abgleich von Vokabular/Special-IDs mit Embedding- und Output-Head-Shapes."),
  qa("Was ist ein echter Tokenizer-Embedding-Fehler?","Wenn z.B. das Vokabular mehr IDs hat als Embedding-Zeilen oder Special-IDs ausserhalb des Bereichs liegen."),
  qa("Was ist Quant-Diagnostik?","Pruefung, ob sensible Rollen wie Norms, Embeddings, Output, attn_v oder ffn_down aggressiv quantisiert sind."),
  qa("Was ist Metadata-Audit?","Pruefung, ob wichtige Angaben wie Name, Architektur, Lizenz, Sprachen oder Herkunft fehlen."),
  qa("Was ist MoE?","Mixture-of-Experts: ein Router waehlt pro Token wenige Expert-FFNs aus vielen Experten."),
  qa("Was prueft die MoE-Diagnostik?","Router/Gate, Expert-IDs, geteilte Experten und Luecken in Expertennummern."),
  qa("Was prueft die Multimodal-Diagnostik?","Ob Vision-/Audio-Tower, Projektoren und multimodale Flags zueinander passen."),
  qa("Was ist ein Lineage-Score?","Heuristische Naehe zweier Modelle aus Architektur, Tensor-, Tokenizer-, Prompt- und Quant-Signalen."),
  qa("Beweist ein hoher Lineage-Score Abstammung?","Nein. Er ist ein Hinweis; Hashes, Modellkarten und Herkunftsdaten bleiben massgeblich.")
];
let TOUR_ON=false, QUIZ_ON=false, TOUR_I=0, QUIZ_I=0;
function tx(o){return LANG==="en"?o.en:o.de;}
function clearTourFocus(){
  document.querySelectorAll(".tour-focus").forEach(el=>el.classList.remove("tour-focus"));
}
function tutorialDemoHtml(){
  return [
    ["Modell","Strata-Tutor-7B"],["Arch","llama"],["Params","7.2B"],
    ["Quant","Q4_K_M"],["Kontext","8K"],["GQA","4×"]
  ].map(([k,v])=>`<div><span style="color:var(--faint)">${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}
function renderTutorialStep(){
  if(!TOUR_ON) return;
  const s=TUTORIAL_STEPS[TOUR_I];
  clearTourFocus();
  document.getElementById("tour-fade").classList.add("on");
  document.getElementById("tour-panel").classList.add("on");
  document.getElementById("tour-title").textContent=tx(s.title);
  document.getElementById("tour-text").textContent=tx(s.text);
  document.getElementById("tour-step").textContent=`${TOUR_I+1} / ${TUTORIAL_STEPS.length}`;
  document.getElementById("tour-demo").innerHTML=tutorialDemoHtml();
  const qb=document.getElementById("tour-quiz");
  if(s.quiz){
    qb.classList.add("on");
    qb.innerHTML=`<div class="q">${esc(tx(s.quiz.q))}</div><button class="tour-btn" id="tour-reveal" type="button">${LANG==="en"?"Show answer":"Antwort zeigen"}</button><div class="a" id="tour-answer-text">${esc(tx(s.quiz.a))}</div>`;
    document.getElementById("tour-answer").style.display="";
    document.getElementById("tour-answer").onclick=()=>document.getElementById("tour-answer-text").classList.add("on");
    document.getElementById("tour-reveal").onclick=()=>document.getElementById("tour-answer-text").classList.add("on");
  } else {
    qb.classList.remove("on");
    qb.innerHTML="";
    document.getElementById("tour-answer").style.display="none";
  }
  document.getElementById("tour-prev").disabled=TOUR_I===0;
  document.getElementById("tour-prev").style.opacity=TOUR_I===0?".45":"1";
  document.getElementById("tour-next").textContent=TOUR_I===TUTORIAL_STEPS.length-1
    ? (LANG==="en"?"Finish":"Fertig") : (LANG==="en"?"Next":"Weiter");
  document.getElementById("tour-exit").textContent=LANG==="en"?"Close":"Schliessen";
  document.getElementById("tour-prev").textContent=LANG==="en"?"Back":"Zurueck";
  document.getElementById("tour-answer").textContent=LANG==="en"?"Answer":"Antwort";
  if(s.demo){
    openModelModal(TUTORIAL_MODEL);
  } else {
    document.getElementById("modal").classList.remove("on");
    if(s.tab) setTab(s.tab);
  }
  window.setTimeout(()=>{
    const target=document.querySelector(s.target);
    if(target){
      target.classList.add("tour-focus");
      target.scrollIntoView({behavior:"smooth",block:"center",inline:"nearest"});
    }
  },80);
}
function renderQuizQuestion(){
  if(!QUIZ_ON) return;
  clearTourFocus();
  const item=QUIZ_BANK[QUIZ_I];
  document.getElementById("modal").classList.remove("on");
  document.getElementById("tour-fade").classList.add("on");
  document.getElementById("tour-panel").classList.add("on");
  document.getElementById("tour-title").textContent=LANG==="en"?"Quiz":"Quiz";
  document.getElementById("tour-text").textContent=LANG==="en"
    ?"Answer modelstrata learning questions. Show the answer, then continue."
    :"Beantworte Lernfragen zu Modellen, Begriffen und statischer Analyse. Antwort anzeigen, dann weiter.";
  document.getElementById("tour-step").textContent=`${QUIZ_I+1} / ${QUIZ_BANK.length}`;
  document.getElementById("tour-demo").innerHTML=[
    [LANG==="en"?"Questions":"Fragen",QUIZ_BANK.length],
    [LANG==="en"?"Topic":"Thema",LANG==="en"?"Models":"Modelle"],
    [LANG==="en"?"Mode":"Modus",LANG==="en"?"Learning":"Lernen"]
  ].map(([k,v])=>`<div><span style="color:var(--faint)">${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
  const qb=document.getElementById("tour-quiz");
  qb.classList.add("on");
  qb.innerHTML=`<div class="q">${esc(tx(item.q))}</div><button class="tour-btn" id="tour-reveal" type="button">${LANG==="en"?"Show answer":"Antwort zeigen"}</button><div class="a" id="tour-answer-text">${esc(tx(item.a))}</div>`;
  document.getElementById("tour-answer").style.display="";
  document.getElementById("tour-answer").onclick=()=>document.getElementById("tour-answer-text").classList.add("on");
  document.getElementById("tour-reveal").onclick=()=>document.getElementById("tour-answer-text").classList.add("on");
  document.getElementById("tour-prev").disabled=QUIZ_I===0;
  document.getElementById("tour-prev").style.opacity=QUIZ_I===0?".45":"1";
  document.getElementById("tour-next").textContent=QUIZ_I===QUIZ_BANK.length-1
    ? (LANG==="en"?"Finish":"Fertig") : (LANG==="en"?"Next":"Weiter");
  document.getElementById("tour-exit").textContent=LANG==="en"?"Close":"Schliessen";
  document.getElementById("tour-prev").textContent=LANG==="en"?"Back":"Zurueck";
  document.getElementById("tour-answer").textContent=LANG==="en"?"Answer":"Antwort";
}
function startTutorial(){QUIZ_ON=false;TOUR_ON=true;TOUR_I=0;renderTutorialStep();}
function startQuiz(){TOUR_ON=false;QUIZ_ON=true;QUIZ_I=0;renderQuizQuestion();}
function stopTutorial(){
  TOUR_ON=false; QUIZ_ON=false; clearTourFocus();
  document.getElementById("tour-panel").classList.remove("on");
  document.getElementById("tour-fade").classList.remove("on");
  document.getElementById("modal").classList.remove("on");
}
document.getElementById("tutorial-start").onclick=startTutorial;
document.getElementById("quiz-start").onclick=startQuiz;
document.getElementById("tour-exit").onclick=stopTutorial;
document.getElementById("tour-prev").onclick=()=>{
  if(QUIZ_ON){if(QUIZ_I>0){QUIZ_I--;renderQuizQuestion();}return;}
  if(TOUR_I>0){TOUR_I--;renderTutorialStep();}
};
document.getElementById("tour-next").onclick=()=>{
  if(QUIZ_ON){
    if(QUIZ_I>=QUIZ_BANK.length-1){stopTutorial();return;}
    QUIZ_I++; renderQuizQuestion(); return;
  }
  if(TOUR_I>=TUTORIAL_STEPS.length-1){stopTutorial();return;}
  TOUR_I++; renderTutorialStep();
};

// ---- Ebene 7: diff note + top-changed list (heatmap via statGrid above) ---
function renderDiffExtra(){
  const note=document.getElementById("df-note"), topEl=document.getElementById("df-top");
  if(!DIFF||!DIFF.length){note.innerHTML="";return;}
  const d=DIFF[0];
  if(d.note) note.innerHTML=`<div class="kv v warn" style="font-size:11px;border:1px solid var(--line);padding:8px">⚠ ${esc(d.note)}</div>`;
  if(d.top&&d.top.length){
    topEl.innerHTML=`<b style="color:var(--ink)">${tr("Am stärksten veränderte Tensoren (Top")} ${d.top.length}):</b> `+
      d.top.map(t=>`${esc(t.name)} <span style="color:var(--amber)">Δ${t.delta}</span>/<span style="color:var(--cyan)">cos ${t.cosine}</span>`).join(" · ");
  }
}

// ---- Ebene 6: embedding geometry (PCA scatter + norm histogram) -----------
function embControls(){
  const ctl=document.getElementById("emb-controls");
  if(!EMBED||!EMBED.length){
    ctl.innerHTML=`<span style='color:var(--faint);font-size:11px'>${tr("keine Daten geladen (mit")} --embedding ${tr("bauen)")}</span>`;return;
  }
  ctl.innerHTML=`<label style="font-size:11px;color:var(--dim)">${tr("Modell")} <select id="emb-model" class="chip">`+
    EMBED.map((e,i)=>`<option value="${i}">${esc(e.label)}</option>`).join("")+`</select></label>`;
  document.getElementById("emb-model").onchange=()=>renderEmbed(+document.getElementById("emb-model").value);
  renderEmbed(0);
}
function renderEmbed(i){
  const e=EMBED[i];
  document.getElementById("emb-stats").innerHTML=[
    kvrow("vocab","Vokabular",fmtN(e.vocab)),
    kvrow("","Embedding-Dim",e.dim),
    kvrow("anisotropy","Anisotropie",e.anisotropy),
    kvrow("","Norm Ø / min / max",`${e.norm_mean} / ${e.norm_min} / ${e.norm_max}`),
    kvrow("reserved","Tokens mit ~0-Norm",e.near_zero, e.near_zero>0?"warn":"good"),
  ].join("");
  if(!e.pca||!e.histogram||!e.low_norm){
    document.getElementById("emb-evr").textContent=tr("keine Embedding-Daten");
    document.getElementById("emb-scatter").innerHTML="";
    document.getElementById("emb-hist").innerHTML="";
    document.getElementById("emb-low").innerHTML=
      `<span style="color:var(--faint)">${tr("no token_embd.weight → embedding analysis skipped")}</span>`;
    return;
  }
  // PCA scatter
  const pts=e.pca.points, W=480,H=360,pad=24;
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const X=v=>pad+(v-xmin)/((xmax-xmin)||1)*(W-2*pad), Y=v=>H-pad-(v-ymin)/((ymax-ymin)||1)*(H-2*pad);
  const nmin=Math.min(...pts.map(p=>p.norm)),nmax=Math.max(...pts.map(p=>p.norm));
  let s=pts.map(p=>`<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="1.7" fill="${heat((p.norm-nmin)/((nmax-nmin)||1))}" fill-opacity=".7" data-t="${esc(p.tok)}" data-n="${p.norm}"/>`).join("");
  const sc=document.getElementById("emb-scatter"); sc.innerHTML=s;
  document.getElementById("emb-evr").textContent=`${tr("erklärte Varianz")}: PC1 ${(e.pca.evr[0]*100).toFixed(1)}% · PC2 ${(e.pca.evr[1]*100).toFixed(1)}% · ${pts.length} ${tr("Tokens (Farbe=Norm)")}`;
  sc.querySelectorAll("circle").forEach(c=>{
    c.onmousemove=ev=>{tip.style.display="block";tip.style.whiteSpace="nowrap";tip.style.maxWidth="none";
      tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY+12)+"px";
      tip.innerHTML=`<b>${esc(JSON.stringify(c.dataset.t))}</b><br>norm ${c.dataset.n}`;};
    c.onmouseleave=hideTip;
  });
  // norm histogram
  const h=e.histogram, cnt=h.counts, ed=h.edges, n=cnt.length, cmax=Math.max(...cnt,1);
  const bw=(W-2*pad)/n;
  let hs="";
  for(let k=0;k<n;k++){const bh=(cnt[k]/cmax)*(H-2*pad);
    hs+=`<rect x="${(pad+k*bw).toFixed(1)}" y="${(H-pad-bh).toFixed(1)}" width="${(bw-1).toFixed(1)}" height="${bh.toFixed(1)}" fill="${heat(k/n)}" data-c="${cnt[k]}" data-e="${ed[k]}"/>`;}
  hs+=`<text class="axislbl" x="${pad}" y="${H-8}">${ed[0]}</text><text class="axislbl" x="${W-pad}" y="${H-8}" text-anchor="end">${ed[n]}</text>`;
  document.getElementById("emb-hist").innerHTML=hs;
  // low-norm tokens
  document.getElementById("emb-low").innerHTML=`<b style="color:var(--ink)">${tr("Niedrigste Normen (Glitch-Hinweise):")}</b> `+
    e.low_norm.slice(0,24).map(x=>`<span style="color:var(--amber)">${esc(JSON.stringify(x.tok))}</span>=${x.norm}`).join(" · ");
}

// ---- generic per-model metric heatmap (Ebene 4 weight-stats & 5 spectral) -
function heat(t){ // 0..1 -> dark -> phosphor -> amber -> red
  if(!isFinite(t)) t=0;
  t=Math.max(0,Math.min(1,t));
  const stops=[[10,14,13],[74,222,128],[255,181,71],[255,93,93]];
  const seg=t*(stops.length-1), i=Math.floor(seg), f=seg-i;
  const a=stops[i], b=stops[Math.min(i+1,stops.length-1)];
  return `rgb(${a.map((x,k)=>Math.round(x+(b[k]-x)*f)).join(",")})`;
}
function statGrid(DATA, ctlId, wrapId, flag){
  const ctl=document.getElementById(ctlId), wrap=document.getElementById(wrapId);
  if(!DATA||!DATA.length){
    ctl.innerHTML=`<span style='color:var(--faint);font-size:11px'>${tr("keine Daten geladen (mit")} ${flag} ${tr("bauen)")}</span>`;
    wrap.innerHTML=""; return;
  }
  const ms=DATA.map((w,i)=>`<option value="${i}">${esc(w.label)}</option>`).join("");
  const mets=(DATA[0].metrics||[]).map(m=>`<option value="${m}">${m}</option>`).join("");
  ctl.innerHTML=`<label style="font-size:11px;color:var(--dim)">${tr("Modell")} <select class="chip sg-model">${ms}</select></label>`+
    `<label style="font-size:11px;color:var(--dim)">${tr("Metrik")} <select class="chip sg-metric">${mets}</select></label>`+
    `<span class="sg-range" style="font-size:10px;color:var(--faint)"></span>`;
  const selM=ctl.querySelector(".sg-model"), selK=ctl.querySelector(".sg-metric"),
        rng=ctl.querySelector(".sg-range");
  const draw=()=>{
    const w=DATA[+selM.value], metric=selK.value, cells=w.cells, roles=w.roles, L=w.layers;
    let lo=Infinity, hi=-Infinity;
    for(const r of roles) for(const l in cells[r]){const v=cells[r][l][metric];
      if(v<lo)lo=v; if(v>hi)hi=v;}
    rng.textContent=isFinite(lo)?`${tr("Bereich")} ${lo.toFixed(4)} … ${hi.toFixed(4)}`:"";
    const norm=v=>hi>lo?(v-lo)/(hi-lo):0.5;
    let th=`<tr><th></th>`+roles.map(r=>`<th class="colh">${esc(r)}</th>`).join("")+`</tr>`;
    let body="";
    for(let l=0;l<L;l++){
      body+=`<tr><th>blk.${l}</th>`+roles.map(r=>{
        const s=(cells[r]||{})[l];
        if(!s||s[metric]==null) return `<td style="background:#0a0e0d"></td>`;
        const v=s[metric];
        return `<td style="background:${heat(norm(v))}" data-v="${v}" data-r="${esc(r)}" data-l="${l}"></td>`;
      }).join("")+`</tr>`;
    }
    wrap.innerHTML=`<table class="hm">${th}${body}</table>`;
    wrap.querySelectorAll("td[data-v]").forEach(td=>{
      td.onmousemove=e=>{tip.style.display="block";tip.style.whiteSpace="nowrap";tip.style.maxWidth="none";
        tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
        tip.innerHTML=`blk.${td.dataset.l}.${esc(td.dataset.r)}<br><b>${metric} = ${(+td.dataset.v).toPrecision(4)}</b>`;};
      td.onmouseleave=hideTip;
    });
  };
  selM.onchange=draw; selK.onchange=draw; draw();
}

// ---- vocab-overlap (Jaccard) heatmap over the unique models --------------
function jcolor(j){ // 0 -> dark, 1 -> phosphor
  const t=Math.max(0,Math.min(1,j));
  const a=[10,14,13], b=[74,222,128];
  const c=a.map((x,i)=>Math.round(x+(b[i]-x)*t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function renderOverlap(){
  const wrap=document.getElementById("overlap-wrap");
  if(!FORENSICS||!FORENSICS.overlap){wrap.innerHTML="<div style='color:var(--faint);font-size:11px'>keine Forensik-Daten (mit --forensics bauen)</div>";return;}
  const ov=FORENSICS.overlap;
  // map each unique model -> index in the forensics overlap labels
  const rows=MODELS.map((m,mi)=>{const fl=forLabel(m);return fl?{mi,name:(m.name||m.label),oi:ov.labels.indexOf(fl)}:null;})
    .filter(r=>r&&r.oi>=0);
  let th=`<tr><th></th>`+rows.map(r=>`<th class="colh">${esc(r.name.slice(0,22))}</th>`).join("")+`</tr>`;
  let body="";
  for(const ra of rows){
    body+=`<tr><th style="text-align:right;color:var(--dim);padding-right:6px">${esc(ra.name.slice(0,26))}</th>`+
      rows.map(rb=>{const j=ov.jaccard[ra.oi][rb.oi];
        return `<td style="background:${jcolor(j)};color:${j>0.5?'#06120c':'#6f8279'}" data-mi="${ra.mi}" data-j="${j}" data-a="${esc(ra.name)}" data-b="${esc(rb.name)}" title="${esc(ra.name)} ∩ ${esc(rb.name)} = ${j}">${j>=0.995?'■':(j>=0.1?Math.round(j*100):'')}</td>`;
      }).join("")+`</tr>`;
  }
  wrap.innerHTML=`<table class="hm" style="font-size:9px">${th}${body}</table>`;
  wrap.querySelectorAll("td[data-mi]").forEach(td=>{
    td.style.width="22px";td.style.height="22px";td.style.textAlign="center";td.style.cursor="pointer";
    td.onmousemove=e=>{tip.style.display="block";tip.style.whiteSpace="nowrap";tip.style.maxWidth="none";
      tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px";
      tip.innerHTML=`${esc(td.dataset.a)}<br>∩ ${esc(td.dataset.b)}<br><b>Jaccard ${td.dataset.j}</b>`;};
    td.onmouseleave=hideTip;
    td.onclick=()=>openModal(+td.dataset.mi);
  });
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    main()
