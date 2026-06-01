#!/usr/bin/env python3
"""Static cross-model comparisons built from existing modelstrata reports.

This tool does not run a model. It combines:

* fleet metadata from ``models.json`` / ``derive()``
* optional tokenizer data loaded directly from model paths
* optional weight/spectral/embedding/diff JSON reports

The output is intentionally conservative: every section is computed only from
available data. Missing tokenizer paths or missing deep reports simply produce
smaller sections, not invented scores.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_dashboard  # noqa: E402
from tokenizer_forensics import analyze as analyze_tokenizer  # noqa: E402
from tokenizer_forensics import read_tokenizer_any  # noqa: E402


ARCH_KEYS = [
    "arch", "n_layers", "d_model", "n_head", "n_head_kv", "gqa", "ffn",
    "ffn_ratio", "ctx", "vocab", "tokenizer_model", "merges_n",
    "tied_embeddings", "rope_freq_base", "rope_scaling_type",
    "sliding_window",
]

BLK_RE = re.compile(
    r"^(?P<pre>(?:[a-z]+\.)*)blk\.(?P<layer>\d+)\.(?P<role>.+?)\.(?P<suf>weight|bias)$")

CHAT_MARKERS = {
    "system": ("system", "<|system|>", "[SYSTEM]"),
    "user": ("user", "<|user|>", "[INST]", "<start_of_turn>user"),
    "assistant": ("assistant", "<|assistant|>", "[/INST]", "<start_of_turn>model"),
    "tool": ("tool", "tools", "tool_call", "function"),
    "image": ("image", "<|image|>", "[IMG]", "vision"),
    "thinking": ("think", "reasoning", "/think", "/nothink"),
}

SPECIAL_KEYS = ("bos", "eos", "pad", "unk", "add_bos", "add_eos")
EXACTISH_TYPES = {"F32", "F16", "BF16"}
LOW_RISK_QUANTS = EXACTISH_TYPES | {"Q8_0", "Q8_K", "Q6_K"}
SENSITIVE_ROLES = {
    "token_embd", "output", "output_norm", "attn_norm", "ffn_norm",
    "attn_q", "attn_k", "attn_v", "attn_output", "ffn_down",
}


def load_json(path, default):
    if not path:
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dedupe_models(raw):
    derived = [build_dashboard.derive(x) for x in raw]
    seen = {}
    for d, src in zip(derived, raw):
        d["_tensors"] = src.get("tensors") or []
        fp = build_dashboard.fingerprint(d)
        if fp in seen:
            seen[fp]["aliases"].append(d["label"])
        else:
            d["aliases"] = []
            seen[fp] = d
    return sorted(seen.values(), key=lambda x: -(x["params"] or 0))


def model_key(m):
    return m.get("label") or m.get("name") or os.path.basename(m.get("path", ""))


def _ratio(num, den):
    return round(num / den, 4) if den else 0.0


def _chat_markers(tmpl):
    low = tmpl.lower()
    return {k: any(x.lower() in low for x in xs) for k, xs in CHAT_MARKERS.items()}


def architecture_summary(models):
    return [
        {
            "model": model_key(m),
            "label": m["label"],
            "arch": m.get("arch"),
            "layers": m.get("n_layers"),
            "d_model": m.get("d_model"),
            "heads": m.get("n_head"),
            "kv_heads": m.get("n_head_kv"),
            "gqa": m.get("gqa"),
            "ffn": m.get("ffn"),
            "ctx": m.get("ctx"),
            "vocab": m.get("vocab"),
            "source": m.get("source"),
            "moe": bool(m.get("moe")),
            "multimodal": bool(m.get("vision") or m.get("audio")),
        }
        for m in models
    ]


def architecture_pairs(models):
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            diffs = {}
            same = 0
            seen = 0
            for k in ARCH_KEYS:
                av, bv = a.get(k), b.get(k)
                if av is None and bv is None:
                    continue
                seen += 1
                if av == bv:
                    same += 1
                else:
                    diffs[k] = [av, bv]
            compatible = all(a.get(k) == b.get(k)
                             for k in ("arch", "n_layers", "d_model")
                             if a.get(k) is not None and b.get(k) is not None)
            out.append({
                "a": model_key(a), "b": model_key(b),
                "compat_core": compatible,
                "same_ratio": _ratio(same, seen),
                "diffs": diffs,
            })
    return out


def context_profile(m):
    return {
        "model": model_key(m),
        "ctx": m.get("ctx"),
        "kv_cache": m.get("kv_cache"),
        "kv_cache_kind": m.get("kv_cache_kind"),
        "head_dim": m.get("head_dim"),
        "kv_heads": m.get("n_head_kv"),
        "rope_freq_base": m.get("rope_freq_base"),
        "ctx_extended": bool(m.get("ctx_extended")),
        "rope_scaling_type": m.get("rope_scaling_type"),
        "rope_scaling_factor": m.get("rope_scaling_factor"),
        "rope_orig_ctx": m.get("rope_orig_ctx"),
        "sliding_window": m.get("sliding_window"),
    }


def _num_ratio(a, b):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return None
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return None
    den = max(abs(a), abs(b))
    return _ratio(min(abs(a), abs(b)), den) if den else 1.0


def context_pairs(profiles):
    out = []
    names = list(profiles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = profiles[a], profiles[b]
            ctx_ratio = _num_ratio(pa.get("ctx"), pb.get("ctx"))
            kv_ratio = _num_ratio(pa.get("kv_cache"), pb.get("kv_cache"))
            rope_fields = [
                "rope_freq_base", "rope_scaling_type", "rope_scaling_factor",
                "rope_orig_ctx", "sliding_window", "kv_cache_kind",
            ]
            seen = 0
            same = 0
            diffs = {}
            for key in rope_fields:
                av, bv = pa.get(key), pb.get(key)
                if av is None and bv is None:
                    continue
                seen += 1
                if av == bv:
                    same += 1
                else:
                    diffs[key] = [av, bv]
            rope_compat = _ratio(same, seen)
            score_parts = []
            if ctx_ratio is not None:
                score_parts.append(ctx_ratio)
            if kv_ratio is not None:
                score_parts.append(kv_ratio)
            if seen:
                score_parts.append(rope_compat)
            score = round(sum(score_parts) / len(score_parts), 4) if score_parts else 0.0
            notes = []
            if ctx_ratio is not None and ctx_ratio < 1.0:
                notes.append("different configured context length")
            if "rope_freq_base" in diffs or "rope_scaling_type" in diffs:
                notes.append("different RoPE parametrization")
            if "sliding_window" in diffs:
                notes.append("different sliding-window attention")
            if kv_ratio is not None and kv_ratio < 0.8:
                notes.append("substantially different KV-cache footprint")
            out.append({
                "a": a, "b": b,
                "score": score,
                "ctx": [pa.get("ctx"), pb.get("ctx")],
                "ctx_ratio": ctx_ratio,
                "kv_cache": [pa.get("kv_cache"), pb.get("kv_cache")],
                "kv_cache_ratio": kv_ratio,
                "rope_compat": rope_compat,
                "diffs": diffs,
                "notes": notes,
            })
    out.sort(key=lambda p: p["score"])
    return out


def architecture_clusters(models):
    """Group models by a conservative core architecture signature."""
    groups = {}
    for m in models:
        key = tuple(m.get(k) for k in (
            "arch", "n_layers", "d_model", "n_head", "n_head_kv", "ffn",
            "vocab", "tokenizer_model", "tied_embeddings"))
        rec = groups.setdefault(key, {"signature": {}, "models": []})
        rec["models"].append(model_key(m))
        rec["signature"] = {
            "arch": m.get("arch"),
            "layers": m.get("n_layers"),
            "d_model": m.get("d_model"),
            "heads": m.get("n_head"),
            "kv_heads": m.get("n_head_kv"),
            "ffn": m.get("ffn"),
            "vocab": m.get("vocab"),
            "tokenizer_model": m.get("tokenizer_model"),
            "tied_embeddings": m.get("tied_embeddings"),
        }
        rec.setdefault("ctx_values", []).append(m.get("ctx"))
        rec.setdefault("formats", set()).add(((m.get("source") or {}).get("format") or "?"))
        rec.setdefault("dominant_quants", set()).add(m.get("dom_quant") or "?")
    out = []
    for rec in groups.values():
        ctx_nums = [c for c in rec.pop("ctx_values", []) if isinstance(c, (int, float))]
        formats = sorted(rec.pop("formats", set()))
        quants = sorted(rec.pop("dominant_quants", set()))
        rec.update({
            "size": len(rec["models"]),
            "formats": formats,
            "dominant_quants": quants,
            "ctx_range": [min(ctx_nums), max(ctx_nums)] if ctx_nums else None,
        })
        out.append(rec)
    out.sort(key=lambda g: (-g["size"], g["models"][0]))
    return out


def _issue(severity, code, msg, **extra):
    row = {"severity": severity, "code": code, "msg": msg}
    row.update({k: v for k, v in extra.items() if v is not None})
    return row


def _tensor_by_name(m):
    return {t.get("name"): t for t in (m.get("_tensors") or []) if t.get("name")}


def _dims(t):
    return list(t.get("dims") or []) if t else []


def _matches_shape(dims, expected):
    if not dims or not expected or len(dims) != len(expected):
        return False
    return all(e is None or d == e for d, e in zip(dims, expected))


def _matches_any_shape(dims, expected_shapes):
    return any(_matches_shape(dims, exp) for exp in expected_shapes)


def _embedding_rows(dims, d_model):
    if len(dims) != 2:
        return None
    if d_model in dims:
        return dims[1] if dims[0] == d_model else dims[0]
    return max(dims)


def tensor_config_checks(models):
    out = []
    for m in models:
        issues = []
        tensors = _tensor_by_name(m)
        d_model = m.get("d_model")
        ffn = m.get("ffn")
        n_layers = m.get("n_layers")
        layer_seen = set()
        role_layers = {}
        for name, t in tensors.items():
            mo = BLK_RE.match(name)
            if not mo:
                continue
            layer = int(mo.group("layer"))
            role = mo.group("pre") + mo.group("role")
            layer_seen.add(layer)
            role_layers.setdefault(role, set()).add(layer)
            dims = _dims(t)
            if mo.group("suf") != "weight" or len(dims) != 2:
                continue
            expected = None
            if role in ("attn_q", "attn_output"):
                expected = [[d_model, d_model]]
            elif role in ("ffn_gate", "ffn_up"):
                expected = [[ffn, d_model], [d_model, ffn]]
            elif role == "ffn_down":
                expected = [[d_model, ffn], [ffn, d_model]]
            if expected and d_model and not _matches_any_shape(dims, expected):
                issues.append(_issue(
                    "warn", "shape_invariant",
                    f"{name} shape {dims} does not match expected architecture dimensions",
                    tensor=name, dims=dims))
        if isinstance(n_layers, int) and n_layers > 0:
            missing = sorted(set(range(n_layers)) - layer_seen)
            extra = sorted(l for l in layer_seen if l >= n_layers)
            if missing:
                issues.append(_issue(
                    "error", "missing_layers",
                    f"{len(missing)} layer(s) from config are absent in tensor directory",
                    layers=missing[:16]))
            if extra:
                issues.append(_issue(
                    "error", "extra_layers",
                    f"{len(extra)} tensor layer(s) exceed configured layer count",
                    layers=extra[:16]))
        required_roles = ["attn_q", "attn_output", "ffn_down"]
        for role in required_roles:
            if n_layers and role in role_layers and len(role_layers[role]) < n_layers:
                missing = sorted(set(range(n_layers)) - role_layers[role])
                issues.append(_issue(
                    "warn", "role_layer_gap",
                    f"{role} is missing on {len(missing)} configured layer(s)",
                    role=role, layers=missing[:16]))
        out.append({"model": model_key(m), "issues": issues})
    return out


def tokenizer_embedding_checks(models):
    out = []
    for m in models:
        issues = []
        tensors = _tensor_by_name(m)
        emb = tensors.get("token_embd.weight") or tensors.get("model.embed_tokens.weight")
        outw = tensors.get("output.weight") or tensors.get("lm_head.weight")
        emb_dims = _dims(emb)
        rows = _embedding_rows(emb_dims, m.get("d_model"))
        vocab = m.get("vocab")
        if emb is None:
            issues.append(_issue("warn", "missing_embedding", "token embedding tensor not found"))
        elif isinstance(vocab, int) and rows is not None and rows != vocab:
            issues.append(_issue(
                "error", "vocab_embedding_mismatch",
                f"vocab size {vocab} does not match embedding rows {rows}",
                vocab=vocab, embedding_rows=rows))
        for key, value in (m.get("special") or {}).items():
            if key not in ("bos", "eos", "pad", "unk"):
                continue
            if isinstance(value, int) and rows is not None and not (0 <= value < rows):
                issues.append(_issue(
                    "error", "special_id_oob",
                    f"{key} token id {value} is outside embedding rows {rows}",
                    token=key, token_id=value, embedding_rows=rows))
        if outw is not None and emb is not None and m.get("tied_embeddings"):
            if _dims(outw) != emb_dims:
                issues.append(_issue(
                    "warn", "tied_shape_mismatch",
                    "model is marked tied but output.weight shape differs from token_embd.weight",
                    embedding=emb_dims, output=_dims(outw)))
        if outw is None and m.get("tied_embeddings") is False:
            issues.append(_issue(
                "warn", "untied_output_missing",
                "model is marked untied but no output.weight/lm_head.weight tensor was found"))
        out.append({"model": model_key(m), "issues": issues})
    return out


def chat_template_lints(models):
    out = []
    for m in models:
        issues = []
        tmpl = (m.get("meta") or {}).get("tokenizer.chat_template")
        markers = _chat_markers(tmpl) if isinstance(tmpl, str) else {}
        family = None
        if isinstance(tmpl, str) and tmpl:
            low = tmpl.lower()
            if "<|im_start|>" in tmpl:
                family = "chatml"
            elif "<|start_header_id|>" in tmpl:
                family = "llama3-header"
            elif "[inst]" in low:
                family = "mistral-inst"
            elif "<start_of_turn>" in tmpl:
                family = "gemma-turn"
            if not markers.get("user"):
                issues.append(_issue("warn", "chat_missing_user", "chat template has no recognizable user marker"))
            if not markers.get("assistant"):
                issues.append(_issue("warn", "chat_missing_assistant", "chat template has no recognizable assistant marker"))
            if (m.get("special") or {}).get("add_bos") and ("bos_token" in tmpl or "<s>" in tmpl):
                issues.append(_issue(
                    "warn", "possible_double_bos",
                    "template references BOS while tokenizer metadata also enables add_bos"))
            if markers.get("tool") and "tool_call" not in low and "function" not in low:
                issues.append(_issue(
                    "info", "tool_marker_weak",
                    "template mentions tools but no explicit tool_call/function marker was recognized"))
        else:
            issues.append(_issue("info", "missing_chat_template", "no tokenizer.chat_template present"))
        out.append({
            "model": model_key(m),
            "family": family,
            "markers": markers,
            "issues": issues,
        })
    return out


def quant_diagnostics(models):
    out = []
    for m in models:
        issues = []
        cells = (m.get("grid") or {}).get("cells") or {}
        globals_t = (m.get("grid") or {}).get("globals") or {}
        for name, qtype in globals_t.items():
            if name in ("token_embd.weight", "output.weight") and qtype not in LOW_RISK_QUANTS:
                issues.append(_issue(
                    "warn", "sensitive_global_quant",
                    f"{name} uses aggressive quantization {qtype}",
                    tensor=name, qtype=qtype))
            if name.endswith("norm.weight") and qtype not in EXACTISH_TYPES:
                issues.append(_issue(
                    "warn", "norm_quantized",
                    f"{name} normalization tensor is not stored as F32/F16/BF16",
                    tensor=name, qtype=qtype))
        for role, layers in cells.items():
            if role not in SENSITIVE_ROLES:
                continue
            counts = {}
            for qtype in layers.values():
                counts[qtype] = counts.get(qtype, 0) + 1
            risky = sum(c for q, c in counts.items() if q not in LOW_RISK_QUANTS)
            total = sum(counts.values())
            if total and risky / total >= 0.5:
                issues.append(_issue(
                    "warn", "sensitive_role_aggressive_quant",
                    f"{role} is mostly aggressive quantization",
                    role=role, risky=risky, total=total, breakdown=counts))
        out.append({
            "model": model_key(m),
            "dominant": m.get("dom_quant"),
            "bits_per_weight": m.get("bits_per_weight"),
            "issues": issues,
        })
    return out


def metadata_audit(models):
    out = []
    for m in models:
        meta = m.get("meta") or {}
        issues = []
        for key in ("general.name", "general.architecture"):
            if not meta.get(key):
                issues.append(_issue("warn", "metadata_missing", f"{key} is missing", key=key))
        if not meta.get("general.license"):
            issues.append(_issue("info", "license_missing", "general.license is missing"))
        if not meta.get("general.languages"):
            issues.append(_issue("info", "languages_missing", "general.languages is missing"))
        src = m.get("source") or {}
        if src.get("warnings"):
            issues.append(_issue(
                "warn", "source_warnings",
                f"source reported {len(src.get('warnings') or [])} warning(s)",
                warnings=src.get("warnings")))
        out.append({"model": model_key(m), "issues": issues})
    return out


def moe_diagnostics(models):
    out = []
    for m in models:
        roles = set((m.get("grid") or {}).get("roles") or [])
        expert_roles = sorted(r for r in roles if "_exps.e" in r)
        shared_roles = sorted(r for r in roles if "_shared" in r)
        gate_roles = sorted(r for r in roles if "gate_inp" in r)
        issues = []
        if m.get("moe") and not expert_roles:
            issues.append(_issue("warn", "moe_no_experts", "model metadata indicates MoE but no expert roles were mapped"))
        if expert_roles:
            role_prefixes = {r.split("_exps.e", 1)[0] for r in expert_roles}
            for prefix in sorted(role_prefixes):
                ids = []
                for role in expert_roles:
                    if not role.startswith(prefix + "_exps.e"):
                        continue
                    tail = role.rsplit("_exps.e", 1)[-1]
                    if tail.isdigit():
                        ids.append(int(tail))
                if ids:
                    missing = sorted(set(range(max(ids) + 1)) - set(ids))
                    if missing:
                        issues.append(_issue(
                            "warn", "moe_expert_gap",
                            f"{prefix} expert ids are not contiguous",
                            role=prefix, missing=missing[:16]))
        out.append({
            "model": model_key(m),
            "expert_roles": expert_roles,
            "shared_roles": shared_roles,
            "gate_roles": gate_roles,
            "issues": issues,
        })
    return out


def multimodal_diagnostics(models):
    out = []
    for m in models:
        roles = set((m.get("grid") or {}).get("roles") or [])
        globals_t = set(((m.get("grid") or {}).get("globals") or {}).keys())
        tower_roles = sorted(r for r in roles if r.startswith(("v.", "mm.", "a.")))
        tower_globals = sorted(g for g in globals_t if g.startswith(("v.", "mm.", "vision", "a.")))
        issues = []
        if (m.get("vision") or m.get("audio")) and not (tower_roles or tower_globals):
            issues.append(_issue("warn", "multimodal_no_tower", "model is marked multimodal but no tower tensors were recognized"))
        if tower_roles and not m.get("vision") and not m.get("audio"):
            issues.append(_issue("info", "tower_without_flag", "tower-prefixed tensors exist but model was not marked multimodal"))
        out.append({
            "model": model_key(m),
            "tower_roles": tower_roles,
            "tower_globals": tower_globals,
            "issues": issues,
        })
    return out


def health_summary(*sections):
    by_model = {}
    for section_name, rows in sections:
        for row in rows or []:
            model = row.get("model")
            if not model:
                continue
            rec = by_model.setdefault(model, {
                "model": model,
                "score": 100,
                "errors": 0,
                "warnings": 0,
                "infos": 0,
                "top_issues": [],
            })
            for issue in row.get("issues") or []:
                sev = issue.get("severity")
                if sev == "error":
                    rec["errors"] += 1
                    rec["score"] -= 20
                elif sev == "warn":
                    rec["warnings"] += 1
                    rec["score"] -= 8
                else:
                    rec["infos"] += 1
                    rec["score"] -= 2
                if len(rec["top_issues"]) < 8:
                    rec["top_issues"].append({
                        "section": section_name,
                        "severity": sev,
                        "code": issue.get("code"),
                        "msg": issue.get("msg"),
                    })
    out = []
    for rec in by_model.values():
        rec["score"] = max(0, min(100, rec["score"]))
        if rec["errors"]:
            rec["class"] = "attention needed"
        elif rec["warnings"]:
            rec["class"] = "warnings"
        else:
            rec["class"] = "clean"
        out.append(rec)
    out.sort(key=lambda r: (r["score"], -r["errors"], -r["warnings"], r["model"]))
    return out


def _tensor_role(name):
    m = BLK_RE.match(name)
    if m:
        return m.group("pre") + m.group("role")
    return name


def tensor_inventory(m):
    tensors = m.get("_tensors") or []
    by_name = {}
    roles = {}
    for t in tensors:
        name = t.get("name")
        if not name:
            continue
        role = _tensor_role(name)
        by_name[name] = {
            "shape": tuple(t.get("dims") or ()),
            "type": t.get("type"),
            "params": t.get("params") or 0,
            "role": role,
        }
        roles[role] = roles.get(role, 0) + 1
    return {
        "model": model_key(m),
        "names": set(by_name),
        "roles": set(roles),
        "by_name": by_name,
        "role_counts": roles,
        "total_tensors": len(by_name),
    }


def tensor_pairs(models):
    invs = [tensor_inventory(m) for m in models]
    out = []
    for i, a in enumerate(invs):
        for b in invs[i + 1:]:
            shared_names = a["names"] & b["names"]
            shared_roles = a["roles"] & b["roles"]
            name_union = a["names"] | b["names"]
            role_union = a["roles"] | b["roles"]
            same_shape = 0
            same_type = 0
            shape_mismatch = []
            type_mismatch = []
            for name in sorted(shared_names):
                ta, tb = a["by_name"][name], b["by_name"][name]
                if ta["shape"] == tb["shape"]:
                    same_shape += 1
                else:
                    shape_mismatch.append({
                        "name": name,
                        "shape": [list(ta["shape"]), list(tb["shape"])],
                    })
                if ta["type"] == tb["type"]:
                    same_type += 1
                else:
                    type_mismatch.append({"name": name, "type": [ta["type"], tb["type"]]})
            only_a = sorted(a["names"] - b["names"],
                            key=lambda n: -a["by_name"][n]["params"])[:8]
            only_b = sorted(b["names"] - a["names"],
                            key=lambda n: -b["by_name"][n]["params"])[:8]
            out.append({
                "a": a["model"], "b": b["model"],
                "name_jaccard": _ratio(len(shared_names), len(name_union)),
                "role_jaccard": _ratio(len(shared_roles), len(role_union)),
                "shared_names": len(shared_names),
                "shared_roles": len(shared_roles),
                "tensor_count": [a["total_tensors"], b["total_tensors"]],
                "same_shape_shared_ratio": _ratio(same_shape, len(shared_names)),
                "same_type_shared_ratio": _ratio(same_type, len(shared_names)),
                "shape_mismatch_count": len(shape_mismatch),
                "type_mismatch_count": len(type_mismatch),
                "shape_mismatch_sample": shape_mismatch[:8],
                "type_mismatch_sample": type_mismatch[:8],
                "only_a_sample": only_a,
                "only_b_sample": only_b,
            })
    out.sort(key=lambda p: (-p["name_jaccard"], -p["role_jaccard"]))
    return out


def quant_profile(m):
    q = m.get("quant") or {}
    total = sum(q.values()) or 0
    cells = (m.get("grid") or {}).get("cells") or {}
    role_majority = {}
    for role, layers in cells.items():
        counts = {}
        for qt in layers.values():
            counts[qt] = counts.get(qt, 0) + 1
        if counts:
            role_majority[role] = max(counts.items(), key=lambda kv: kv[1])[0]
    sensitive = ["attn_v", "attn_output", "ffn_down", "token_embd", "output"]
    high = sum(c for t, c in q.items() if t in ("F32", "F16", "BF16", "Q8_0", "Q8_K", "Q6_K"))
    return {
        "model": model_key(m),
        "dominant": m.get("dom_quant"),
        "file_type": m.get("file_type_label"),
        "bits_per_weight": m.get("bits_per_weight"),
        "breakdown": q,
        "high_precisionish_ratio": _ratio(high, total),
        "role_majority": role_majority,
        "sensitive_roles": {r: role_majority.get(r) for r in sensitive if r in role_majority},
    }


def quant_pairs(profiles):
    out = []
    by = {p["model"]: p for p in profiles}
    names = list(by)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            qa, qb = by[a]["breakdown"], by[b]["breakdown"]
            keys = set(qa) | set(qb)
            inter = sum(min(qa.get(k, 0), qb.get(k, 0)) for k in keys)
            union = sum(max(qa.get(k, 0), qb.get(k, 0)) for k in keys)
            changed_roles = {}
            ra, rb = by[a]["role_majority"], by[b]["role_majority"]
            for role in sorted(set(ra) | set(rb)):
                if ra.get(role) != rb.get(role):
                    changed_roles[role] = [ra.get(role), rb.get(role)]
            out.append({
                "a": a, "b": b,
                "breakdown_jaccard": _ratio(inter, union),
                "role_quant_diffs": changed_roles,
            })
    return out


def read_tokenizers(model_args):
    out = {}
    for label, path in model_args:
        tk = read_tokenizer_any(path, label=label)
        if tk is None:
            continue
        tokens = [str(t) for t in tk.get("tokens") or []]
        by_token = {}
        for i, tok in enumerate(tokens):
            by_token.setdefault(tok, i)
        out[label] = {
            "path": path,
            "tokens": tokens,
            "by_token": by_token,
            "merges": [str(m) for m in tk.get("merges") or []],
            "analysis": analyze_tokenizer(tk),
        }
    return out


def tokenizer_pairs(tokenizers):
    names = list(tokenizers)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = tokenizers[a], tokenizers[b]
            sa, sb = set(ta["tokens"]), set(tb["tokens"])
            common = sa & sb
            union = sa | sb
            same_id = sum(1 for tok in common
                          if ta["by_token"].get(tok) == tb["by_token"].get(tok))
            row = {
                "a": a, "b": b,
                "vocab_jaccard": _ratio(len(common), len(union)),
                "shared_tokens": len(common),
                "same_id_shared_ratio": _ratio(same_id, len(common)),
                "same_id_count": same_id,
                "vocab_size": [len(ta["tokens"]), len(tb["tokens"])],
                "reserved_delta": [
                    ta["analysis"]["reserved_count"],
                    tb["analysis"]["reserved_count"],
                ],
                "special_delta": [
                    ta["analysis"]["special_count"],
                    tb["analysis"]["special_count"],
                ],
                "unknown_delta": [
                    ta["analysis"]["unknown_count"],
                    tb["analysis"]["unknown_count"],
                ],
            }
            ma, mb = set(ta.get("merges") or []), set(tb.get("merges") or [])
            if ma or mb:
                shared_merges = len(ma & mb)
                row.update({
                    "merge_jaccard": _ratio(shared_merges, len(ma | mb)),
                    "shared_merges": shared_merges,
                    "merges_n": [len(ta.get("merges") or []), len(tb.get("merges") or [])],
                })
            out.append(row)
    return out


def _report_by_label(report):
    return {r.get("label"): r for r in report or []}


def _cell_values(entry):
    vals = []
    for role, layers in (entry.get("cells") or {}).items():
        for layer, metrics in layers.items():
            for metric, value in (metrics or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vals.append((role, int(layer), metric, float(value)))
    return vals


def metric_anomalies(report, *, limit=8):
    out = []
    for entry in report or []:
        vals = _cell_values(entry)
        by_metric = {}
        for role, layer, metric, value in vals:
            by_metric.setdefault(metric, []).append((role, layer, value))
        hits = []
        for metric, rows in by_metric.items():
            nums = sorted(v for _r, _l, v in rows)
            if len(nums) < 4:
                continue
            med = nums[len(nums) // 2]
            devs = sorted(abs(v - med) for v in nums)
            mad = devs[len(devs) // 2] or 1e-12
            for role, layer, value in rows:
                z = abs(value - med) / mad
                if z >= 6.0:
                    hits.append({
                        "metric": metric, "role": role, "layer": layer,
                        "value": round(value, 6), "robust_z": round(z, 3),
                    })
        hits.sort(key=lambda h: -h["robust_z"])
        out.append({"model": entry.get("label"), "anomalies": hits[:limit]})
    return out


def role_aggregates(report):
    out = []
    for entry in report or []:
        rows = []
        for role, layers in (entry.get("cells") or {}).items():
            vals = {}
            for metrics in (layers or {}).values():
                for metric, value in (metrics or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        vals.setdefault(metric, []).append(float(value))
            if vals:
                rows.append({
                    "role": role,
                    "metrics": {k: round(sum(v) / len(v), 6) for k, v in vals.items()},
                    "n": max(len(v) for v in vals.values()),
                })
        out.append({"model": entry.get("label"), "roles": rows})
    return out


def moe_analysis(weight_stats, spectral):
    out = {}
    for source_name, report in (("weight_stats", weight_stats or []), ("spectral", spectral or [])):
        for entry in report:
            model = entry.get("label")
            for role, layers in (entry.get("cells") or {}).items():
                if "_exps.e" not in role and "_shared" not in role and "gate_inp" not in role:
                    continue
                rec = out.setdefault(model, {"model": model, "sources": {}, "expert_roles": []})
                rec["expert_roles"].append(role)
                vals = []
                for metrics in layers.values():
                    vals.extend(v for v in metrics.values()
                                if isinstance(v, (int, float)) and not isinstance(v, bool))
                if vals:
                    rec["sources"].setdefault(source_name, {})[role] = {
                        "mean": round(sum(vals) / len(vals), 6),
                        "n": len(vals),
                    }
    return list(out.values())


def chat_template_summary(models):
    out = []
    for m in models:
        tmpl = (m.get("meta") or {}).get("tokenizer.chat_template")
        if not isinstance(tmpl, str) or not tmpl:
            continue
        out.append({
            "model": model_key(m),
            "length": len(tmpl),
            "markers": _chat_markers(tmpl),
            "preview": tmpl[:240],
        })
    return out


def _special_similarity(a, b):
    sa, sb = a.get("special") or {}, b.get("special") or {}
    seen = 0
    same = 0
    mismatches = {}
    for key in SPECIAL_KEYS:
        av, bv = sa.get(key), sb.get(key)
        if av is None and bv is None:
            continue
        seen += 1
        if av == bv:
            same += 1
        else:
            mismatches[key] = [av, bv]
    return _ratio(same, seen), mismatches


def chat_template_pairs(models):
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ta = (a.get("meta") or {}).get("tokenizer.chat_template")
            tb = (b.get("meta") or {}).get("tokenizer.chat_template")
            ta = ta if isinstance(ta, str) else ""
            tb = tb if isinstance(tb, str) else ""
            ma = _chat_markers(ta) if ta else {}
            mb = _chat_markers(tb) if tb else {}
            set_a = {k for k, v in ma.items() if v}
            set_b = {k for k, v in mb.items() if v}
            marker_jaccard = _ratio(len(set_a & set_b), len(set_a | set_b))
            special_ratio, special_mismatch = _special_similarity(a, b)
            length_ratio = _ratio(min(len(ta), len(tb)), max(len(ta), len(tb)))
            exact = bool(ta and tb and ta == tb)
            score = (
                0.45 * marker_jaccard
                + 0.35 * special_ratio
                + 0.10 * length_ratio
                + 0.10 * (1.0 if exact else 0.0)
            )
            if exact and not special_mismatch:
                klass = "same chat template"
            elif score >= 0.8:
                klass = "likely prompt-compatible"
            elif score >= 0.5:
                klass = "partial prompt compatibility"
            else:
                klass = "weak / unknown prompt compatibility"
            out.append({
                "a": model_key(a), "b": model_key(b),
                "score": round(score, 4),
                "class": klass,
                "exact_template": exact,
                "template_lengths": [len(ta), len(tb)],
                "length_ratio": length_ratio,
                "marker_jaccard": marker_jaccard,
                "markers": [sorted(set_a), sorted(set_b)],
                "shared_markers": sorted(set_a & set_b),
                "missing_template": [
                    model_key(m) for m, tmpl in ((a, ta), (b, tb)) if not tmpl
                ],
                "special_id_same_ratio": special_ratio,
                "special_mismatches": special_mismatch,
            })
    out.sort(key=lambda p: -p["score"])
    return out


def multimodal_inventory(models):
    out = []
    for m in models:
        if not (m.get("vision") or m.get("audio")):
            continue
        roles = (m.get("grid") or {}).get("roles") or []
        out.append({
            "model": model_key(m),
            "vision": bool(m.get("vision")),
            "audio": bool(m.get("audio")),
            "tower_roles": [r for r in roles if r.startswith(("v.", "mm.", "a."))],
            "source": m.get("source"),
        })
    return out


def embedding_summaries(embedding):
    out = []
    for e in embedding or []:
        out.append({
            "model": e.get("label"),
            "seed_neighbors": e.get("seed_neighbors") or [],
            "output_head_compare": e.get("output_head_compare"),
            "near_zero": e.get("near_zero"),
            "anisotropy": e.get("anisotropy"),
        })
    return out


def diff_explanations(diff_report):
    out = []
    for d in diff_report or []:
        role_delta = {}
        layer_delta = {}
        for role, layers in (d.get("cells") or {}).items():
            for layer, metrics in layers.items():
                delta = metrics.get("delta")
                if not isinstance(delta, (int, float)):
                    continue
                role_delta.setdefault(role, []).append(delta)
                layer_delta.setdefault(int(layer), []).append(delta)
        top_roles = sorted(
            [{"role": r, "mean_delta": round(sum(v) / len(v), 6), "n": len(v)}
             for r, v in role_delta.items()],
            key=lambda x: -x["mean_delta"])[:8]
        top_layers = sorted(
            [{"layer": l, "mean_delta": round(sum(v) / len(v), 6), "n": len(v)}
             for l, v in layer_delta.items()],
            key=lambda x: -x["mean_delta"])[:8]
        notes = []
        if d.get("shape_mismatch"):
            notes.append(f"{d['shape_mismatch']} tensors had shape mismatches")
        if d.get("matched") == 0:
            notes.append("no comparable block-weight tensors matched")
        elif top_roles:
            notes.append(f"largest mean delta role: {top_roles[0]['role']}")
        out.append({
            "label": d.get("label"),
            "matched": d.get("matched"),
            "shape_mismatch": d.get("shape_mismatch"),
            "top_roles": top_roles,
            "top_layers": top_layers,
            "top_tensors": d.get("top") or [],
            "notes": notes,
            "source_a": d.get("source_a"),
            "source_b": d.get("source_b"),
        })
    return out


def lineage_pairs(models, arch_pairs, context_pair_rows, tok_pairs, quant_pair_rows,
                  tensor_pair_rows, chat_pair_rows, diff_report):
    arch_by = {(p["a"], p["b"]): p for p in arch_pairs}
    ctx_by = {(p["a"], p["b"]): p for p in context_pair_rows}
    tok_by = {(p["a"], p["b"]): p for p in tok_pairs}
    quant_by = {(p["a"], p["b"]): p for p in quant_pair_rows}
    tensor_by = {(p["a"], p["b"]): p for p in tensor_pair_rows}
    chat_by = {(p["a"], p["b"]): p for p in chat_pair_rows}
    diff_by = {}
    for d in diff_report or []:
        if d.get("label_a") and d.get("label_b") and d.get("matched"):
            cos = []
            for _role, layers in (d.get("cells") or {}).items():
                for metrics in layers.values():
                    c = metrics.get("cosine")
                    if isinstance(c, (int, float)):
                        cos.append(c)
            diff_by[(d["label_a"], d["label_b"])] = sum(cos) / len(cos) if cos else None
    names = [model_key(m) for m in models]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            parts = []
            arch = arch_by.get((a, b))
            if arch:
                parts.append(("architecture", 1.0 if arch["compat_core"] else arch["same_ratio"], 0.25))
            ctx = ctx_by.get((a, b))
            if ctx:
                parts.append(("context_rope", ctx["score"], 0.10))
            tok = tok_by.get((a, b))
            if tok:
                parts.append(("tokenizer_vocab", tok["vocab_jaccard"], 0.25))
                parts.append(("tokenizer_ids", tok["same_id_shared_ratio"], 0.15))
            quant = quant_by.get((a, b))
            if quant:
                parts.append(("quant_profile", quant["breakdown_jaccard"], 0.10))
            tensor = tensor_by.get((a, b))
            if tensor:
                parts.append(("tensor_names", tensor["name_jaccard"], 0.10))
                parts.append(("tensor_shapes", tensor["same_shape_shared_ratio"], 0.10))
            chat = chat_by.get((a, b))
            if chat:
                parts.append(("chat_template", chat["score"], 0.10))
                parts.append(("special_tokens", chat["special_id_same_ratio"], 0.05))
            dcos = diff_by.get((a, b))
            if dcos is not None:
                parts.append(("weight_cosine", max(0.0, min(1.0, dcos)), 0.25))
            weight = sum(w for _n, _v, w in parts)
            score = sum(v * w for _n, v, w in parts) / weight if weight else 0.0
            if score >= 0.9:
                klass = "probable same base / close finetune"
            elif score >= 0.7:
                klass = "same family likely"
            elif score >= 0.45:
                klass = "partial relation"
            else:
                klass = "weak static relation"
            out.append({
                "a": a, "b": b, "score": round(score, 4),
                "class": klass,
                "signals": {n: round(v, 4) for n, v, _w in parts},
            })
    out.sort(key=lambda p: -p["score"])
    return out


def build_report(raw_models, model_args, forensics=None, weight_stats=None,
                 spectral=None, embedding=None, diff=None):
    models = dedupe_models(raw_models)
    arch_pairs = architecture_pairs(models)
    ctx_profiles = {model_key(m): context_profile(m) for m in models}
    ctx_pairs = context_pairs(ctx_profiles)
    q_profiles = [quant_profile(m) for m in models]
    q_pairs = quant_pairs(q_profiles)
    t_pairs = tensor_pairs(models)
    tokenizers = read_tokenizers(model_args)
    tok_pairs = tokenizer_pairs(tokenizers)
    chat_pairs = chat_template_pairs(models)
    tensor_checks = tensor_config_checks(models)
    tok_emb_checks = tokenizer_embedding_checks(models)
    chat_lints = chat_template_lints(models)
    quant_diag = quant_diagnostics(models)
    meta_audit = metadata_audit(models)
    moe_diag = moe_diagnostics(models)
    mm_diag = multimodal_diagnostics(models)
    return {
        "schema": 5,
        "models": architecture_summary(models),
        "architecture_clusters": architecture_clusters(models),
        "architecture_pairs": arch_pairs,
        "context_profiles": list(ctx_profiles.values()),
        "context_pairs": ctx_pairs,
        "quant_profiles": q_profiles,
        "quant_pairs": q_pairs,
        "tensor_pairs": t_pairs,
        "tokenizer_pairs": tok_pairs,
        "chat_template_pairs": chat_pairs,
        "tensor_config_checks": tensor_checks,
        "tokenizer_embedding_checks": tok_emb_checks,
        "chat_template_lints": chat_lints,
        "quant_diagnostics": quant_diag,
        "metadata_audit": meta_audit,
        "moe_diagnostics": moe_diag,
        "multimodal_diagnostics": mm_diag,
        "health_summary": health_summary(
            ("tensor_config", tensor_checks),
            ("tokenizer_embedding", tok_emb_checks),
            ("chat_template", chat_lints),
            ("quant", quant_diag),
            ("metadata", meta_audit),
            ("moe", moe_diag),
            ("multimodal", mm_diag),
        ),
        "lineage_pairs": lineage_pairs(
            models, arch_pairs, ctx_pairs, tok_pairs, q_pairs, t_pairs, chat_pairs, diff or []),
        "anomalies": {
            "weight_stats": metric_anomalies(weight_stats or []),
            "spectral": metric_anomalies(spectral or []),
        },
        "role_aggregates": {
            "weight_stats": role_aggregates(weight_stats or []),
            "spectral": role_aggregates(spectral or []),
        },
        "moe_analysis": moe_analysis(weight_stats or [], spectral or []),
        "chat_templates": chat_template_summary(models),
        "multimodal_inventory": multimodal_inventory(models),
        "embedding_summaries": embedding_summaries(embedding or []),
        "diff_explanations": diff_explanations(diff or []),
        "coverage": {
            "model_count": len(models),
            "tokenizer_models": len(tokenizers),
            "has_forensics": bool(forensics),
            "has_weight_stats": bool(weight_stats),
            "has_spectral": bool(spectral),
            "has_embedding": bool(embedding),
            "has_diff": bool(diff),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="static cross-model comparisons")
    ap.add_argument("models_json")
    ap.add_argument("-o", "--out", default="static_compare.json")
    ap.add_argument("--model", nargs=2, action="append", default=[],
                    metavar=("LABEL", "PATH"),
                    help="model path for tokenizer-aware pair comparisons")
    ap.add_argument("--forensics")
    ap.add_argument("--weight-stats")
    ap.add_argument("--spectral")
    ap.add_argument("--embedding")
    ap.add_argument("--diff")
    args = ap.parse_args()

    raw = load_json(args.models_json, [])
    report = build_report(
        raw,
        args.model,
        forensics=load_json(args.forensics, {}) if args.forensics else {},
        weight_stats=load_json(args.weight_stats, []) if args.weight_stats else [],
        spectral=load_json(args.spectral, []) if args.spectral else [],
        embedding=load_json(args.embedding, []) if args.embedding else [],
        diff=load_json(args.diff, []) if args.diff else [],
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[static_compare: {args.out}  {report['coverage']['model_count']} models]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
