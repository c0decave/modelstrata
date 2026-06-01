#!/usr/bin/env python3
"""
analyze.py — Orchestrator: füttere beliebige Verzeichnisse / Modelle und erhalte
einen kompletten Report-Satz (+ optional das fertige Dashboard) in einem Schritt.

Verkettet die Einzel-Tools:
  gguf_inspect (Ebene 1-2) · tokenizer_forensics (Ebene 3) ·
  weight_stats/spectral/embedding (Ebene 4-6, --deep) · model_diff (Ebene 7, --diff) ·
  static_compare (cross-model static comparisons) · build_dashboard (HTML).

Beispiele:
  # ein beliebiges Verzeichnis inspizieren -> Dashboard
  python analyze.py --scan /pfad/zu/ggufs --out report

  # Ollama-Store + zusätzliche Einzeldatei, mit Gewichts-Analysen
  ~/.venv/bin/python analyze.py --ollama /usr/share/ollama/.ollama/models \
      --model /path/to/models/foo.gguf --deep --out report

  # Base gegen Finetune diffen
  ~/.venv/bin/python analyze.py --scan /path/to/models --diff BASE.gguf FT.gguf --out report

Ebene 4-7 brauchen numpy+gguf (auf dem Host: ~/.venv/bin/python).
Inspect/Forensik/Dashboard laufen mit reiner stdlib.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# analyze.py is the application entry point, so it may import the modelsource
# library directly. The deep tools run as SUBPROCESSES; each builds its own
# RunLog and (Task 15 fix) writes it to a temp ``--run-log`` file that we merge
# back into THE one shared RunLog — so per-tensor degradation (tensor_unmapped,
# bad_offset, dtype_unsupported, shard_missing, …) emitted inside a subprocess
# reaches the authoritative reports/run_log.json. aggregate() then supplies only
# each model's precision badge (via detect()/metadata(), no weight iteration —
# so it does NOT duplicate those merged per-tensor/arch warnings).
sys.path.insert(0, HERE)
from modelsource import detect, RunLog  # noqa: E402
from gguf_inspect import discover_ollama  # noqa: E402


def run(script, *args, rl=None, stage=None):
    """Invoke a sub-tool as a subprocess.

    With ``rl`` (and ``stage``) the call is NON-FATAL: a non-zero exit is
    recorded as one ``tool_failed`` error in the shared RunLog and the batch
    continues — so a deep tool that legitimately bails on an inventory-only
    model (e.g. embedding_geometry has no token_embd.weight) does not abort the
    whole run before reports/run_log.json is written. Without ``rl`` the old
    fatal (``check=True``) behaviour is preserved.

    When ``rl`` is given the caller appends ``--run-log <tmp>`` to ``args``;
    after the subprocess returns (success OR failure) this merges whatever that
    tool managed to write into ``rl`` and deletes the temp file — see
    ``_merge_run_log``.
    """
    cmd = [PY, os.path.join(HERE, script), *map(str, args)]
    print("  $", " ".join(os.path.basename(c) if i == 1 else c
                           for i, c in enumerate(cmd)), file=sys.stderr)
    if rl is None:
        subprocess.run(cmd, check=True)
        return
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        rl.emit("error", model="*", stage=(stage or script),
                code="tool_failed",
                msg=f"{script} exited {cp.returncode} (see its stderr above)")


def _entry_key(e):
    """Identity of a log entry for de-duplication — everything EXCEPT ``ts``
    (subprocess logs leave ts empty; the app stamps it later)."""
    return (e["model"], e["stage"], e["severity"], e["code"], e["msg"])


def _merge_run_log(rl, path):
    """Merge a deep tool's written RunLog (``path``) into ``rl`` and delete it.

    Runs after every deep-tool subprocess (success OR failure): the tool writes
    its RunLog in a try/finally, so even a partial/failed run leaves whatever it
    managed to record — those per-tensor/arch warnings are the AUTHORITATIVE
    degradation view and belong in run_log.json. Missing file (tool didn't get
    far enough to write) is a no-op.

    The deep tools share the SAME modelsource backend, so each emits a
    BYTE-IDENTICAL warning for the same model degradation (e.g. all three log
    one ``arch_unmapped`` for an unmappable model). De-dup on ``_entry_key`` so
    the merged log records each distinct degradation ONCE — the run log stays
    authoritative and complete without N copies per tool."""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            incoming = json.load(fh)
        seen = {_entry_key(e) for e in rl.entries}
        for e in incoming:
            k = _entry_key(e)
            if k in seen:
                continue
            seen.add(k)
            rl.entries.append(e)
    finally:
        os.remove(path)


def synth_hf_entry(path, rl=None):
    """Synthesize a MINIMAL fleet entry for a non-GGUF (HF / inventory) model.

    GGUF models flow through gguf_inspect into models.json with the full
    header-only schema (quant heatmap, hex header, ggml tensor types). HF and
    inventory models have NO gguf header, so they would be invisible in the
    fleet. This builds a minimal entry from ``detect(path).metadata()`` so the
    model appears as a fleet card carrying its ``source`` block — honestly: only
    the fields that exist (name/label, arch, n_layers, quant_breakdown of the
    real safetensors dtypes, the source block). GGUF-only widgets (quant
    heatmap, hex/ascii header, ggml types) have no equivalent here; we leave
    those empty/None so the card renders gracefully (no fabricated gguf data).

    Returns the raw entry dict (in the gguf_inspect schema ``derive()`` consumes)
    or None if the path is not a recognizable model (detect() already logged it).
    """
    name = os.path.basename(os.path.normpath(str(path)))
    src = detect(path, log=rl, model=name)
    if src is None:
        return None
    meta = src.metadata()
    arch = (meta.get("metadata") or {}).get("general.architecture")
    tensors = meta.get("tensors") or []
    # total params from declared shapes when present (honest, header-only).
    total_params = 0
    for t in tensors:
        dims = t.get("dims") or []
        n = 1
        for d in dims:
            n *= d
        total_params += n if dims else 0
    try:
        file_size = os.path.getsize(path) if os.path.isfile(path) else _dir_size(path)
    except OSError:
        file_size = 0
    return {
        # gguf_inspect schema fields derive() needs (gguf-only ones left empty)
        "path": str(path),
        "label": name,
        # convenience top-level fields for the minimal HF card (arch/n_layers are
        # also recomputed by derive() from gguf-style keys; harmless duplicates).
        "name": name,
        "arch": arch,
        # honest config.json values (num_hidden_layers / hidden_size); derive()
        # falls back to these for the card when the gguf block_count key is
        # absent. NOT gguf fabrications (heatmap/hex/bits-per-weight stay empty).
        "n_layers": meta.get("n_layers"),
        "n_heads": meta.get("n_heads"),
        "n_kv_heads": meta.get("n_kv_heads"),
        "hidden": meta.get("hidden"),
        "d_model": meta.get("hidden"),
        "ffn": meta.get("ffn"),
        "ctx": meta.get("ctx"),
        # honest tied-embedding flag from the backend (HF tie_word_embeddings +
        # lm_head presence). derive() recomputes tied via the GGUF naming
        # convention, which is always False for HF names; forwarding the real
        # value lets derive() PREFER it so HF cards show tied truthfully.
        "tied": meta.get("tied"),
        "file_size": file_size,
        "n_tensors": len(tensors),
        "total_params": total_params,
        "gguf_version": None,
        "quant_breakdown": meta.get("quant_breakdown") or {},
        "header_hex": "", "header_ascii": "",
        "header_end": None, "alignment": None,
        "data_start": None, "data_bytes": None, "bits_per_weight": None,
        "file_type_label": None,
        "metadata": meta.get("metadata") or {},
        "tensors": tensors,
        # the source block is the whole point — pass it straight through so the
        # card shows its format · precision badge and the modal its warnings.
        "source": meta.get("source"),
    }


def _dir_size(path):
    """Sum of regular-file sizes under ``path`` (HF model dir on-disk size)."""
    total = 0
    for dp, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def _is_gguf_path(path):
    """True for normal ``*.gguf`` files and extensionless GGUF blobs."""
    if str(path).lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _scan_gguf_targets(scan_dirs):
    """Return ``[(label, path), ...]`` for GGUFs below every scan dir."""
    targets = []
    for d in scan_dirs:
        for dp, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    targets.append((fn, os.path.join(dp, fn)))
    return targets


def _ollama_gguf_targets(ollama_dirs):
    """Return ``[(ollama:tag, blob_path), ...]`` for every Ollama model blob."""
    targets = []
    for d in ollama_dirs:
        for name, blob in sorted(discover_ollama(d).items()):
            targets.append((f"ollama:{name}", blob))
    return targets


def _dedupe_targets(targets):
    """Stable de-duplication by real path while preserving the first label."""
    out = []
    seen = set()
    for label, path in targets:
        key = os.path.realpath(str(path))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, path))
    return out


def _append_synth_entries(non_gguf, fleet, rl):
    """Append a synthesized fleet entry for each non-gguf model to ``fleet``.

    Per-model robust: if one model's ``synth_hf_entry`` raises (e.g. its
    ``detect().metadata()`` blows up), record ONE error in the run log and
    continue with the rest — a single bad model must never abort the whole
    append (no-crash / surface-every-degradation mandate)."""
    for m in non_gguf:
        name = os.path.basename(os.path.normpath(str(m)))
        try:
            entry = synth_hf_entry(m, rl=rl)
        except Exception as e:  # one model's failure must not sink the others
            rl.emit("error", model=name, stage="fleet",
                    code="synth_failed", msg=str(e))
            continue
        if entry is not None:
            fleet.append(entry)


def aggregate(models, rl):
    """Per-model precision badge for the summary — WITHOUT re-emitting warnings.

    Returns a list of ``{"name", "path", "precision"}`` where precision is the
    model's ``source.precision`` (exact / approx / inventory-only) or None if
    the path was unrecognized (detect() already logged one ``unknown_format``).

    precision comes from ``detect(...).metadata()["source"]["precision"]`` ONLY.
    metadata() computes mapped/precision from the model's config + header
    directory WITHOUT iterating weights, so it emits NO per-tensor warnings and
    reads no weight bytes. The arch-level AND per-tensor degradation warnings
    come instead from the deep tools' merged RunLogs (the real, authoritative
    view) — so aggregate() must NOT drive iter_weights() here or every
    arch_unmapped/tensor_unmapped would be DUPLICATED.
    """
    info = []
    seen = {_entry_key(e) for e in rl.entries}
    for path in models:
        name = os.path.basename(os.path.normpath(str(path)))
        # Detect into a side log so the only entry it can add (unknown_format for
        # an unrecognized path) is de-duped against the merged tool logs — the
        # deep tools already ran detect() on this path and may have logged it.
        side = RunLog()
        src = detect(path, log=side, model=name)
        for e in side.entries:
            k = _entry_key(e)
            if k not in seen:
                seen.add(k)
                rl.entries.append(e)
        precision = None
        if src is not None:
            precision = (src.metadata().get("source") or {}).get("precision")
        info.append({"name": name, "path": str(path), "precision": precision})
    return info


def summarize(models, rl):
    """Build the one-line honest degradation summary.

    ``models`` is aggregate()'s list of ``{"name", "precision"}``. A model is
    "ok" when its precision is exact or approx; "inventory-only" when its
    precision is exactly ``inventory-only``. A model whose path could not be
    detected (precision None) is UNRECOGNIZED: it produced only an
    ``unknown_format`` warn (not an error entry), so it is counted under errors
    explicitly here — otherwise it would fall into no bucket and its name would
    vanish from the line.

    The error tally is ``rl.counts()["error"]`` (deep-tool/stage failures) PLUS
    each unrecognized model that does not already own an error entry (so a model
    that both failed to detect AND triggered a tool error is not double-counted).
    Offending model names (inventory-only, then the union of errored stages and
    unrecognized models) are appended so the line is actionable.
    """
    ok = [m for m in models if m["precision"] in ("exact", "approx")]
    inv = [m for m in models if m["precision"] == "inventory-only"]
    unknown = [m for m in models if m["precision"] is None]

    # Names that already carry an error-severity entry in the run log. A deep
    # tool that failed over the whole batch is recorded under model "*"; surface
    # that as the tool/stage name instead of a bare asterisk.
    errored = []
    for e in rl.entries:
        if e["severity"] != "error":
            continue
        label = e["stage"] if e["model"] == "*" else e["model"]
        if label not in errored:
            errored.append(label)

    # Unrecognized models that have no error entry of their own — count them in
    # and list them, without double-counting any already in ``errored``.
    extra_unknown = [m["name"] for m in unknown if m["name"] not in errored]
    errors = rl.counts()["error"] + len(extra_unknown)

    line = (f"{len(ok)} ok (exact/approx) · "
            f"{len(inv)} inventory-only · {errors} errors")
    if inv:
        line += "  | inventory-only: " + ", ".join(m["name"] for m in inv)
    error_names = errored + extra_unknown
    if error_names:
        line += "  | errors: " + ", ".join(error_names)
    return line


def stamp_ts(rl, ts):
    """Fill any empty ``ts`` on run-log entries with the run timestamp. The
    library never reads the clock; the app stamps the single captured time."""
    for e in rl.entries:
        if not e["ts"]:
            e["ts"] = ts


def main():
    ap = argparse.ArgumentParser(description="modelstrata orchestrator")
    ap.add_argument("--scan", action="append", default=[], help="dir to scan for *.gguf")
    ap.add_argument("--ollama", action="append", default=[], help="ollama models dir")
    ap.add_argument("--model", action="append", default=[],
                    help="explicit model: a .gguf file OR an HF model dir "
                         "(config.json + *.safetensors) — both flow through the "
                         "deep tools via modelsource.detect()")
    ap.add_argument("--hf", action="append", default=[],
                    help="explicit HF model dir (alias of --model for an HF dir)")
    ap.add_argument("--out", default="report", help="output directory")
    ap.add_argument("--deep", action="store_true",
                    help="run Ebene 4-6 (weight stats/spectral/embedding) on "
                         "--model/--hf plus GGUFs found via --scan/--ollama")
    ap.add_argument("--diff", nargs=2, metavar=("BASE", "FINETUNE"),
                    help="run Ebene 7 model diff on two model paths (GGUF or "
                         "mappable HF safetensors)")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--max-dim", type=int, default=None, help="spectral SVD size cap")
    args = ap.parse_args()

    # --hf is just an explicit-HF alias of --model; merge so the deep tools see
    # one model list. detect() (used inside the deep tools) picks the backend.
    explicit_models = list(args.model) + list(args.hf)

    if not (args.scan or args.ollama or explicit_models):
        ap.error("nothing to analyze — give --scan, --ollama and/or --model/--hf")
    os.makedirs(args.out, exist_ok=True)
    o = lambda f: os.path.join(args.out, f)

    # ONE RunLog for the whole run. Each deep-tool subprocess writes its own
    # RunLog to a temp ``--run-log`` file we merge back into this one (the
    # authoritative per-tensor + arch degradation view). aggregate() runs LATER
    # (after the tools) to add only each model's precision badge for the summary
    # — it does not iterate weights, so it adds no duplicate warnings. The whole
    # log is written to reports/run_log.json. App entry point reads the clock once.
    rl = RunLog()
    run_ts = datetime.datetime.now().isoformat(timespec="seconds")

    scan_args = []
    for d in args.scan:
        scan_args += ["--scan", d]
    oll_args = []
    for d in args.ollama:
        oll_args += ["--ollama", d]

    # Resolve scan/ollama targets once so --deep can run on the same GGUFs that
    # the header/tokenizer stages see. Ollama blobs are extensionless but carry
    # GGUF magic bytes; modelsource.detect() sniffs those too.
    scan_targets = _scan_gguf_targets(args.scan)
    ollama_targets = _ollama_gguf_targets(args.ollama)

    # Ebene 1-2 is the pure-stdlib GGUF header inspector. HF dirs are surfaced
    # instead by synthesized fleet entries below. Pick out explicit GGUF files by
    # extension OR magic so extensionless blob paths work when passed as --model.
    gguf_models = [m for m in explicit_models if _is_gguf_path(m)]

    # Ebene 1-2: static header inspection (gguf model files + scan + ollama).
    # Skip cleanly when only HF models are given (gguf_inspect needs gguf targets).
    if gguf_models or args.scan or args.ollama:
        run("gguf_inspect.py", *gguf_models, *scan_args, *oll_args, "--json", o("models.json"))
    else:
        # Write an empty models.json so the dashboard step still has its input.
        with open(o("models.json"), "w") as f:
            f.write("[]")
        print("  [skip] Ebene 1-2 (gguf_inspect): no gguf/scan/ollama targets",
              file=sys.stderr)
    # Surface non-GGUF (HF / inventory) models in the fleet: gguf_inspect only
    # handles .gguf, so any --model/--hf that is NOT a .gguf would be invisible.
    # Synthesize a minimal fleet entry (name/arch/n_layers + source block) per
    # such model and APPEND it to models.json so it renders as a (gguf-only
    # widgets empty) card. detect() logs anything unrecognized into rl.
    non_gguf = [m for m in explicit_models if not _is_gguf_path(m)]
    if non_gguf:
        with open(o("models.json")) as f:
            fleet = json.load(f)
        _append_synth_entries(non_gguf, fleet, rl)
        with open(o("models.json"), "w") as f:
            json.dump(fleet, f, indent=2, default=str)

    # Ebene 3: tokenizer forensics (scan/ollama dirs + explicit GGUF/HF models)
    forensics = None
    if args.scan or args.ollama or explicit_models:
        model_tok_args = []
        for m in explicit_models:
            model_tok_args += ["--model", m]
        run("tokenizer_forensics.py", *model_tok_args, *scan_args, *oll_args,
            "-o", o("tokenizer.json"))
        forensics = o("tokenizer.json")

    wstats = spectral = embedding = None
    deep_targets = _dedupe_targets(
        [(None, m) for m in explicit_models] + scan_targets + ollama_targets)
    deep_models = [p for _label, p in deep_targets]
    deep_labels = [label or os.path.basename(os.path.normpath(str(p)))
                   for label, p in deep_targets]

    if args.deep and deep_models:
        # Deep tools source tensors via modelsource.detect() -> they accept BOTH
        # .gguf files and HF dirs in the same invocation. Each is told to write
        # its own RunLog to a temp file we merge back (authoritative per-tensor
        # degradation), then delete — even on a non-zero exit (try/finally in
        # the tool guarantees the file exists).
        for script, outfile, stage in (
                ("weight_stats.py", "weight_stats.json", "weight_stats"),
                ("spectral.py", "spectral.json", "spectral"),
                ("embedding_geometry.py", "embedding.json", "embedding")):
            tmp_log = o(f"_runlog_{stage}.json")
            extra = (["--max-dim", args.max_dim]
                     if stage == "spectral" and args.max_dim else [])
            label_args = []
            for label in deep_labels:
                label_args += ["--label", label]
            run(script, *deep_models, *label_args, *extra, "-o", o(outfile),
                "--run-log", tmp_log, rl=rl, stage=stage)
            _merge_run_log(rl, tmp_log)
        wstats = o("weight_stats.json")
        spectral = o("spectral.json")
        embedding = o("embedding.json")
    elif args.deep:
        print("  [skip] --deep found no model files", file=sys.stderr)

    diff = None
    if args.diff:
        tmp_log = o("_runlog_diff.json")
        run("model_diff.py", args.diff[0], args.diff[1], "-o", o("diff.json"),
            "--run-log", tmp_log, rl=rl, stage="diff")
        _merge_run_log(rl, tmp_log)
        diff = o("diff.json")

    # Cross-model static comparison layer: architecture/tokenizer/quant/profile
    # diffs, lineage scoring, anomaly surfacing and diff explanations. It uses
    # only JSON reports plus direct tokenizer reads from the known model paths.
    compare = o("static_compare.json")
    cmp_model_args = []
    for label, path in deep_targets:
        cmp_model_args += ["--model", label or os.path.basename(os.path.normpath(str(path))), path]
    cmp_opts = []
    for flag, val in [("--forensics", forensics), ("--weight-stats", wstats),
                      ("--spectral", spectral), ("--embedding", embedding),
                      ("--diff", diff)]:
        if val:
            cmp_opts += [flag, val]
    run("static_compare.py", o("models.json"), *cmp_model_args, *cmp_opts,
        "-o", compare, rl=rl, stage="static_compare")
    if not os.path.exists(compare):
        compare = None

    # After the tools' logs are merged, add each model's precision badge (and the
    # detect-stage unknown_format for any unrecognized path) for the summary.
    summary_models = deep_models if args.deep else explicit_models
    model_info = aggregate(summary_models, rl) if summary_models else []

    # The run log is now COMPLETE (merged deep-tool degradations + aggregate's
    # precision/unknown_format). Stamp the single run timestamp and write the ONE
    # authoritative run_log.json BEFORE building the dashboard, so the dashboard
    # can load & render it (its Log tab). This is the single write — nothing
    # after it overwrites the file, so it is never left empty/truncated.
    run_log_path = o("run_log.json")
    stamp_ts(rl, run_ts)
    rl.write(run_log_path)

    if not args.no_dashboard:
        opts = []
        for flag, val in [("--forensics", forensics), ("--weight-stats", wstats),
                          ("--spectral", spectral), ("--embedding", embedding),
                          ("--diff", diff), ("--compare", compare)]:
            if val:
                opts += [flag, val]
        # Pass --run-log explicitly so build_dashboard renders the entries we just
        # wrote (don't rely on its auto-load + step ordering).
        run("build_dashboard.py", o("models.json"), *opts,
            "--run-log", run_log_path, "-o", o("dashboard.html"))
        print(f"\n[done] {o('dashboard.html')}")
    else:
        print(f"\n[done] reports in {args.out}/")

    # Print the honest degradation summary (the run log is already on disk).
    print(summarize(model_info, rl), file=sys.stderr)


if __name__ == "__main__":
    main()
