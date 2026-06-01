<p align="center">
  <img src="images/logo.png" alt="modelstrata — static, no-forward-pass LLM dissection" width="660">
</p>

<p align="center"><em>Deutsch · <a href="README.en.md">English</a></em></p>

<p align="center">
  <a href="https://c0decave.github.io/modelstrata/"><b>🔗 Live-Demo</b></a> — interaktives Dashboard (synthetische Daten)
</p>

# modelstrata

Statische und (später) dynamische Analyse von LLMs — *was steckt in den Modellen, ohne sie für Inferenz laufen zu lassen.* Der Name: das Dashboard legt ein Modell in **Schichten** (strata) frei — Header, Tokenizer, Gewichte, Spektrum, Embedding, Diff.

![Fleet-Übersicht des modelstrata-Dashboards](images/01-fleet.png)

Die Tools lesen **GGUF** *und* **HF-safetensors** (gespeicherte Float-Gewichte exakt gelesen) — lokal **oder** auf einem remote Host — plus einen **Inventar-Pfad** für PyTorch `.bin`/`.pth` und ONNX (nur Header/Tensor-Namen, keine Gewichts-Mathematik). Daraus entstehen JSON-Reports und **ein einziges, self-contained HTML-Dashboard ohne externe Assets** — **ohne Forward-Pass**, ohne GPU. Ein format-agnostisches `modelsource`-Paket vereinheitlicht alle Formate auf das GGUF-Schema, sodass die Gewichts-Ebenen (4–7) unverändert auf safetensors laufen.

## 🔬 Was modelstrata aus einem Modell herausholt

Mehrere Dashboard-Ansichten und acht statische Analyse-Ebenen — alles von der Platte gelesen, ohne das Modell für Inferenz zu starten:

| | |
|---|---|
| ![Tokenizer-Tab: Vokab-Overlap-Heatmap](images/02-tokenizer.png) | ![Gewichte-Tab: Statistik- und Spektral-Heatmaps](images/03-weights.png) |
| **Tokenizer** — Vokab-Overlap (Jaccard) deckt gemeinsame Tokenizer-Herkunft / Finetune-Verwandtschaft auf | **Gewichte** — Per-Layer-Statistik (Ebene 4) & Spektral-/WeightWatcher-Analyse (Ebene 5) als Heatmap |
| ![Embedding & Diff-Tab: PCA, Norm-Histogramm, Modell-Diff](images/04-embedding-diff.png) | ![Log-Tab: filterbare Degradations-Tabelle](images/05-log.png) |
| **Embedding & Diff** — PCA, Norm-Histogramm, Glitch-Token-Hinweise + Per-Tensor-Diff (wo hat ein Finetune verändert?) | **Log** — jede Degradation/Approximation transparent aus `run_log.json`, filterbar |

<sub>Alle Screenshots stammen aus **synthetischen Demo-Modellen** (keine echten Gewichte/Daten) — daher die generischen `demo-*`-Namen.</sub>

## Installation

Voraussetzung: **Python ≥ 3.11** (für `tomllib`; getestet mit 3.13/3.14). Git zum Klonen.

```bash
git clone <repo-url> modelstrata && cd modelstrata

# Ebenen 1–3 (Header, Tokenizer, Dashboard) brauchen NUR die Python-stdlib —
# nichts installieren, läuft sofort:
python3 tools/analyze.py --scan /pfad/zu/ggufs --out report

# Ebenen 4–7 brauchen numpy. GGUF-Gewichte brauchen zusätzlich das `gguf`-Paket;
# HF-safetensors brauchen NUR numpy (bf16 wird selbst dekodiert, kein torch):
python3 -m venv .venv && . .venv/bin/activate
pip install numpy gguf
```

Host-Regel: Auf Analyse-/Modell-Hosts wird kein Node.js, npm, Playwright oder
Browser-Driver ausgeführt. Dashboard-Build und Test-Suite laufen über Python;
die Testsuite (`tests/run.sh`) braucht nur numpy.

## Tools

| Tool | Zweck | Deps |
|---|---|---|
| `tools/gguf_inspect.py` | GGUF-**Header**-Parser (nur Header, nie Gewichtsdaten): Architektur, alle Metadaten, Tensor-Verzeichnis, Quant-Map, rohe Header-Bytes, bits/weight, Alignment | stdlib |
| `tools/tokenizer_forensics.py` | Volles Vokabular pro Modell: Token-Typen, reservierte vs. funktionale Special-Tokens, Skript-Abdeckung, **Vokab-Overlap (Jaccard)** zwischen Modellen; läuft auf GGUF, Ollama-Scans und expliziten HF-Tokenizer-Dirs | stdlib |
| `tools/build_dashboard.py` | Verdichtet die JSON-Reports zu einem eigenständigen, offline-fähigen **HTML-Dashboard** ohne externe Assets (Tabs, Plots, Heatmaps, Header-Dump, Glossar mit Suche, **EN/DE-Umschalter**, Tutorial-Modus mit Mini-Checks, Quiz mit 100 Lernfragen) | stdlib |
| `tools/modelsource/` | **Format-agnostische Quelle**: `detect(path)` wählt ein Backend (GGUF · HF-safetensors · Inventar) und liefert Metadaten im GGUF-Schema + Gewichte unter der kanonischen GGUF-Benennung (`blk.N.<role>.weight`) — so laufen Ebene 4–7 unverändert. safetensors braucht nur numpy; GGUF-Gewichte das `gguf`-Paket; Inventar ist header-only | numpy (+gguf für GGUF) |
| `tools/weight_stats.py` / `spectral.py` / `embedding_geometry.py` / `model_diff.py` | Ebene 4–7: Gewichts-Analysen (GGUF dequant „approx" · safetensors auf gespeicherten Float-Werten) | numpy (+gguf) |
| `tools/static_compare.py` | Statische Cross-Model-Vergleiche und Health-Checks: Architektur-/Kontext-/RoPE-Cluster und -Diffs, Config↔Tensor-Invarianten, Tokenizer↔Embedding-Konsistenz, Chat-Linting, Quant-/Metadata-/MoE-/Multimodal-Diagnosen, Lineage-Score, Layer-Anomalien und Diff-Erklärungen aus echten Reports/Tokenizern | stdlib |
| `tools/interp/` | Phase-2-Grundgerüst: Aktivierungs-Cache-Manifest, Logit-Lens-Projektion und Activation-Patching-Helfer ohne harte torch-Abhängigkeit | numpy |
| `tools/analyze.py` | **Orchestrator** — Verzeichnisse/GGUFs/HF-Modell-Dirs füttern (`--scan`/`--model`/`--hf`), ganze Pipeline + Dashboard in einem Befehl; schreibt `reports/run_log.json` | s.o. |

## Formate & Präzisions-Labels

Jeder Modell-Report trägt einen `source`-Block mit ehrlichem **Präzisions**-Label:

| Format | Umfang | Präzision | Deps |
|---|---|---|---|
| **GGUF** | voll (Header + dequantisierte Gewichte) | `approx` (Quant-Rauschen) | gguf für Gewichte, sonst stdlib |
| **HF safetensors** (Dir mit `config.json` + `*.safetensors` [+ `model.safetensors.index.json` für Sharding]) | voll, gespeicherte Float-Gewichte exakt gelesen: fp32/fp16/**bf16** (selbst dekodiert; nach fp32 für Analysen), Sharding, tied embeddings, Attention-Biases, `tokenizer.json` → dieselbe Tokenizer-Forensik | `exact` (gespeicherte unquantisierte Floats) | nur numpy |
| **PyTorch `.bin`/`.pth`** & **ONNX `.onnx`** (Datei ODER Modell-Dir) | **nur Inventar**: Header/Tensor-Namen, keine Gewichts-Mathematik. Pickles werden per Opcode-Disassembly gescannt und **nie ausgeführt** | `inventory-only` | stdlib |

Gemappte Text-Architekturen für safetensors: **llama / qwen2 / qwen3 / mistral / gemma / gemma2 / gemma3** plus gängige **Qwen-/Mixtral-MoE**-Router/Experten-Tensoren. Multimodale/Conditional-Generation-Varianten bleiben bewusst `inventory-only`, solange Tower und verschachtelte LM-Tensoren nicht vollständig gemappt sind. Unbekannte Arch → `inventory-only` (nie raten). Degradationen werden nie verschluckt — sie landen in `reports/run_log.json` (stderr + Dashboard-**Log-Tab**), und `analyze.py` druckt eine einzeilige Summary (`N ok · M inventory-only · K errors`).

## Dashboard

Gegliedert in **sieben Tabs** (Tab-Wahl wird in `localStorage` gemerkt):

| Tab | Inhalt |
|---|---|
| **Flotte** | KPIs, Fleet-Karten (01) und Architektur-Topologie-Plots (02) |
| **Tokenizer** | Vokab-Overlap-Heatmap (03, Jaccard) |
| **Gewichte** | Gewichts-Statistik (04) und Spektral-Analyse / WeightWatcher (05) |
| **Embedding & Diff** | Embedding-Geometrie (06) und Modell-Diff ★ (07) |
| **Vergleich** | Statische Cross-Model-Vergleiche + Health-Summary: Lineage-Scores, Architektur-/Kontext-/Tensor-/Tokenizer-/Prompt-/Quant-Diffs, Konsistenzdiagnosen, Anomalien und Diff-Erklärungen |
| **Log** | Filterbare Tabelle (Severity / Modell / Stage) aus `run_log.json`, neueste zuerst; eine Severity-Badge im Tab-Header (`Log ⚠ 3`) zeigt Probleme ohne Reinklicken |
| **Glossar** | 101 Begriffe — Kennzahlen **und** Transformer-/LLM-Konzepte, nach 9 Themen gruppiert, mit Live-Suche |

Jede Modell-Karte trägt eine `format · precision`-Badge (`safetensors · exact`, `gguf · approx`, `inventory-only`). Ein Klick auf eine Modell-Karte öffnet das Detail-Modal (Steckbrief, per-Modell-Warnungen, Tokenizer mit ein-/ausklappbaren Special-/UNKNOWN-Token-Listen und Erklärungen, roher Hex/ASCII-Header, Tensor-Heatmap, komplette Metadaten). Jede Kennzahl trägt einen `?`-Hilfe-Tooltip; dieselben Texte stehen ausführlich im Glossar-Tab. Oben rechts schaltet ein Klick die Sprache (DE ⇄ EN) um; daneben starten Tutorial-Modus und Quiz. Das Quiz enthält 100 Fragen mit korrekten Antworten zu Modellen, Begriffen, Metriken und typischen Fehlinterpretationen.

*(Vorschau aller Tabs oben im [Überblick](#-was-modelstrata-aus-einem-modell-herausholt).)*

## Bedienung

### Variante A — nur lokal (GGUFs liegen auf diesem Rechner)

Kein Host nötig. `--scan` auf dein lokales GGUF-Verzeichnis:

```bash
# Ebenen 1–3 (stdlib, ohne venv): nur Header + Tokenizer + Dashboard
python3 tools/analyze.py --scan ~/models --out report

# Ebenen 1–7 (mit Gewichten): venv mit numpy(+gguf), dann --deep.
# --deep läuft auf expliziten --model/--hf-Pfaden UND auf GGUFs aus --scan/--ollama.
. .venv/bin/activate
python tools/analyze.py --scan ~/models --deep \
    --diff BASE.gguf FINETUNE.gguf --out report

# HF-safetensors-Modell (Ebene 4–7 auf gespeicherten Float-Werten, nur numpy nötig):
python tools/analyze.py --hf /pfad/zu/hf-model-dir --deep --out report

# Ergebnis im Browser / VSCode-Preview öffnen:
#   report/dashboard.html
```

### Variante B — remote (GGUFs liegen auf einem anderen Host)

Tools **auf dem Host** laufen lassen (dort liegen die Modelle), Reports herunterladen, Dashboard lokal bauen — oder direkt auf dem Host bauen und nur die self-contained `dashboard.html` holen.

```bash
# auf dem Host (venv ~/.venv mit numpy+gguf):
~/.venv/bin/python tools/analyze.py \
    --ollama /usr/share/ollama/.ollama/models \
    --scan /pfad/zu/modellen --deep \
    --diff BASE.gguf FINETUNE.gguf --out report

# Reports lokal holen und Dashboard bauen:
scp host:~/modelstrata/report/*.json reports/
python3 tools/build_dashboard.py reports/models.json \
  --forensics reports/tokenizer.json --weight-stats reports/weight_stats.json \
  --spectral reports/spectral.json --embedding reports/embedding.json --diff reports/diff.json \
  --compare reports/static_compare.json --run-log reports/run_log.json \
  -o reports/dashboard.html
```

### Einzel-Schritte (statt des Orchestrators)

```bash
# Reports erzeugen (lokal oder auf dem Host)
python3 tools/gguf_inspect.py --scan /pfad/zu/modellen \
  --ollama /usr/share/ollama/.ollama/models --json gguf_report_ollama.json
python3 tools/tokenizer_forensics.py --scan /pfad/zu/modellen -o tokenizer_forensics.json
# Gewichts-Analysen (venv mit numpy; GGUF zusätzlich gguf-Paket)
python tools/weight_stats.py MODELL.gguf /pfad/zu/hf-model-dir -o weight_stats.json
python tools/spectral.py MODELL.gguf -o spectral.json
python tools/embedding_geometry.py MODELL.gguf -o embedding.json
python tools/model_diff.py BASE.gguf FINETUNE.gguf -o diff.json
```

> `build_dashboard.py` ist reine stdlib und läuft überall — Reports werden also typischerweise dort erzeugt, wo die GGUFs liegen, und das HTML lokal (oder auf dem Host) verdichtet.

## Tests

```bash
tests/run.sh          # bzw. cd tests && python3 -m unittest discover -p 'test_*.py'
```

332 Tests (330 bestanden, 2 übersprungen): GGUF-Parser (Magic/Counts/Quant/Alignment/bits-per-weight), `derive()` (GQA-/MLA-KV-Cache, weight-vor-bias-Heatmap, Tower-Trennung, tied embeddings, dedup), Multi-Format-Routing (GGUF, HF-safetensors, sharded safetensors, PyTorch/ONNX-Inventar inkl. Initializer-Shape/Dtype), Run-Log/Dashboard-Badges, No-Node-Host-Policy, Glossar-Abdeckung für Lernbegriffe, Quiz mit 100 Lernfragen, Tokenizer-Forensik (gpt2-Byte-Decode, Skript-Klassifikation, Jaccard, Merge-Regeln, reserved-/special-/UNKNOWN-Aufteilung, HF-Tokenizer), Gewichts-Analysen (Ebene 4–7: std/l2/sparsity/kurtosis/outlier, SVD/alpha/stable_rank, PCA/Norm/Glitch, delta/cosine — inkl. MoE-Experten, NaN- und Edge-Fälle), statische Cross-Model-Vergleiche inkl. Health-Checks, Kontext-/RoPE-Kompatibilität, Tensor-Inventar-Overlap und Chat-/Prompt-Kompatibilität sowie erste Interpretability-Helfer.

## Troubleshooting

| Symptom | Ursache / Lösung |
|---|---|
| Dashboard-Sektion zeigt „keine Daten geladen (mit `--…` bauen)" | Die Ebene wurde nicht erzeugt. Mit `analyze.py --deep` bzw. den Flags `--weight-stats/--spectral/--embedding/--diff` von `build_dashboard.py` nachziehen. |
| `ModuleNotFoundError: numpy` / `gguf` | Nur für Ebenen 4–7 nötig: venv aktivieren und `pip install numpy gguf`. Ebenen 1–3 laufen ohne. |
| torch-Wheels fehlen (z.B. Python 3.14) | Für die **statischen** Ebenen wird torch NICHT gebraucht (nur numpy+gguf). torch erst für die Interpretability-Phase. |
| Spektral-SVD dauert ewig / scheint zu hängen | SVD ist CPU-teuer (≈O(n³)); bei 7B+ Minuten. Im Hintergrund laufen lassen oder `--max-dim` kleiner setzen. |
| Token-Tab: UNKNOWN-Liste fehlt / Hinweis „Forensik neu bauen" | Die Reports stammen aus einer älteren `tokenizer_forensics.py`. Einmal neu erzeugen, dann sind die vollständigen Listen da. |
| Dashboard zeigt im EN-Modus deutschen Text | Neuer UI-String ohne EN-Mapping — Eintrag in `EN` / `I18N_HTML` / `GLOSSARY_EN` ergänzen. |
| Modell-Diff trägt `note: mixed quant` | Base & Finetune haben verschiedene Quant-Level → Quant-Rauschen. Für ein sauberes Diff dieselbe Präzision (idealerweise HF-Gewichte) ziehen. |
| Modell erscheint nur als `inventory-only` | Entweder ein `.bin`/`.pth`/`.onnx` (per Design header-only), ein multimodales/Conditional-Generation-safetensors-Modell oder eine safetensors-Arch, die `modelsource` nicht mappt. Qwen-/Mixtral-MoE-Experten sind gemappt; andere MoE-Layouts können noch `tensor_unmapped` melden. Grund steht im Log-Tab / `run_log.json` (`arch_unmapped`/`tensor_unmapped`). |
| HF-Dir wird nicht erkannt | `detect()` braucht im Dir eine `config.json` **und** mindestens ein `*.safetensors` (oder `model.safetensors.index.json`). Fehlt das, kommt eine `unknown_format`-Warnung. |
| `bits/weight` weicht stark vom `file_type` ab | **Kein** Mislabel: Ganzdatei-Mittel inkl. unquantisierter Embeddings/Norms/Tower (siehe Glossar `bits/weight`). |
| Dashboard-Verifikation | Kein Node.js/Playwright auf Analyse-Hosts. JSON/HTML mit Python validieren und das erzeugte `dashboard.html` lokal im Browser ansehen. |

## Analyse-Ebenen (statisch, ohne Forward-Pass)

- **1 Header / 1b Abgeleitet / 2 Tensor-Verzeichnis** — ✅ im Dashboard.
- **3 Tokenizer-Forensik** — ✅ Overlap-Matrix + Token-Typen + reserved/special/UNKNOWN + Skripte; auch für explizite HF-Tokenizer.
- **4 Gewichts-Statistik** — ✅ `weight_stats.py` (std/l2/sparsity/kurtosis/outlier, dequantisiert).
- **5 Spektral (WeightWatcher)** — ✅ `spectral.py` (alpha/stable_rank/eff_rank via SVD).
- **6 Embedding-Geometrie** — ✅ `embedding_geometry.py` (PCA, Norm-Histogramm, Glitch-Hinweise, Anisotropie).
- **7 Modell-Diff ★** — ✅ `model_diff.py` (per-Tensor delta/cosine, Base↔Finetune). Ebene 4–7 brauchen numpy (GGUF zusätzlich `gguf`); Werte aus dequantisierten GGUF sind „approx"; auf gespeicherten, unquantisierten HF-safetensors-Float-Werten werden sie ohne Quant-Rekonstruktion berechnet (über `modelsource`, gemappte Text-Archs: llama/qwen2/qwen3/mistral/gemma + gängige MoE-Rollen).
- **8 Statische Cross-Model-Vergleiche** — ✅ `static_compare.py` (Architektur-Cluster/-Diff, Kontext-/RoPE-/KV-Cache-Kompatibilität, Config↔Tensor-Invarianten, Tokenizer↔Embedding-Konsistenz, Chat-Template-Linting, Quant-/Metadata-/MoE-/Multimodal-Diagnosen, Health-Summary, Tensor-Inventar-Overlap inkl. Shape-/Dtype-Mismatch, Chat-/Prompt-Kompatibilität aus Template-Markern und Special-Token-IDs, Tokenizer-ID-/Merge-Diff, Quant-Profil-Diff, Lineage-Score, robuste Layer-Anomalien, Diff-Erklärungen, Chat-Template-Marker, MoE-Übersicht, Multimodal-Inventar, Embedding-Seed-Nachbarn und Output-Head-vs-Embedding).

Phase-2-Basis: `tools/interp/` enthält erste, testbare Helfer für Aktivierungs-Caches, Logit-Lens-Projektionen und Activation-Patching. Das ersetzt noch keinen echten Forward-Pass, macht den späteren Interpretability-Track aber reproduzierbarer.

Querschnitt **Phase 1 (Black-box)**: Verhalten/Refusals über ollama, abliterated vs. normal.

## Reports

`reports/models.json`, `reports/tokenizer.json`, `reports/static_compare.json`, `reports/weight_stats.json`, `reports/spectral.json`, `reports/embedding.json`, `reports/diff.json`, `reports/run_log.json` (Degradations-Log → Log-Tab), `reports/dashboard.html`.
