# English-to-Persian PDF Translator — NLLB-200 (CPU + CUDA)

Translate **text-based English PDFs into Persian PDFs** locally using Meta's [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M). One script, `translate_nllb.py`, runs on either **CPU** or an **NVIDIA CUDA GPU**. No translation API key is required. The model downloads on the first online run and can be reused offline.

> **Scope:** This is a text translation and PDF-generation utility, not a layout-preserving PDF editor. It does not perform OCR; images, diagrams, tables, and equations are not reconstructed in the output.

## Features

- English (`eng_Latn`) → Persian (`pes_Arab`) translation
- Select `--device cpu`, `--device cuda`, or `--device auto`
- FP16 inference on CUDA, FP32 on CPU; configurable batch size and CPU threads
- Automatic model download and reuse through the Hugging Face cache
- Chunk-level JSON caching, restart/resume, and progress output
- Persian right-to-left text rendering with a TrueType font (e.g., Vazirmatn)
- Choose a page range or process the entire PDF
- Automatic batch-size reduction after a CUDA out-of-memory error (where possible)

## Requirements

- A compatible Python 3 installation (tested in this project with Python 3.14.8 on CachyOS; library support can vary by version)
- Enough RAM/disk space for PyTorch, Transformers, and the NLLB model
- For GPU mode: NVIDIA GPU, working NVIDIA driver, and a **CUDA-enabled PyTorch** installation
- An installed Persian-capable `.ttf` font
- A **text-based** PDF with extractable text
- Internet access on the first run to download model files

## Installation

### 1. Create and activate a virtual environment

On Linux with Bash or Zsh:

```bash
python -m venv ~/pdf-translator-env
source ~/pdf-translator-env/bin/activate
python -m pip install --upgrade pip
```

On **Fish Shell** (including CachyOS/Arch with Fish):

```fish
python -m venv ~/pdf-translator-env
source ~/pdf-translator-env/bin/activate.fish
python -m pip install --upgrade pip
```

### 2. Install PyTorch

**CPU-only:**

```bash
python -m pip install torch
```

**NVIDIA CUDA:** Choose a PyTorch CUDA wheel compatible with your driver from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For example, the CUDA 12.6 build used for this project's RTX A2000 test was installed with:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

This is an **example**, not a universal command: available CUDA wheel versions can change. Arch Linux and CachyOS do not require a special PyTorch distribution; install inside the virtual environment. A system-wide CUDA Toolkit is usually not necessary to run the prebuilt PyTorch wheel.

Check GPU availability:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

You need `CUDA: True` to use `--device cuda`.

### 3. Install translator dependencies

```bash
python -m pip install numpy transformers sentencepiece safetensors pymupdf reportlab arabic-reshaper python-bidi
```

Verify PDF extraction support:

```bash
python -c "import pymupdf; print('PyMuPDF OK')"
```

### 4. Find a Persian font

On Linux:

```bash
fc-match -f '%{file}\n' Vazirmatn
```

You can install Vazirmatn from your distribution's packages or download it from the [Vazirmatn project](https://github.com/rastikerdar/vazirmatn). The `--font` option must point to an existing **TTF** file.

## Quick start

All commands below assume `translate_nllb.py` and the input PDF are accessible from your current directory. Replace `english_book.pdf` with your actual filename.

### CPU: translate the first 10 pages

```bash
python translate_nllb.py \
  --input english_book.pdf \
  --pages 10 \
  --device cpu \
  --precision fp32 \
  --batch-size 1 \
  --cpu-threads 4 \
  --font /path/to/Vazirmatn-Regular.ttf \
  --output sample_cpu_fa.pdf
```

### CUDA: translate the first 10 pages

```bash
python translate_nllb.py \
  --input english_book.pdf \
  --pages 10 \
  --device cuda \
  --precision fp16 \
  --batch-size 8 \
  --cache gpu_sample_cache.json \
  --output sample_gpu_fa.pdf \
  --font /path/to/Vazirmatn-Regular.ttf
```

**Fish Shell** users can replace the font path with:

```fish
--font (fc-match -f '%{file}' Vazirmatn)
```

### Translate an entire book using CUDA

**Important: `--pages` defaults to `3`, not all pages. Use `--pages 0` to process from the chosen start page through the end of the PDF.**

```fish
python translate_nllb.py \
  --input english_book.pdf \
  --pages 0 \
  --start-page 1 \
  --device cuda \
  --precision fp16 \
  --batch-size 8 \
  --cache full_book_cache.json \
  --output translated_book_fa.pdf \
  --font (fc-match -f '%{file}' Vazirmatn)
```

For a specific page range, use `--start-page 11 --pages 10` to translate source pages 11–20.

### Offline use

Once the NLLB model has been downloaded successfully, append `--offline` to any command. Do not enable it before the first successful download.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--input` | `english_book.pdf` | Source PDF path |
| `--output` | `persian_book.pdf` | Generated Persian PDF |
| `--font` | `Vazirmatn-Regular.ttf` | Path to Persian TTF font |
| `--cache` | `nllb_translation_cache.json` | JSON translation cache |
| `--pages` | `3` | Number of pages; **`0` means all remaining pages** |
| `--start-page` | `1` | First source page (1-based) |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` (auto selects CUDA if available) |
| `--precision` | `auto` | `auto`, `fp16`, or `fp32`; auto uses FP16 on CUDA, FP32 on CPU |
| `--batch-size` | `4` | Number of chunks per inference batch; actual batches may be smaller |
| `--cpu-threads` | `8` | PyTorch CPU thread count |
| `--offline` | disabled | Load model files from local Hugging Face cache only |

`--precision fp16` is not supported on CPU by this script. Use `--precision fp32` or `auto` instead.

## Resume, caching, and benchmarking

Each completed translation batch is written to the JSON cache. If processing is interrupted, **run the same command with the same PDF and `--cache` path**. Previously translated chunks should be reused. The script validates the source PDF fingerprint and model/language identity before accepting a cache.

Use a **different cache file for each different book**. The output PDF is written after all selected pages finish; an interrupted run may have a usable cache without a new output PDF.

For performance comparisons, use the **same PDF and page range**, use **fresh caches** for each benchmark, and repeat measurements. A warm cache can make a run appear much faster without performing translation. `time python ...` measures end-to-end runtime, including model startup and PDF generation; the script's displayed `elapsed` value begins after model loading.

### Example measurements (one run each; not controlled benchmarks)

On the same 10 source pages, this project's preliminary observations were:

| System / settings | End-to-end time |
|---|---:|
| Intel Core m7 laptop, CPU | approximately 360 s |
| NVIDIA RTX A2000, CUDA FP16, batch 4 | 66.10 s |
| NVIDIA RTX A2000, CUDA FP16, batch 8 | 30.91 s |

Batch 8 was approximately **11.65× faster** than the laptop CPU in these runs. These measurements are workload-specific; they are not general hardware benchmarks. Monitor VRAM with `nvidia-smi`. A larger batch is not always faster.

## Troubleshooting

**`ModuleNotFoundError: No module named 'pymupdf'`** — Activate the correct virtual environment (`source ~/pdf-translator-env/bin/activate.fish` for Fish), then run `python -m pip install pymupdf`. Check `which python`.

**`CUDA unavailable`** — Run `nvidia-smi`, verify the NVIDIA driver, and check `torch.cuda.is_available()`. Install a CUDA-enabled PyTorch wheel compatible with the driver.

**`CUDA out of memory`** — Lower `--batch-size` (for example, from 8 to 4 or 2). The script can retry with smaller batches for some out-of-memory failures.

**Only three pages were translated** — The default is three pages. Specify `--pages 0` for the entire book or an explicit count such as `--pages 389`.

**`Input PDF not found`** — Supply the actual path using `--input` or change into the directory containing the PDF.

**`Persian TTF font not found`** — Pass the full path to a `.ttf` font using `--font`.

**Hugging Face authentication warning** — Public model downloads may work without authentication but can be rate-limited. A token is optional for this public model.

**`max_new_tokens` and `max_length` warning** — This is generally nonfatal; the generation setting `max_new_tokens` takes precedence.

**Scanned page / no extractable text** — OCR is not included. Use a text-based PDF or perform OCR separately before translation.

**Cache belongs to another PDF/model** — Use a new cache file or verify the input PDF is unchanged.

## Model storage

The model is normally cached under:

```text
~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/
```

The model can be reused for multiple books; do not commit its weights to GitHub.

## Limitations and licensing

- Image-only PDFs require OCR; this script does not translate their images.
- Original formatting, diagrams, formulas, and tables are not preserved.
- Technical terminology and equations may be mistranslated; review results manually.
- Output PDF pagination may differ from source pagination.
- Review the [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M) for licensing and usage restrictions, especially before commercial use. Respect the copyright of source books.

## Suggested repository structure

```text
pdf-translator/
├── translate_nllb.py
├── README.md
└── .gitignore
```

Example `.gitignore`:

```gitignore
pdf-translator-env/
__pycache__/
*.pyc
*.pdf
*_cache.json
nllb_translation_cache.json
```

## Acknowledgements

[Meta NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) · [PyTorch](https://pytorch.org/) · [Hugging Face Transformers](https://huggingface.co/docs/transformers/) · [PyMuPDF](https://pymupdf.readthedocs.io/) · [ReportLab](https://www.reportlab.com/) · [Vazirmatn](https://github.com/rastikerdar/vazirmatn)
