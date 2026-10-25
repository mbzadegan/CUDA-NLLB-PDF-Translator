# NLLB-PDF-Translator

# Offline English-to-Persian PDF Translator (NLLB-200)

Translate text-based English PDF books into Persian PDFs **locally on your CPU** using Meta's [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M). No NVIDIA GPU or translation API key is required.

The model is downloaded automatically **on the first online run** and reused for subsequent books. After downloading, translation can run offline.

## Features

- English → Persian translation using NLLB-200 (`eng_Latn` → `pes_Arab`)
- CPU-only execution; suitable for Linux and other supported Python platforms
- Automatic model download and Hugging Face caching
- Resume after interruption: each translated text chunk is saved immediately
- Progress reporting by page and chunk
- Persian right-to-left PDF output using a TrueType font
- Configurable input/output files, font, cache, and page range
- Source-file fingerprint in the translation cache to prevent mixing books

## Requirements

- Python 3.11 or 3.12 recommended (package compatibility may vary on newer versions)
- Sufficient RAM and free disk space for the NLLB-200 600M model and dependencies; a CPU-only machine with 16 GB RAM has been tested
- Internet connection for the initial model download
- A Persian-capable `.ttf` font, such as **Vazirmatn Regular**
- A **text-based** English PDF (scanned/image-only PDFs require OCR, which this script does not provide)

## Installation

Clone the repository or download `translate_nllb.py`, then create a virtual environment:

```bash
python3.12 -m venv pdf-translator-env
source pdf-translator-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers sentencepiece safetensors pymupdf reportlab arabic-reshaper python-bidi
```

On Fedora, if Python 3.12 is missing:

```bash
sudo dnf install python3.12
```

On Windows, activate the virtual environment with `pdf-translator-env\Scripts\activate` instead. Install a CPU-compatible PyTorch build for your platform if the standard installation is unsuitable.

### Persian font

On Fedora, a common Vazirmatn font path is:

```text
/usr/share/fonts/vazirmatn-fonts/Vazirmatn-Regular.ttf
```

You can find an installed font with:

```bash
fc-list :lang=fa family file
```

Alternatively, place a file named `Vazirmatn-Regular.ttf` alongside the script. Obtain the font from its [official project](https://github.com/rastikerdar/vazirmatn).

## Usage

Place your input PDF alongside the script as `english_book.pdf`, or specify another path with `--input`.

**First run (online):** do **not** pass `--offline` until the model has been downloaded successfully.

```bash
python translate_nllb.py --pages 3 --font /usr/share/fonts/vazirmatn-fonts/Vazirmatn-Regular.ttf
```

This translates the first three pages and writes `persian_book.pdf`.

**Translate the whole book after the model is cached:**

```bash
python translate_nllb.py --pages 0 --offline --font /usr/share/fonts/vazirmatn-fonts/Vazirmatn-Regular.ttf
```

**Translate a specific range:**

```bash
python translate_nllb.py --start-page 11 --pages 10 --offline --font /usr/share/fonts/vazirmatn-fonts/Vazirmatn-Regular.ttf
```

**Use custom paths:**

```bash
python translate_nllb.py \
  --input my_book.pdf \
  --output my_book_fa.pdf \
  --cache my_book_translation_cache.json \
  --font /path/to/PersianFont.ttf \
  --pages 0 \
  --offline
```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--input` | `english_book.pdf` | Source PDF path |
| `--output` | `persian_book.pdf` | Translated PDF path |
| `--font` | `Vazirmatn-Regular.ttf` | Persian TTF font path |
| `--cache` | `nllb_translation_cache.json` | Saved translations |
| `--pages` | `3` | Number of pages to process; `0` means from the start page to the end |
| `--start-page` | `1` | First source page (1-based) |
| `--offline` | Off | Require model files to exist in the local Hugging Face cache |

## Resuming an interrupted translation

The program saves each completed chunk to the JSON cache. Run the **same command again**, using the same input PDF and cache file. Already translated chunks are reused.

When changing books, use a separate `--cache` filename. The script checks the PDF's SHA-256 fingerprint and rejects a cache belonging to another book or model.

The output PDF is generated **after the selected pages finish processing**. If interrupted, the cached translations remain, but the final PDF may not yet exist or may be from an earlier completed run.

## Where is the NLLB model stored?

By default, Hugging Face stores downloaded files under:

```text
~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/
```

The model is **not specific to one book**. It can be reused across books and across compatible machines. You do **not** need to commit its multi-gigabyte weights to GitHub. If moving the model cache between computers, preserve the full cache directory structure, including symlink targets.

## CPU performance

The script currently sets:

```python
torch.set_num_threads(2)
```

You can edit this line to benchmark other values, such as `4`, `8`, `14`, or `28`, depending on your CPU. More threads do not necessarily translate faster: test the same pages and compare timings. A 14-core/28-thread Xeon may outperform a low-power laptop CPU, but actual speed depends on memory, clock rates, workload, and thermal limits.

## Limitations

- **No OCR:** scanned/image-only pages will not be translated.
- **No layout preservation:** original images, diagrams, tables, page design, and mathematical formatting are not reconstructed.
- **Technical content:** equations, code, and specialist terminology may be mistranslated or damaged; review important passages manually.
- **Page count:** translated output pagination may differ from the original.
- **Model quality:** machine translations can be inaccurate; this is not a substitute for professional review.
- **Licensing:** review the [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M) and its license/usage restrictions before redistribution or commercial use. Also respect the copyright and permissions applicable to the books you translate.

## Troubleshooting

**`Persian TTF font not found`** — Supply an existing font path with `--font` or place `Vazirmatn-Regular.ttf` beside the script.

**Offline model load fails** — Run once without `--offline` while connected to the internet, and allow the download to finish.

**`Cache belongs to another PDF/model`** — Use a different `--cache` path for the new PDF.

**No text extracted** — The PDF may be scanned and require OCR before translation.

**Slow translation** — This is a large neural translation model running on CPU. Benchmark a few pages first, and test `torch.set_num_threads(...)` values appropriate for your hardware.

## Suggested GitHub files

```text
repository/
├── translate_nllb.py
├── README.md
└── requirements.txt       # optional; list the dependencies above
```

Keep book PDFs, generated PDFs, translation caches, virtual environments, and downloaded model weights out of version control. A useful `.gitignore` includes:

```gitignore
pdf-translator-env/
__pycache__/
*.pyc
*_translation_cache.json
nllb_translation_cache.json
persian_book.pdf
```

## Acknowledgements

Translation model: [Meta NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M). PDF extraction: [PyMuPDF](https://pymupdf.readthedocs.io/). PDF creation: [ReportLab](https://www.reportlab.com/). Persian font: [Vazirmatn](https://github.com/rastikerdar/vazirmatn).
