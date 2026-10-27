#!/usr/bin/env python3
"""English PDF -> Persian PDF using Meta NLLB-200 distilled 600M on CPU or CUDA.
Requires a text-based PDF and a Persian TTF font (e.g. Vazirmatn-Regular.ttf).
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pymupdf
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

MODEL = 'facebook/nllb-200-distilled-600M'
SRC_LANG = 'eng_Latn'
DST_LANG = 'pes_Arab'


def atomic_json_save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def chunk_text(text, tokenizer, max_tokens=220):
    """Split text to small token-budget chunks, including oversized sentences."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Decode token spans so no content is discarded for long paragraphs.
    return [tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True).strip()
            for i in range(0, len(ids), max_tokens)]


def page_paragraphs(page):
    """Extract text blocks in approximate reading order; no OCR."""
    blocks = sorted(page.get_text('blocks'), key=lambda b: (round(b[1] / 12), b[0]))
    return [b[4].strip() for b in blocks if len(b) > 6 and b[6] == 0 and b[4].strip()]


def persian_visual(s):
    return get_display(arabic_reshaper.reshape(s))


def wrap_rtl(text, font, size, width):
    """Wrap logical Persian text before reshaping/bidi for each displayed line."""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        candidate = f'{current} {word}'.strip()
        if pdfmetrics.stringWidth(persian_visual(candidate), font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = ''
            # Split unusually long words so they do not overflow the page.
            for char in word:
                candidate = current + char
                if current and pdfmetrics.stringWidth(persian_visual(candidate), font, size) > width:
                    lines.append(current)
                    current = char
                else:
                    current = candidate
    if current:
        lines.append(current)
    return lines


def make_pdf(cache, output, font_path, source_pages):
    pdfmetrics.registerFont(TTFont('PersianBookFont', str(font_path)))
    c = canvas.Canvas(str(output), pagesize=A4)
    page_w, page_h = A4
    margin = 48
    y = page_h - margin
    font_size = 12
    line_height = 21
    c.setTitle('Persian Translation')
    c.setFont('PersianBookFont', font_size)

    def next_sheet():
        nonlocal y
        c.showPage()
        c.setFont('PersianBookFont', font_size)
        y = page_h - margin

    for page_number in source_pages:
        if page_number != source_pages[0]:
            next_sheet()
        entries = cache['pages'].get(str(page_number), {})
        for translated in entries.get('translations', []):
            for para in translated.splitlines():
                if not para.strip():
                    continue
                for line in wrap_rtl(para, 'PersianBookFont', font_size, page_w - 2 * margin):
                    if y < margin + line_height:
                        next_sheet()
                    c.drawRightString(page_w - margin, y, persian_visual(line))
                    y -= line_height
                y -= 8
    c.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='english_book.pdf')
    ap.add_argument('--output', default='persian_book.pdf')
    ap.add_argument('--font', default='Vazirmatn-Regular.ttf')
    ap.add_argument('--cache', default='nllb_translation_cache.json')
    ap.add_argument('--pages', type=int, default=3, help='Number of pages (default: 3); 0 means all')
    ap.add_argument('--start-page', type=int, default=1)
    ap.add_argument('--offline', action='store_true', help='Only load cached model files')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu',
                    help='Execution device (default: cpu; select cuda explicitly for GPU)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--precision', choices=['auto', 'fp16', 'fp32'], default='auto')
    ap.add_argument('--cpu-threads', type=int, default=8)
    args = ap.parse_args()

    source = Path(args.input)
    font = Path(args.font)
    cache_path = Path(args.cache)
    if not source.is_file():
        ap.error(f'Input PDF not found: {source}')
    if not font.is_file():
        ap.error(f'Persian TTF font not found: {font}')
    if args.batch_size < 1 or args.cpu_threads < 1:
        ap.error('--batch-size and --cpu-threads must be >= 1')
    if args.start_page < 1 or args.pages < 0:
        ap.error('--start-page must be >= 1 and --pages must be >= 0')

    doc = pymupdf.open(str(source))
    end = len(doc) if args.pages == 0 else min(len(doc), args.start_page + args.pages - 1)
    if args.start_page > len(doc):
        ap.error(f'PDF has only {len(doc)} pages')
    selected = list(range(args.start_page, end + 1))

    # Hash full source bytes to avoid accidentally reusing translations for a different PDF.
    file_hash = hashlib.sha256()
    with source.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            file_hash.update(data)
    identity = {'sha256': file_hash.hexdigest(), 'model': MODEL, 'source': SRC_LANG, 'target': DST_LANG}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
        if cache.get('identity') != identity:
            ap.error(f'Cache belongs to another PDF/model. Use a different --cache path: {cache_path}')
    else:
        cache = {'identity': identity, 'pages': {}}

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA unavailable. Install a CUDA-enabled PyTorch build and check nvidia-smi.')
    if args.precision == 'fp16' and device == 'cpu':
        ap.error('FP16 on CPU is not supported by this script; use --precision fp32.')
    dtype = torch.float16 if (args.precision == 'fp16' or
                              (args.precision == 'auto' and device == 'cuda')) else torch.float32
    torch.set_num_threads(args.cpu_threads)
    print(f'Loading NLLB model on {device} ({dtype}), batch size {args.batch_size}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC_LANG, local_files_only=args.offline)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, local_files_only=args.offline, use_safetensors=True, dtype=dtype
    ).to(device)
    model.eval()
    target_id = tokenizer.convert_tokens_to_ids(DST_LANG)
    if device == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)

    def translate_batch(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=256).to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, forced_bos_token_id=target_id,
                                       max_new_tokens=300, num_beams=1)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    started = time.monotonic()

    for idx, page_number in enumerate(selected, 1):
        key = str(page_number)
        paragraphs = page_paragraphs(doc[page_number - 1])
        chunks = [chunk for para in paragraphs for chunk in chunk_text(para, tokenizer)]
        # Cache also stores the original chunks so resuming cannot silently mix text.
        entry = cache['pages'].get(key)
        if not entry or entry.get('chunks') != chunks:
            entry = {'chunks': chunks, 'translations': []}
            cache['pages'][key] = entry
            atomic_json_save(cache_path, cache)
        print(f'Page {page_number}/{len(doc)} ({idx}/{len(selected)} selected): '
              f'{len(entry["translations"])}/{len(chunks)} chunks cached', flush=True)
        if not chunks:
            print('  No extractable text (scanned page or image-only).', flush=True)
        i = len(entry['translations'])
        batch_size = args.batch_size
        while i < len(chunks):
            size = min(batch_size, len(chunks) - i)
            try:
                translated_batch = translate_batch(chunks[i:i + size])
            except torch.cuda.OutOfMemoryError:
                if device != 'cuda' or size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, size // 2)
                print(f'  CUDA OOM; reducing batch size to {batch_size}', flush=True)
                continue
            if len(translated_batch) != size:
                raise RuntimeError('Unexpected number of translations; cache not updated')
            entry['translations'].extend(translated_batch)
            atomic_json_save(cache_path, cache)
            i += size
            print(f'  chunks {i - size + 1}-{i}/{len(chunks)} saved', flush=True)
        percent = 100 * idx / len(selected)
        elapsed = (time.monotonic() - started) / 60
        print(f'Progress: {percent:.1f}% of selected pages; elapsed {elapsed:.1f} min', flush=True)

    doc.close()
    make_pdf(cache, Path(args.output), font, selected)
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
