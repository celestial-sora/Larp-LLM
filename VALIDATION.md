# Validation — 2026-10-11

## Repository inspection

Inspected the existing README, requirements and every legacy notebook cell before changes. The original install cell contained literal escaped newlines; the old UI used removed Gradio arguments. Verified the public adapter configuration identifies Gemma3ForConditionalGeneration, its Unsloth Gemma 3 12B 4-bit base, and PEFT 0.19.1 metadata. The original notebook is retained; the README directs users to the new file.

## Executed checks

Environment: isolated CPU environment, Python 3.14, Gradio 6.30.0.

- `nbformat.validate`: passed; notebook has separate install, configuration, load, generation, UI, launch and shutdown cells, with no saved outputs.
- All code cells compile after IPython transformation (including `%pip` cells).
- `pytest`: **5 passed**. Checks real Gradio UI construction and ChatInterface callback with an additional metrics output; structured and string history; input scope; output-ID token counting; deterministic elapsed time; CUDA synchronization calls; greedy vs sampling arguments; pad ID zero; seed and context/empty-message errors.
- Actual Gradio server launch with CSS/theme passed through `launch()`: app and config endpoints both returned HTTP 200; server closed cleanly.
- `pip check` in the CPU/UI test environment: no broken requirements.
- Unsloth package availability queried successfully (latest visible version 2026.10.3). GPU dependencies deliberately remain runtime-resolved; the notebook prints exact installed versions and runs `%pip check`.
- `git diff --check`: passed.

## Limits

This host has no NVIDIA GPU. Full Unsloth/Torch/CUDA installation, model loading, actual Gemma generation and measured GPU speed were **not executed**. The fake model validates callback behavior, not model quality or hardware compatibility. Colab/Kaggle GPU execution remains required. Real iPad/mobile hardware, virtual keyboard behavior and browser screenshots were not tested; responsive CSS uses viewport units, touch-size controls and safe-area padding.

Token statistics count generated IDs, including emitted special tokens. Timing includes model.generate prefill/decoding and first-run overhead, excludes tokenization/rendering, and is not a decode-only benchmark. Responses are displayed after generation completes.

## Reproduce CPU checks

```sh
python -m pip install 'gradio>=6,<7' nbformat ipython pytest
python -m pytest -q
```

For full verification use a fresh Colab/Kaggle GPU runtime, install once, restart, and run the remaining cells in order. Confirm a second chat turn preserves context and produces token/time statistics. Use the share URL in desktop/iPad/mobile browsers.
