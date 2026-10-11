# Larp-LLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/celestial-sora/Larp-LLM/blob/feat/gemma3-chat-notebook/Larp_LLM_Colab_Kaggle_Chat.ipynb)

A chat-first Colab/Kaggle notebook for the [celestial-sora/Larp-LLM adapter](https://huggingface.co/celestial-sora/Larp-LLM), based on `unsloth/gemma-3-12b-it-unsloth-bnb-4bit`.

## Run

[Open the new chat notebook in Google Colab](https://colab.research.google.com/github/celestial-sora/Larp-LLM/blob/feat/gemma3-chat-notebook/Larp_LLM_Colab_Kaggle_Chat.ipynb), or import `Larp_LLM_Colab_Kaggle_Chat.ipynb` into Kaggle. Enable a single NVIDIA GPU and Internet access, install dependencies, restart the kernel once, then run configuration through launch in order. A 16 GB GPU may be tight; 24 GB or more gives more headroom. Model weights are downloaded from Hugging Face and are not stored here.

The Gradio 6 UI has a responsive chat layout, collapsed advanced settings (system prompt, temperature, top-p, top-k, min-p, output limit, repetition penalty and seed), new-chat control and last-response statistics. It accepts Gradio 6 structured text history. Output appears when generation completes.

Generated token count comes from actual output IDs (including emitted special tokens), not re-tokenized text. Measured tok/s is output tokens divided by synchronized `model.generate` wall time, including prefill and decoding; tokenization and UI time are excluded. First-run compilation may affect the result.

`SHARE=True` creates a temporary public URL while the runtime is active. Set it to `False` for local access. Optional `HF_TOKEN` comes from environment/Colab/Kaggle secrets. The notebook never prints it.

The original `Larp_LLM_Colab_Playground.ipynb` is retained as a legacy reference; use the new notebook for Gradio 6.

## Validation

See [VALIDATION.md](VALIDATION.md) for executed checks and limits. CPU regression tests use real Gradio 6 and a deterministic fake model:

```sh
python -m pip install 'gradio>=6,<7' nbformat ipython pytest
python -m pytest -q
```

GPU runtime dependencies are listed in `requirements.txt`; the notebook installs these itself. Unsloth/Torch/CUDA compatibility must be checked in the chosen Colab/Kaggle runtime.
