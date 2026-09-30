# Larp-LLM\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/celestial-sora/Larp-LLM/blob/main/Larp_LLM_Colab_Playground.ipynb)

A Colab playground for **celestial-sora/Larp-LLM**, a LoRA adapter based on Gemma 3 12B.

## Run in Google Colab

Click **Open in Colab** above. Colab opens a personal copy of the notebook environment; select a GPU runtime and run the cells from top to bottom.

The notebook installs Unsloth and Gradio, loads `celestial-sora/Larp-LLM` from Hugging Face, and launches a temporary Gradio chat interface.

## Model

Hugging Face: https://huggingface.co/celestial-sora/Larp-LLM

Base model: `unsloth/gemma-3-12b-it-unsloth-bnb-4bit`

## Notes

- GPU runtime is required.
- Model weights are not stored in this GitHub repository.
- The Gradio share URL is temporary and exists only while the Colab runtime is active.
- Gemma 3 uses multimodal-style chat content blocks, including for text-only messages.
