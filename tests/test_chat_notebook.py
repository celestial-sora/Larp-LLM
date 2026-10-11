"""CPU regression checks: real Gradio 6, deterministic fake model (no weights)."""
import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import nbformat
import pytest
import gradio as gr

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Larp_LLM_Colab_Kaggle_Chat.ipynb'

def code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return [cell.source for cell in notebook.cells if cell.cell_type == 'code']

class Tensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))
    def to(self, device):
        return self
    def __getitem__(self, index):
        return Tensor(self.ids[index[1]])
    def numel(self):
        return len(self.ids)

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {'input_ids': Tensor([10, 11, 12]), 'attention_mask': Tensor([1, 1, 1])}
    def decode(self, tokens, **kwargs):
        return 'answer'

class Model:
    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type='cuda')))
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return Tensor([10, 11, 12, 21, 22, 2])

@pytest.fixture
def env():
    sync = []
    seed = []
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext, manual_seed=seed.append,
                            cuda=SimpleNamespace(synchronize=sync.append, OutOfMemoryError=MemoryError))
    scope = dict(gr=gr, torch=torch, model=Model(), tokenizer=Tokenizer(), MAX_SEQ_LENGTH=2048,
                 DEFAULT_SYSTEM='Be helpful.', sync=sync, seeds=seed)
    source = next(s for s in code_cells() if 'def respond(' in s)
    exec(compile(source, str(NOTEBOOK), 'exec'), scope)
    return scope

def test_cells_compile():
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    for source in code_cells():
        ast.parse(shell.input_transformer_manager.transform_cell(source))

def test_gradio6_history(env):
    history = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}, {'type': 'file', 'file': {'path': 'x'}}]},
               {'role': 'assistant', 'content': 'hi'}]
    messages = env['build_messages']('next', history, 'system')
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[1]['content'] == [{'type': 'text', 'text': 'hello'}]

def test_scope_count_timing_and_greedy(env):
    env['time'] = SimpleNamespace(perf_counter=iter([100., 102.]).__next__)
    answer, stats = env['respond']('test', [], '', 0, .9, 64, 0, 16, 1.05, 42)
    assert answer == 'answer'
    assert stats == '**3 tok** · **1.50 tok/s** · **2.00 s** · 3 input tok'
    assert len(env['sync']) == 2
    assert env['seeds'] == [42]
    assert env['model'].kwargs['pad_token_id'] == 0
    assert 'temperature' not in env['model'].kwargs
    assert 'inputs' not in env

def test_sampling_and_context_error(env):
    env['respond']('test', [], '', .8, .9, 64, .1, 16, 1.05, -1)
    assert env['model'].kwargs['min_p'] == .1
    env['MAX_SEQ_LENGTH'] = 16
    with pytest.raises(gr.Error, match='Context limit'):
        env['respond']('test', [], '', .8, .9, 64, 0, 16, 1.05, -1)
    with pytest.raises(gr.Error, match='Enter a message'):
        env['respond'](' ', [], '', .8, .9, 64, 0, 16, 1.05, -1)

def test_real_gradio_build_and_callback(env):
    source = next(s for s in code_cells() if s.startswith('CSS ='))
    exec(compile(source, str(NOTEBOOK), 'exec'), env)
    demo = env['demo']
    assert demo.config['components']
    # Exercise the real ChatInterface wrapper with structured history and extra output.
    import asyncio
    result = asyncio.run(env['chat']._submit_fn('hello', [], 'system', .8, .9, 64, 0, 16, 1.05, -1))
    assert 'tok/s' in str(result)
    demo.close()
