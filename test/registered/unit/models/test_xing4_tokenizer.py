"""Xing tokenizer loading with data-only model mounts and no remote Python.

Set XING4_TOKENIZER_ASSETS to a directory with bf16/ and fp8/ subdirectories
containing the fixed official config/tokenizer.model/JSON/Jinja assets to run
the real vocabulary probes on the authorized CPU builder as well.
"""

import os
import pickle
from pathlib import Path

import pytest
import sentencepiece as spm

from sglang.srt.tokenizer.xing4_0 import Xing4_0Tokenizer
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

FIXTURES = Path(__file__).with_name("xing4_configs")
TEXTS = ["Hello, world!", "中国电信星辰模型", "a  b\n\tmath: 2 + 2 = 4", "🙂 café"]


@pytest.fixture
def data_only_model(tmp_path):
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(TEXTS * 10),
        model_prefix=str(tmp_path / "tokenizer"),
        model_type="bpe",
        vocab_size=128,
        hard_vocab_limit=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        bos_piece="<_start>",
        eos_piece="<_end>",
        pad_piece="<_pad>",
        user_defined_symbols=[
            "<_user>", "<_bot>", "<_system>", "<unused7>", "<unused8>",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>",
        ],
        minloglevel=2,
    )
    for source, target in [
        ("bf16.json", "config.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("chat_template.jinja", "chat_template.jinja"),
    ]:
        (tmp_path / target).write_bytes((FIXTURES / source).read_bytes())
    assert not list(tmp_path.glob("*.py"))
    return tmp_path


def assert_tokenizer_contract(path):
    tokenizer = get_tokenizer(str(path), trust_remote_code=False, local_files_only=True)
    assert isinstance(tokenizer, Xing4_0Tokenizer)
    assert tokenizer.__class__.__module__ == "sglang.srt.tokenizer.xing4_0"
    assert tokenizer.add_bos_token is False
    assert tokenizer.add_eos_token is False
    sentencepiece = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
    for text in TEXTS:
        assert tokenizer.encode(text, add_special_tokens=False) == sentencepiece.encode(text)
        assert tokenizer.encode(text) == sentencepiece.encode(text)
    for marker, token_id in [
        ("<_start>", 1), ("<_end>", 2), ("<_user>", 4), ("<_bot>", 5),
        ("<think>", 9), ("</think>", 10), ("<tool_call>", 11),
        ("</tool_call>", 12), ("<tool_response>", 13), ("</tool_response>", 14),
    ]:
        assert tokenizer.encode(marker, add_special_tokens=False) == [token_id]
        assert tokenizer.decode([token_id], skip_special_tokens=False) == marker
    assert tokenizer.decode(
        [9, 10, 11, 12], skip_special_tokens=False, spaces_between_special_tokens=False
    ) == "<think></think><tool_call></tool_call>"
    return tokenizer


def test_registered_loader_with_official_auto_map_and_no_python(data_only_model):
    assert_tokenizer_contract(data_only_model)


def test_chat_template_and_reconstruction(data_only_model, tmp_path):
    tokenizer = assert_tokenizer_contract(data_only_model)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert "<_user>Hello" in rendered
    assert rendered.endswith("<_bot><think>\n")
    expected = tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) == expected
    restored = pickle.loads(pickle.dumps(tokenizer))
    assert restored.encode(rendered, add_special_tokens=False) == expected
    target = tmp_path / "saved"
    tokenizer.save_pretrained(target)
    assert not list(target.glob("*.py"))
    reloaded = get_tokenizer(str(target), trust_remote_code=False, local_files_only=True)
    assert reloaded.encode(rendered, add_special_tokens=False) == expected


@pytest.mark.skipif(not os.environ.get("XING4_TOKENIZER_ASSETS"), reason="official data not configured")
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_fixed_official_data_only_checkpoint(kind):
    path = Path(os.environ["XING4_TOKENIZER_ASSETS"]) / kind
    assert not list(path.glob("*.py"))
    assert_tokenizer_contract(path)
