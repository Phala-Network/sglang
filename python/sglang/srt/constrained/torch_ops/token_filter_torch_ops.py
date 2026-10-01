# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Torch fallback for token filter operations (non-CUDA devices and HIP).

Sets or clears specific bits in an int32 bitmask by token ID.  The token list
is typically tiny (< 10 entries) and the same few lists recur, so the prepared indices,
bit masks and reset rows are cached.
"""

import ctypes
import functools
from typing import List, Tuple

import torch


@functools.lru_cache(maxsize=256)
def _filter_rows(
    token_ids: Tuple[int, ...], is_allowed: bool, words: int, dtype, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Element indices and bit masks for token_ids, and the whole row they produce on a
    reset row. The same few token lists recur for every request and draft token."""
    aggregated: dict[int, int] = {}
    for token_id in token_ids:
        aggregated[token_id // 32] = aggregated.get(token_id // 32, 0) | (
            1 << (token_id % 32)
        )
    element_indices = torch.tensor(list(aggregated), dtype=torch.long, device=device)
    bitmasks = torch.tensor(
        [
            ctypes.c_int32(mask if is_allowed else ~mask).value
            for mask in aggregated.values()
        ],
        dtype=dtype,
        device=device,
    )
    row = torch.full((words,), 0 if is_allowed else -1, dtype=dtype, device=device)
    _apply(row, element_indices, bitmasks, is_allowed)
    return element_indices, bitmasks, row


def _apply(row, element_indices, bitmasks, is_allowed):
    if is_allowed:
        row[element_indices] = torch.bitwise_or(row[element_indices], bitmasks)
    else:
        row[element_indices] = torch.bitwise_and(row[element_indices], bitmasks)


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if not token_ids:
        if reset_vocab_mask:
            vocab_mask[batch_idx].fill_(-1 if (not is_allowed) else 0)
        return

    element_indices, bitmasks, reset_row = _filter_rows(
        tuple(token_ids),
        is_allowed,
        vocab_mask.shape[-1],
        vocab_mask.dtype,
        vocab_mask.device,
    )
    row = vocab_mask[batch_idx]
    if reset_vocab_mask:
        row.copy_(reset_row)
    else:
        _apply(row, element_indices, bitmasks, is_allowed)
