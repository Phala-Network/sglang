#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
<<<<<<< HEAD
#include <cassert>
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
#include <cstdint>
#include <limits>

namespace sglang {

/// Finalises the sparse indexer's block table: the top-k block ids a row
/// selected (any order, -1 padded) become, in place, the same ids ascending with
/// INT32_MAX past the row's count, plus each block as a pool slot / 8
/// (`page_table[b, id / bpp] * bpp + id % bpp`, `bpp` blocks per index page). A
/// row with at most `topk` blocks gets the identity table without reading its
/// input.
///
<<<<<<< HEAD
/// Counting sort over a per-row bitmap (one bit per block, 16/32 KiB for a
/// 1M/2M-token addressable extent): single-bit words are emitted by their owner,
/// denser words go to a block-wide queue the warps drain one lane per bit.
/// Capacity follows the page table, including speculative scratch positions.
template <uint32_t kWordsPerThread_>
=======
/// Counting sort over a per-row bitmap (one bit per block, 16 KiB for a 1M-token
/// row): single-bit words are emitted by their owner, denser words go to a
/// block-wide queue the warps drain one lane per bit.
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
struct CandidateBlockTableConfig {
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kBlockTokens = 8;
<<<<<<< HEAD
  static constexpr uint32_t kWordsPerThread = kWordsPerThread_;
  static constexpr uint32_t kMaxBlocks = kWordsPerThread * kBlockSize * 32;
  static constexpr uint32_t kMaxTopK = 2048;
  static_assert((kWordsPerThread == 4 || kWordsPerThread == 8) && kNumWarps == device::kWarpThreads);
  // The queue packs the bitmap word index and the output rank into 16 bits each.
  static_assert(kMaxBlocks / 32 <= (1u << 16) && kMaxTopK <= (1u << 16));
  static constexpr int32_t kPad = std::numeric_limits<int32_t>::max();
  // Keep 16-byte transactions on every architecture; the larger variant uses
  // two adjacent vectors per thread, retaining ascending word ownership.
  static constexpr uint32_t kVecWords = 4;
  static constexpr uint32_t kVecsPerThread = kWordsPerThread / kVecWords;
  using word_vec_t = device::AlignedVector<uint32_t, kVecWords>;
=======
  static constexpr uint32_t kMaxSeqLen = 128 * 1024;  // blocks: 1M tokens / kBlockTokens
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kWordsPerThread = kMaxSeqLen / 32 / kBlockSize;
  static_assert(kWordsPerThread == 4 && kNumWarps == device::kWarpThreads);
  static constexpr int32_t kPad = std::numeric_limits<int32_t>::max();
  using word_vec_t = device::AlignedVector<uint32_t, kWordsPerThread>;
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  struct WriteItem {
    uint32_t start;  // rank of the word's first bit | word index << 16
    uint32_t bits;
  };
  struct Smem {
    uint32_t queue_size;
    uint32_t warp_sum[kNumWarps];
    union {
<<<<<<< HEAD
      alignas(16) uint32_t bitmap[kMaxBlocks / 32];
=======
      alignas(16) uint32_t bitmap[kMaxSeqLen / 32];
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
      WriteItem write_queue[kMaxTopK];  // a queued word holds >= 2 of the topk bits
    };
  };
};

struct CandidateBlockTableParams {
  const uint32_t* __restrict__ seq_len;    // [rows] tokens
  const int32_t* __restrict__ page_table;  // [rows, pages] index-pool pages
  int32_t* __restrict__ indices;           // [rows, topk] blocks, -1 padded in, ascending + kPad out
  int32_t* __restrict__ out_pages;         // [rows, topk] the same blocks as pool slots / 8
  int64_t page_table_stride;
<<<<<<< HEAD
  uint32_t page_table_pages;  // logical width, not the possibly padded row stride
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  int64_t indices_stride;
  int64_t out_pages_stride;
  uint32_t topk;
  uint32_t page_bits;  // log2(page_size / kBlockTokens)
};

/// One CTA per row.
<<<<<<< HEAD
template <bool kUsePDL, uint32_t kWordsPerThread>
__global__ __launch_bounds__(CandidateBlockTableConfig<kWordsPerThread>::kBlockSize,
                             CandidateBlockTableConfig<kWordsPerThread>::kOccupancy)  //
    void sort_bitmap_transform(const __grid_constant__ CandidateBlockTableParams params) {
  using namespace device;
  using C = CandidateBlockTableConfig<kWordsPerThread>;
  __shared__ typename C::Smem smem;
=======
template <bool kUsePDL>
__global__ __launch_bounds__(CandidateBlockTableConfig::kBlockSize, CandidateBlockTableConfig::kOccupancy)  //
    void sort_128k_transform(const __grid_constant__ CandidateBlockTableParams params) {
  using namespace device;
  using C = CandidateBlockTableConfig;
  __shared__ C::Smem smem;
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto lanemask_lt = (1u << lane_id) - 1u;

  PDLWaitPrimary<kUsePDL>();  // indices is the block top-k's output
  const auto seq_len = params.seq_len[bx];
<<<<<<< HEAD
  assert(seq_len <= C::kMaxBlocks * C::kBlockTokens && "candidate row exceeds bitmap capacity");
  const auto nblocks = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  assert(nblocks <= (params.page_table_pages << params.page_bits) && "candidate row exceeds page table");
=======
  const auto nblocks = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
  auto* __restrict__ indices = params.indices + bx * params.indices_stride;
  auto* __restrict__ pages = params.out_pages + bx * params.out_pages_stride;
  const auto bpp_mask = (1u << params.page_bits) - 1u;
  const auto emit = [&](uint32_t rank, uint32_t id) {
<<<<<<< HEAD
    assert(rank < params.topk && id < nblocks && "invalid sorted candidate");
    assert((id >> params.page_bits) < params.page_table_pages && "candidate page exceeds page table");
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
    indices[rank] = static_cast<int32_t>(id);
    pages[rank] = (table[id >> params.page_bits] << params.page_bits) | static_cast<int32_t>(id & bpp_mask);
  };
  const auto pad = [&](uint32_t rank) {
    indices[rank] = C::kPad;
    pages[rank] = C::kPad;
  };

  if (nblocks <= params.topk) {  // every block is selected: the identity table
    for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
      if (t < nblocks) {
        emit(t, t);
      } else {
        pad(t);
      }
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  // 1. the selected blocks as a bitmap
<<<<<<< HEAD
  typename C::word_vec_t words[C::kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < C::kVecsPerThread; ++v) {
    words[v].fill(0u);
    words[v].store(smem.bitmap, tx * C::kVecsPerThread + v);
  }
=======
  C::word_vec_t words;
  words.fill(0u);
  words.store(smem.bitmap, tx);
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  if (tx == 0) smem.queue_size = 0;
  __syncthreads();
  for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
    const auto id = indices[t];
<<<<<<< HEAD
    assert(id >= 0 && static_cast<uint32_t>(id) < nblocks && "invalid selected candidate");
    assert(static_cast<uint32_t>(id) < C::kMaxBlocks && "candidate exceeds bitmap capacity");
    atomicOr(&smem.bitmap[id >> 5], 1u << (id & 31));
=======
    if (id >= 0) atomicOr(&smem.bitmap[id >> 5], 1u << (id & 31));
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  }
  __syncthreads();

  // 2. rank of every word's first bit: block-wide exclusive scan of the popcounts
<<<<<<< HEAD
#pragma unroll
  for (uint32_t v = 0; v < C::kVecsPerThread; ++v) {
    words[v].load(smem.bitmap, tx * C::kVecsPerThread + v);
  }
=======
  words.load(smem.bitmap, tx);
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  uint32_t count[C::kWordsPerThread];
  uint32_t local = 0;
#pragma unroll
  for (uint32_t j = 0; j < C::kWordsPerThread; ++j) {
<<<<<<< HEAD
    count[j] = __popc(words[j / C::kVecWords][j % C::kVecWords]);
=======
    count[j] = __popc(words[j]);
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
    local += count[j];
  }
  const auto warp_inc = warp::inclusive_sum(lane_id, local);
  if (lane_id == kWarpThreads - 1) smem.warp_sum[warp_id] = warp_inc;
  __syncthreads();  // also: every thread holds its words, the bitmap may become the queue
  const auto peer_sum = smem.warp_sum[lane_id];
  const auto warp_prefix = warp::reduce_sum(lane_id < warp_id ? peer_sum : 0u);
  const auto total = warp::reduce_sum(peer_sum);
<<<<<<< HEAD
  assert(total == params.topk && "selected candidates must be distinct and complete");
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  uint32_t base = warp_prefix + warp_inc - local;
  PDLTriggerSecondary<kUsePDL>();

  // 3. single bits by their owner, denser words queued for the warps
#pragma unroll
  for (uint32_t j = 0; j < C::kWordsPerThread; ++j) {
    const auto word_idx = tx * C::kWordsPerThread + j;
<<<<<<< HEAD
    const auto bits = words[j / C::kVecWords][j % C::kVecWords];
    if (count[j] == 1) {
      emit(base, word_idx * 32 + __ffs(bits) - 1);
    } else if (count[j] >= 2) {
      const auto slot = atomicAdd(&smem.queue_size, 1u);
      smem.write_queue[slot] = {base | (word_idx << 16), bits};
=======
    if (count[j] == 1) {
      emit(base, word_idx * 32 + __ffs(words[j]) - 1);
    } else if (count[j] >= 2) {
      const auto slot = atomicAdd(&smem.queue_size, 1u);
      smem.write_queue[slot] = {base | (word_idx << 16), words[j]};
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
    }
    base += count[j];
  }
  for (uint32_t t = total + tx; t < params.topk; t += C::kBlockSize) {
    pad(t);
  }
  __syncthreads();

  // 4. drain the queue: one word per warp step, one lane per bit
  const auto queue_size = smem.queue_size;
  for (uint32_t q = warp_id; q < queue_size; q += C::kNumWarps) {
    const auto item = smem.write_queue[q];
    if ((item.bits >> lane_id) & 1u) {
      emit((item.start & 0xFFFFu) + __popc(item.bits & lanemask_lt), (item.start >> 16) * 32 + lane_id);
    }
  }
}

/// Host entry: `indices` is rewritten in place; `page_size` is the index pool's,
/// a power of two >= 8, and the row's page table must cover its length.
template <bool kPDL>
struct CandidateBlockTableKernel {
  static void transform(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out_pages,
      const uint32_t page_size) {
<<<<<<< HEAD
    using namespace host;
    using C = CandidateBlockTableConfig<4>;
    using LargeC = CandidateBlockTableConfig<8>;
=======
    launch<sort_128k_transform<kPDL>>(indices, seq_lens, page_table, out_pages, page_size);
  }

 private:
  template <auto kKernel>
  static void launch(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out_pages,
      const uint32_t page_size) {
    using namespace host;
    using C = CandidateBlockTableConfig;
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
    auto B = SymbolicSize{"batch_size"};
    auto K = SymbolicSize{"topk_blocks"};
    auto Si = SymbolicSize{"indices_stride"};
    auto Sp = SymbolicSize{"out_pages_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, K}).with_strides({Si, 1}).with_dtype<int32_t>().with_device(device_).verify(indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({B, K}).with_strides({Sp, 1}).with_dtype<int32_t>().with_device(device_).verify(out_pages);
    RuntimeCheck(
        std::has_single_bit(page_size) && page_size >= C::kBlockTokens,
        "page_size must be a power of two of at least 8");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= C::kMaxTopK, "topk_blocks must be in (0, kMaxTopK]");
<<<<<<< HEAD
    const auto blocks_per_page = page_size / C::kBlockTokens;
    const auto num_pages = page_table.size(1);
    RuntimeCheck(
        num_pages > 0 && num_pages <= LargeC::kMaxBlocks / blocks_per_page,
        "candidate page table must cover at most 2097152 tokens, including speculative scratch");
    const auto addressable_blocks = static_cast<uint32_t>(num_pages) * blocks_per_page;
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
    const auto params = CandidateBlockTableParams{
        .seq_len = static_cast<const uint32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .out_pages = static_cast<int32_t*>(out_pages.data_ptr()),
        .page_table_stride = page_table.stride(0),
<<<<<<< HEAD
        .page_table_pages = static_cast<uint32_t>(num_pages),
=======
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
        .indices_stride = Si.unwrap(),
        .out_pages_stride = Sp.unwrap(),
        .topk = topk,
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size / C::kBlockTokens)),
    };
<<<<<<< HEAD
    auto launch = LaunchKernel(static_cast<uint32_t>(B.unwrap()), C::kBlockSize, device_.unwrap());
    launch.config({.use_pdl = kPDL});
    if (addressable_blocks <= C::kMaxBlocks) {
      launch.launch(sort_bitmap_transform<kPDL, 4>, params);
    } else {
      launch.launch(sort_bitmap_transform<kPDL, 8>, params);
    }
=======
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(kKernel, params);
>>>>>>> ae829a492f (dsv4.1: Top-k kernels and candidate selection (#39648))
  }
};

}  // namespace sglang
