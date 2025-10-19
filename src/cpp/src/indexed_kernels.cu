#include "indexed_kernels.h"

#include <ATen/AccumulateType.h> // For at::acc_type
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

#define TORCH_CUDA_CHECK_CUBLAS(status)                                           \
    TORCH_CHECK(                                                                  \
        (status) == CUBLAS_STATUS_SUCCESS,                                        \
        "cuBLASLt error: ",                                                      \
        static_cast<int>(status))

// -----------------------------------------------------------------------------
// Embedding Kernel
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void indexed_batched_embedding_kernel(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ policy_indices,
    const scalar_t* __restrict__ weight_cache,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t time_steps,
    int64_t hidden_dim,
    int64_t vocab_size) {
    const int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_tokens = batch_size * time_steps;
    if (linear_index >= total_tokens) {
        return;
    }

    const int64_t b = linear_index / time_steps;
    const int64_t t = linear_index % time_steps;

    const int64_t policy_idx = policy_indices[b];
    const int64_t token_idx = indices[linear_index];

    const scalar_t* table_ptr = weight_cache + policy_idx * vocab_size * hidden_dim;
    const scalar_t* src_ptr = table_ptr + token_idx * hidden_dim;
    scalar_t* dst_ptr = output + linear_index * hidden_dim;

#pragma unroll 4
    for (int64_t h = 0; h < hidden_dim; ++h) {
        dst_ptr[h] = src_ptr[h];
    }
}

// -----------------------------------------------------------------------------
// LayerNorm Kernel
// -----------------------------------------------------------------------------

template <typename acc_t>
__device__ acc_t warp_reduce_sum(acc_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t, typename acc_t>
__global__ void indexed_batched_layer_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma_cache,
    const scalar_t* __restrict__ beta_cache,
    const int64_t* __restrict__ policy_indices,
    int64_t batch_size,
    int64_t time_steps,
    int64_t hidden_dim,
    double eps) {
    const int64_t token_index = blockIdx.x;
    if (token_index >= batch_size * time_steps) {
        return;
    }

    const int64_t b = token_index / time_steps;
    const int64_t policy_idx = policy_indices[b];

    const scalar_t* input_ptr = input + token_index * hidden_dim;
    scalar_t* output_ptr = output + token_index * hidden_dim;

    const scalar_t* gamma_ptr = gamma_cache + policy_idx * hidden_dim;
    const scalar_t* beta_ptr = beta_cache + policy_idx * hidden_dim;

    __shared__ acc_t shared_partial[32];
    __shared__ acc_t shared_sum;
    __shared__ acc_t shared_var;

    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    acc_t thread_sum = static_cast<acc_t>(0);
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        thread_sum += static_cast<acc_t>(input_ptr[h]);
    }
    acc_t warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_partial[warp_id] = warp_sum;
    }
    __syncthreads();

    acc_t block_sum = (threadIdx.x < warp_count) ? shared_partial[threadIdx.x] : static_cast<acc_t>(0);
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
        shared_sum = block_sum / static_cast<acc_t>(hidden_dim);
    }
    __syncthreads();
    const acc_t mean = shared_sum;

    acc_t thread_var = static_cast<acc_t>(0);
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t diff = static_cast<acc_t>(input_ptr[h]) - mean;
        thread_var += diff * diff;
    }
    warp_sum = warp_reduce_sum(thread_var);
    if (lane == 0) {
        shared_partial[warp_id] = warp_sum;
    }
    __syncthreads();

    block_sum = (threadIdx.x < warp_count) ? shared_partial[threadIdx.x] : static_cast<acc_t>(0);
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
        shared_var = block_sum / static_cast<acc_t>(hidden_dim);
    }
    __syncthreads();

    const acc_t inv_std = rsqrt(shared_var + static_cast<acc_t>(eps));

    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        const acc_t norm = (static_cast<acc_t>(input_ptr[h]) - mean) * inv_std;
        const acc_t scaled = norm * static_cast<acc_t>(gamma_ptr[h]) + static_cast<acc_t>(beta_ptr[h]);
        output_ptr[h] = static_cast<scalar_t>(scaled);
    }
}

// -----------------------------------------------------------------------------
// Helper utilities for cuBLASLt
// -----------------------------------------------------------------------------

cublasComputeType_t compute_type_for(const at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return CUBLAS_COMPUTE_32F;
        case at::kHalf:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        default:
            TORCH_CHECK(false, "Unsupported dtype for indexed_batched_linear");
    }
}

cudaDataType_t cuda_dtype_for(const at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:
            return CUDA_R_32F;
        case at::kHalf:
            return CUDA_R_16F;
        default:
            TORCH_CHECK(false, "Unsupported dtype for indexed_batched_linear");
    }
}

struct MatmulDescriptors {
    cublasLtMatmulDesc_t op_desc{nullptr};
    cublasLtMatrixLayout_t layout_a{nullptr};
    cublasLtMatrixLayout_t layout_b{nullptr};
    cublasLtMatrixLayout_t layout_c{nullptr};

    ~MatmulDescriptors() {
        if (layout_c) cublasLtMatrixLayoutDestroy(layout_c);
        if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
        if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    }
};

MatmulDescriptors create_descriptors(
    const at::ScalarType dtype,
    int64_t M,
    int64_t N,
    int64_t K) {
    MatmulDescriptors desc{};

    const auto compute_type = compute_type_for(dtype);
    const auto data_type = cuda_dtype_for(dtype);

    TORCH_CUDA_CHECK_CUBLAS(
        cublasLtMatmulDescCreate(&desc.op_desc, compute_type, data_type));

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &desc.layout_a, data_type, M, K, K));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &desc.layout_b, data_type, N, K, K));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &desc.layout_c, data_type, M, N, N));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    return desc;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

torch::Tensor indexed_batched_embedding(
    const torch::Tensor& weight_cache,
    const torch::Tensor& indices,
    const torch::Tensor& policy_indices) {
    TORCH_CHECK(weight_cache.dim() == 3, "Embedding cache must be [W, vocab, hidden]");
    TORCH_CHECK(indices.dim() == 2, "Indices must be [B, T]");
    TORCH_CHECK(policy_indices.dim() == 1, "policy_indices must be [B]");

    auto vocab_size = weight_cache.size(1);
    auto hidden_dim = weight_cache.size(2);
    auto batch_size = indices.size(0);
    auto time_steps = indices.size(1);

    TORCH_CHECK(weight_cache.is_cuda(), "indexed_batched_embedding expects CUDA tensors for weight_cache");

    auto weight_contig = weight_cache.contiguous();
    auto indices_contig = indices.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != weight_cache.device()) {
        policy_contig = policy_contig.to(weight_cache.device());
    }

    auto output = torch::empty({batch_size, time_steps, hidden_dim}, weight_cache.options());

    const int threads = 256;
    const int64_t total_tokens = batch_size * time_steps;
    const dim3 blocks((total_tokens + threads - 1) / threads);

    c10::cuda::CUDAGuard guard(weight_cache.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weight_cache.scalar_type(), "indexed_batched_embedding_cuda", [&] {
            indexed_batched_embedding_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                indices_contig.data_ptr<int64_t>(),
                policy_contig.data_ptr<int64_t>(),
                weight_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                time_steps,
                hidden_dim,
                vocab_size);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor indexed_batched_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma_cache,
    const torch::Tensor& beta_cache,
    const torch::Tensor& policy_indices,
    double eps) {
    TORCH_CHECK(input.dim() == 3, "input must be [B, T, H]");
    TORCH_CHECK(gamma_cache.dim() == 2, "gamma cache must be [W, H]");
    TORCH_CHECK(beta_cache.dim() == 2, "beta cache must be [W, H]");

    auto batch_size = input.size(0);
    auto time_steps = input.size(1);
    auto hidden_dim = input.size(2);

    TORCH_CHECK(input.is_cuda(), "indexed_batched_layer_norm expects CUDA tensors for input");

    auto input_contig = input.contiguous();
    auto gamma_contig = gamma_cache.contiguous();
    auto beta_contig = beta_cache.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }

    auto output = torch::empty_like(input_contig, input_contig.options());

    const int threads = hidden_dim >= 256 ? 256 : 128;
    const int64_t total_tokens = batch_size * time_steps;

    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "indexed_batched_layer_norm_cuda", [&] {
            using acc_t = at::acc_type<scalar_t, true>;
            indexed_batched_layer_norm_kernel<scalar_t, acc_t><<<total_tokens, threads, 0, stream>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma_contig.data_ptr<scalar_t>(),
                beta_contig.data_ptr<scalar_t>(),
                policy_contig.data_ptr<int64_t>(),
                batch_size,
                time_steps,
                hidden_dim,
                eps);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor indexed_batched_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight_cache,
    const torch::Tensor& bias_cache,
    const torch::Tensor& policy_indices,
    IndexedLinearEpilogue epilogue) {
    TORCH_CHECK(weight_cache.dim() == 3, "weight cache must be [W, out, in]");
    TORCH_CHECK(bias_cache.dim() == 2, "bias cache must be [W, out]");
    TORCH_CHECK(input.dim() == 3, "input must be [B, T, in]");

    const int64_t batch_size = input.size(0);
    const int64_t time_steps = input.size(1);
    const int64_t in_dim = input.size(2);
    const int64_t out_dim = weight_cache.size(1);

    TORCH_CHECK(weight_cache.size(2) == in_dim, "Input dim mismatch");

    if (!input.is_cuda()) {
        auto policy_cpu = policy_indices.cpu();
        auto weight = weight_cache.index_select(0, policy_cpu);
        auto bias = bias_cache.index_select(0, policy_cpu);
        auto x = input.to(weight.scalar_type());
        auto result = torch::matmul(x, weight.transpose(-1, -2)) + bias.unsqueeze(1);
        if (epilogue == IndexedLinearEpilogue::BiasGELU) {
            result = torch::gelu(result);
        }
        return result;
    }

    TORCH_CHECK(weight_cache.device().is_cuda(), "weight cache must be CUDA when input is CUDA");
    TORCH_CHECK(bias_cache.device().is_cuda(), "bias cache must be CUDA when input is CUDA");

    c10::cuda::CUDAGuard guard(input.device());

    auto input_cast = input.to(weight_cache.scalar_type()).contiguous();
    auto weight_contig = weight_cache.contiguous();
    auto bias_contig = bias_cache.contiguous();
    auto policy_contig = policy_indices.to(torch::kLong).contiguous();
    if (policy_contig.device() != input.device()) {
        policy_contig = policy_contig.to(input.device());
    }
    auto policy_host = policy_contig.to(torch::kCPU);

    auto output = torch::empty({batch_size, time_steps, out_dim}, input_cast.options());

    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype = input_cast.scalar_type();
    auto desc = create_descriptors(dtype, time_steps, out_dim, in_dim);

    size_t workspace_size = 1 << 22; // 4MB
    auto workspace = torch::empty({static_cast<long>(workspace_size)}, input_cast.options().dtype(torch::kByte));

    cublasLtMatmulPreference_t preference;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned_results = 0;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        desc.op_desc,
        desc.layout_a,
        desc.layout_b,
        desc.layout_c,
        desc.layout_c,
        preference,
        1,
        &heuristic,
        &returned_results));
    TORCH_CHECK(returned_results > 0, "No cuBLASLt heuristic found");

    float alpha_float = 1.0f;
    float beta_float = 0.0f;
    auto alpha_dev = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat).device(input_cast.device()));
    auto beta_dev = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat).device(input_cast.device()));
    alpha_dev.fill_(alpha_float);
    beta_dev.fill_(beta_float);
    const void* alpha_ptr = alpha_dev.data_ptr();
    const void* beta_ptr = beta_dev.data_ptr();

    cublasLtEpilogue_t epilogue_attr = (epilogue == IndexedLinearEpilogue::BiasGELU)
        ? CUBLASLT_EPILOGUE_GELU_BIAS
        : CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue_attr,
        sizeof(epilogue_attr)));

    const char* input_bytes = static_cast<const char*>(input_cast.data_ptr());
    const char* weight_bytes = static_cast<const char*>(weight_contig.data_ptr());
    char* output_bytes = static_cast<char*>(output.data_ptr());
    char* bias_bytes = static_cast<char*>(bias_contig.data_ptr());

    std::vector<void*> a_pointers(batch_size);
    std::vector<void*> b_pointers(batch_size);
    std::vector<void*> c_pointers(batch_size);
    std::vector<void*> bias_pointers(batch_size);

    const auto* policy_host_ptr = policy_host.data_ptr<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t policy = policy_host_ptr[b];
        TORCH_CHECK(policy >= 0 && policy < weight_cache.size(0), "policy index out of range");

        const auto input_offset = b * time_steps * in_dim * input_cast.element_size();
        const auto weight_offset = policy * out_dim * in_dim * weight_contig.element_size();
        const auto output_offset = b * time_steps * out_dim * output.element_size();
        const auto bias_offset = policy * out_dim * bias_contig.element_size();

        a_pointers[b] = const_cast<char*>(input_bytes) + input_offset;
        b_pointers[b] = const_cast<char*>(weight_bytes) + weight_offset;
        c_pointers[b] = output_bytes + output_offset;
        bias_pointers[b] = bias_bytes + bias_offset;
    }

    const size_t pointer_bytes = batch_size * sizeof(void*);
    auto deleter = [](void* ptr) {
        if (ptr) {
            c10::cuda::CUDACachingAllocator::raw_delete(ptr);
        }
    };

    std::unique_ptr<void, decltype(deleter)> d_a_raw(
        c10::cuda::CUDACachingAllocator::raw_alloc(pointer_bytes), deleter);
    std::unique_ptr<void, decltype(deleter)> d_b_raw(
        c10::cuda::CUDACachingAllocator::raw_alloc(pointer_bytes), deleter);
    std::unique_ptr<void, decltype(deleter)> d_c_raw(
        c10::cuda::CUDACachingAllocator::raw_alloc(pointer_bytes), deleter);
    std::unique_ptr<void, decltype(deleter)> d_bias_raw(
        c10::cuda::CUDACachingAllocator::raw_alloc(pointer_bytes), deleter);

    void** d_a = static_cast<void**>(d_a_raw.get());
    void** d_b = static_cast<void**>(d_b_raw.get());
    void** d_c = static_cast<void**>(d_c_raw.get());
    void** d_bias = static_cast<void**>(d_bias_raw.get());

    TORCH_CUDA_CHECK(cudaMemcpyAsync(d_a, a_pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream));
    TORCH_CUDA_CHECK(cudaMemcpyAsync(d_b, b_pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream));
    TORCH_CUDA_CHECK(cudaMemcpyAsync(d_c, c_pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream));
    TORCH_CUDA_CHECK(cudaMemcpyAsync(d_bias, bias_pointers.data(), pointer_bytes, cudaMemcpyHostToDevice, stream));

    const int32_t batch_count = static_cast<int32_t>(batch_size);
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc,
        CUBLASLT_MATMUL_DESC_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &d_bias,
        sizeof(d_bias)));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc.op_desc,
        CUBLASLT_MATMUL_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_a,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_b,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_c,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_a,
        CUBLASLT_MATRIX_LAYOUT_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_b,
        CUBLASLT_MATRIX_LAYOUT_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));
    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        desc.layout_c,
        CUBLASLT_MATRIX_LAYOUT_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmul(
        handle,
        desc.op_desc,
        alpha_ptr,
        d_a,
        desc.layout_a,
        d_b,
        desc.layout_b,
        beta_ptr,
        d_c,
        desc.layout_c,
        d_c,
        desc.layout_c,
        &heuristic.algo,
        workspace_size ? workspace.data_ptr() : nullptr,
        workspace_size,
        stream));

    TORCH_CUDA_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    return output;
}

