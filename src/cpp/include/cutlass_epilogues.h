#pragma once

#include <cstdint>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace lb {
namespace cutlass_ext {

/**
 * @brief Epilogue functor that fuses bias addition and row-wise scaling.
 *
 * This operator expects callers to provide the row index associated with the
 * fragment being processed so that the correct scale factor can be applied.
 * The layout assumptions mirror CUTLASS defaults (row-major accumulators and
 * contiguous bias values).
 */
template <
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementBias,
    typename ElementRowScale,
    typename ElementCompute = ElementOutput,
    int ElementsPerAccess = 1,
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class EpilogueOpBiasRowScale {
 public:
  using FragmentOutput = cutlass::Array<ElementOutput, ElementsPerAccess>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, ElementsPerAccess>;
  using FragmentBias = cutlass::Array<ElementBias, ElementsPerAccess>;

  struct Arguments {
    ElementCompute alpha{ElementCompute(1)};
    ElementCompute beta{ElementCompute(0)};
    ElementBias const* bias{nullptr};
    ElementRowScale const* row_scales{nullptr};
    int64_t ld_bias{0};
    int64_t ld_row_scale{0};
  };

  struct Params {
    ElementCompute alpha{ElementCompute(1)};
    ElementCompute beta{ElementCompute(0)};
    ElementBias const* bias{nullptr};
    ElementRowScale const* row_scales{nullptr};
    int64_t ld_bias{0};
    int64_t ld_row_scale{0};

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const& args)
        : alpha(args.alpha),
          beta(args.beta),
          bias(args.bias),
          row_scales(args.row_scales),
          ld_bias(args.ld_bias),
          ld_row_scale(args.ld_row_scale) {}
  };

 private:
  ElementCompute alpha_;
  ElementCompute beta_;
  ElementBias const* bias_;
  ElementRowScale const* row_scales_;
  int64_t ld_bias_;
  int64_t ld_row_scale_;

 public:
  CUTLASS_HOST_DEVICE
  explicit EpilogueOpBiasRowScale(Params const& params)
      : alpha_(params.alpha),
        beta_(params.beta),
        bias_(params.bias),
        row_scales_(params.row_scales),
        ld_bias_(params.ld_bias),
        ld_row_scale_(params.ld_row_scale) {}

  CUTLASS_HOST_DEVICE
  EpilogueOpBiasRowScale(ElementCompute alpha = ElementCompute(1),
                         ElementCompute beta = ElementCompute(0),
                         ElementBias const* bias = nullptr,
                         ElementRowScale const* row_scales = nullptr,
                         int64_t ld_bias = 0,
                         int64_t ld_row_scale = 0)
      : alpha_(alpha),
        beta_(beta),
        bias_(bias),
        row_scales_(row_scales),
        ld_bias_(ld_bias),
        ld_row_scale_(ld_row_scale) {}

  CUTLASS_HOST_DEVICE
  void set_k_partition(int) {}

  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput& output,
                  FragmentAccumulator const& accum,
                  FragmentOutput const& source,
                  int row_idx,
                  int column_offset) const {
    output = apply(accum, source, row_idx, column_offset);
  }

  CUTLASS_HOST_DEVICE
  void operator()(FragmentOutput& output,
                  FragmentAccumulator const& accum,
                  FragmentOutput const& source,
                  int row_idx) const {
    operator()(output, accum, source, row_idx, 0);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput apply(FragmentAccumulator const& accum,
                       FragmentOutput const& source,
                       int row_idx,
                       int column_offset) const {
    FragmentOutput result;
    ElementCompute row_scale = ElementCompute(1);
    if (row_scales_) {
      int64_t scale_stride = ld_row_scale_ == 0 ? int64_t(1) : ld_row_scale_;
      row_scale = static_cast<ElementCompute>(row_scales_[row_idx * scale_stride]);
    }

    cutlass::NumericConverter<ElementOutput, ElementCompute, Round> converter;

    for (int i = 0; i < ElementsPerAccess; ++i) {
      ElementCompute acc = static_cast<ElementCompute>(accum[i]);
      ElementCompute bias = ElementCompute(0);
      if (bias_) {
        int64_t bias_stride = ld_bias_ == 0 ? int64_t(1) : ld_bias_;
        bias = static_cast<ElementCompute>(bias_[row_idx * bias_stride + column_offset + i]);
      }
      ElementCompute src = static_cast<ElementCompute>(source[i]);
      ElementCompute value = alpha_ * acc + bias;
      if (beta_ != ElementCompute(0)) {
        value += beta_ * src;
      }
      value *= row_scale;
      result[i] = converter(value);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput apply(FragmentAccumulator const& accum,
                       FragmentOutput const& source,
                       int row_idx) const {
    return apply(accum, source, row_idx, 0);
  }
};

}  // namespace cutlass_ext
}  // namespace lb

