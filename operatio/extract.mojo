# operatio/extract.mojo

from max.tensor import Tensor
from time import perf_counter_ns
from algorithm.functional import vectorize
from testing import assert_equal

fn extract_weight_diff(base: Tensor[DType.float32], ft: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    start_time = perf_counter_ns()
    assert_equal(base.shape(), ft.shape(), "Tensors must have the same shape for subtraction.")
    output = Tensor[DType.float32](base.shape())
    num_elements = base.num_elements()
    alias simd_width = 4  # Adjust as needed

    @parameter
    fn vec_subtract[simd_width: Int](i: Int):
        base_vector = base.load[simd_width](i)
        ft_vector = ft.load[simd_width](i)
        result_vector = ft_vector - base_vector
        output.store[simd_width](i, result_vector)

    vectorizable_elements = (num_elements // simd_width) * simd_width
    vectorize[vec_subtract, simd_width](vectorizable_elements)

    # Handle remaining elements
    for i in range(vectorizable_elements, num_elements):
        output[i] = ft[i] - base[i]

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000
    print("Extracting weight difference took: ", elapsed_time, " ms")
    return output