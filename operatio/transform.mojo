from max.tensor import Tensor
from time import perf_counter_ns
from sys.info import simdbitwidth
from algorithm import vectorize

fn transform_weights(
    pretrained: Tensor[DType.float32], 
    task_vector: Tensor[DType.float32], 
    scaling_coef: Float32
) raises -> Tensor[DType.float32]:
    
    if not pretrained.shape() == task_vector.shape():
        raise "Shape mismatch between pretrained and task vector tensors."

    if scaling_coef < 0:
        raise "Scaling coefficient must be non-negative."

    start_time = perf_counter_ns()

    updated_weights = Tensor[DType.float32](pretrained.shape())
    num_elements = pretrained.num_elements()
    
    alias simd_width = simdbitwidth() // 32  # 32 bits per float32

    @parameter
    fn vec_apply_task[simd_width: Int](i: Int):
        if i + simd_width <= num_elements:
            pretrained_vector = pretrained.load[simd_width](i)
            task_vector_scaled = task_vector.load[simd_width](i) * SIMD[DType.float32, simd_width](scaling_coef)
            updated_vector = pretrained_vector + task_vector_scaled
            updated_weights.store[simd_width](i, updated_vector)
        else:
            for j in range(i, num_elements):
                updated_weights[j] = pretrained[j] + scaling_coef * task_vector[j]

    vectorize[vec_apply_task, simd_width](num_elements)

    end_time = perf_counter_ns()
    elapsed_time = (end_time - start_time) / 1_000_000

    print("Transforming weights took: ", elapsed_time, " ms")
    return updated_weights