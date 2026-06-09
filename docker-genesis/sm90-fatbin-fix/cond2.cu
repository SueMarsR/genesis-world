#include <cuda_runtime.h>
// Exact mirror of quadrants _qd_graph_do_while_cond:
//   param0: cudaGraphConditionalHandle handle
//   param1: int** pflag   (pointer to pointer to the i32 counter)
//   sets conditional = (**pflag != 0)
extern "C" __global__ void _qd_graph_do_while_cond(
        cudaGraphConditionalHandle handle, int** pflag) {
    int v = **pflag;
    cudaGraphSetConditional(handle, v != 0 ? 1u : 0u);
}
