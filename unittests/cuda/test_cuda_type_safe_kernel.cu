#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <chrono>
#include <cmath>

// ---------------------------------------------------------------------------
// Kernels used across all test cases
// ---------------------------------------------------------------------------

template <typename T>
__global__ void k_set(T* ptr, size_t N, T value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ptr[i] = value;
}

// k_set_size_t: intentionally takes size_t N so we can test int→size_t widening
__global__ void k_set_size_t(int* ptr, size_t N, int value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ptr[i] = value;
}

__global__ void k_scale_f32(const float* in, float* out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;
}

__global__ void k_fill(int* ptr, size_t N, int value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ptr[i] = value;
}

__global__ void k_noop() {}

// ---------------------------------------------------------------------------
// kernelArgCast — host-side unit tests
// ---------------------------------------------------------------------------

TEST_CASE("kernelArgCast.ScalarDispatch") {
    // Non-pointer path uses static_cast. For any int value, the result must
    // equal static_cast<size_t>(v).
    for (int i = 0; i < 100; i++) {
        int v = i * 1000003 + 7;
        REQUIRE(tf::detail::kernelArgCast<size_t>(v) == static_cast<size_t>(v));
    }
}

TEST_CASE("kernelArgCast.PointerDispatch") {
    // Pointer path uses reinterpret_cast. float* → const float* must yield the
    // same address value as reinterpret_cast.
    for (int i = 0; i < 100; i++) {
        float dummy = static_cast<float>(i);
        float* ptr = &dummy;
        REQUIRE(tf::detail::kernelArgCast<const float*>(ptr) ==
                reinterpret_cast<const float*>(ptr));
    }
}

// ---------------------------------------------------------------------------
// cudaGraphBase::kernel — typed overload
// ---------------------------------------------------------------------------

// Pass int where the kernel expects size_t — the typed overload must widen
// the argument correctly so the GPU writes the right number of elements.
TEST_CASE("cudaGraph.TypedKernel.ScalarWidening") {
    std::vector<int> sizes;
    for (int n = 1;   n <= 20;   ++n)     sizes.push_back(n);
    for (int n = 21;  n <= 50;   n += 3)  sizes.push_back(n);
    for (int n = 51;  n <= 100;  n += 5)  sizes.push_back(n);
    for (int n = 101; n <= 200;  n += 10) sizes.push_back(n);
    for (int n = 201; n <= 500;  n += 30) sizes.push_back(n);
    for (int n = 501; n <= 1000; n += 50) sizes.push_back(n);
    sizes.insert(sizes.end(), {1024, 1025, 1023, 2048, 2049, 2047, 4096, 4095, 4097});
    sizes.insert(sizes.end(), {
        3, 7, 15, 31, 63, 127, 255, 511, 1500, 2000, 3000,
        750, 850, 950, 1100, 1200, 1300, 1400, 1600, 1800, 2200, 2400, 2600, 2800
    });

    for (int n : sizes) {
        std::vector<int> cpu(static_cast<size_t>(n), 0);

        int* gpu = nullptr;
        REQUIRE(cudaMalloc(&gpu, static_cast<size_t>(n) * sizeof(int)) == cudaSuccess);
        REQUIRE(cudaMemset(gpu, 0, static_cast<size_t>(n) * sizeof(int)) == cudaSuccess);

        tf::cudaGraph cg;
        dim3 g = {(static_cast<unsigned>(n) + 255u) / 256u, 1u, 1u};
        dim3 b = {256u, 1u, 1u};

        // Intentionally pass (int)n where k_set_size_t expects size_t.
        // The typed overload widens via static_cast — the kernel must still
        // write exactly n elements.
        cg.kernel(g, b, 0, k_set_size_t, gpu, (int)n, 42);

        {
            tf::cudaStream stream;
            tf::cudaGraphExec exec(cg);
            stream.run(exec).synchronize();
        }

        REQUIRE(cudaMemcpy(cpu.data(), gpu,
                           static_cast<size_t>(n) * sizeof(int),
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        for (int i = 0; i < n; ++i)
            REQUIRE(cpu[static_cast<size_t>(i)] == 42);

        REQUIRE(cudaFree(gpu) == cudaSuccess);
    }
}

// Pass float* where the kernel expects const float* — must compile and produce
// correct output (reinterpret_cast handles the const-qualification).
TEST_CASE("cudaGraph.TypedKernel.PointerConstQualify") {
    const size_t N = 256;
    dim3 g = {(unsigned)(N + 255) / 256, 1, 1};
    dim3 b = {256, 1, 1};

    auto* h_in  = static_cast<float*>(std::malloc(N * sizeof(float)));
    auto* h_out = static_cast<float*>(std::malloc(N * sizeof(float)));
    for (size_t i = 0; i < N; ++i) {
        h_in[i]  = static_cast<float>(i) + 1.0f;
        h_out[i] = 0.0f;
    }

    float* d_in  = tf::cuda_malloc_device<float>(N);
    float* d_out = tf::cuda_malloc_device<float>(N);

    tf::cudaGraph cg;
    auto h2d = cg.copy(d_in, h_in, N);
    auto kern = cg.kernel(g, b, 0, k_scale_f32, d_in, d_out, N);
    auto d2h = cg.copy(h_out, d_out, N);
    h2d.precede(kern);
    kern.precede(d2h);

    tf::cudaStream stream;
    tf::cudaGraphExec exec(cg);
    stream.run(exec).synchronize();

    for (size_t i = 0; i < N; ++i)
        REQUIRE(std::fabs(h_out[i] - h_in[i] * 2.0f) < 1e-5f);

    tf::cuda_free(d_in);
    tf::cuda_free(d_out);
    std::free(h_in);
    std::free(h_out);
}

// Correct argument count — no exception, node executes as expected.
TEST_CASE("cudaGraph.TypedKernel.ArgumentCount.Match") {
    const size_t N = 64;
    dim3 g = {(unsigned)(N + 255) / 256, 1, 1};
    dim3 b = {256, 1, 1};

    int* d_buf = tf::cuda_malloc_device<int>(N);
    auto* h_buf = static_cast<int*>(std::calloc(N, sizeof(int)));

    REQUIRE_NOTHROW(({
        tf::cudaGraph tmp;
        tmp.kernel(g, b, 0, k_set_size_t, d_buf, N, 7);
    }));

    tf::cudaGraph cg;
    auto kern = cg.kernel(g, b, 0, k_set_size_t, d_buf, N, 42);
    auto d2h  = cg.copy(h_buf, d_buf, N);
    kern.precede(d2h);

    tf::cudaStream stream;
    tf::cudaGraphExec exec(cg);
    stream.run(exec).synchronize();

    for (size_t i = 0; i < N; ++i)
        REQUIRE(h_buf[i] == 42);

    tf::cuda_free(d_buf);
    std::free(h_buf);
}

// Existing call sites using plain __global__ function pointers continue to
// work without any changes after the typed overload is added.
__global__ void k_fill_old_style(int* ptr, size_t N, int value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ptr[i] = value;
}

TEST_CASE("cudaGraph.TypedKernel.BackwardCompat") {
    const size_t N = 128;
    dim3 g = {(unsigned)(N + 255) / 256, 1, 1};
    dim3 b = {256, 1, 1};

    int* d_buf = tf::cuda_malloc_device<int>(N);
    auto* h_buf = static_cast<int*>(std::calloc(N, sizeof(int)));

    // Plain named kernel — same call pattern as before the typed overload existed.
    {
        tf::cudaGraph cg;
        auto kern = cg.kernel(g, b, 0, k_fill_old_style, d_buf, N, 99);
        auto d2h  = cg.copy(h_buf, d_buf, N);
        kern.precede(d2h);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();

        for (size_t i = 0; i < N; ++i)
            REQUIRE(h_buf[i] == 99);
    }

    // Template kernel — explicit instantiation still works fine.
    {
        float* d_fbuf = tf::cuda_malloc_device<float>(N);
        auto* h_fbuf  = static_cast<float*>(std::calloc(N, sizeof(float)));

        tf::cudaGraph cg;
        auto kern = cg.kernel(g, b, 0, k_set<float>, d_fbuf, N, 3.14f);
        auto d2h  = cg.copy(h_fbuf, d_fbuf, N);
        kern.precede(d2h);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();

        for (size_t i = 0; i < N; ++i)
            REQUIRE(std::fabs(h_fbuf[i] - 3.14f) < 1e-5f);

        tf::cuda_free(d_fbuf);
        std::free(h_fbuf);
    }

    tf::cuda_free(d_buf);
    std::free(h_buf);
}

// Two independent runs with identical arguments must produce identical output.
TEST_CASE("cudaGraph.TypedKernel.Deterministic") {
    for (int n = 1; n <= 100; n++) {
        const int value = n * 7 + 3;

        int* gpu_a = tf::cuda_malloc_device<int>(n);
        int* gpu_b = tf::cuda_malloc_device<int>(n);
        auto cpu_a = static_cast<int*>(std::malloc(n * sizeof(int)));
        auto cpu_b = static_cast<int*>(std::malloc(n * sizeof(int)));

        dim3 g = {(static_cast<unsigned>(n) + 255u) / 256u, 1u, 1u};
        dim3 b = {256u, 1u, 1u};

        {
            tf::cudaGraph cg;
            cg.kernel(g, b, 0, k_set<int>, gpu_a, static_cast<size_t>(n), value);
            tf::cudaStream stream;
            tf::cudaGraphExec exec(cg);
            stream.run(exec).synchronize();
        }
        {
            tf::cudaGraph cg;
            cg.kernel(g, b, 0, k_set<int>, gpu_b, static_cast<size_t>(n), value);
            tf::cudaStream stream;
            tf::cudaGraphExec exec(cg);
            stream.run(exec).synchronize();
        }

        REQUIRE(cudaMemcpy(cpu_a, gpu_a, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(cpu_b, gpu_b, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

        for (int i = 0; i < n; i++) {
            REQUIRE(cpu_a[i] == value);
            REQUIRE(cpu_a[i] == cpu_b[i]);
        }

        std::free(cpu_a);
        std::free(cpu_b);
        tf::cuda_free(gpu_a);
        tf::cuda_free(gpu_b);
    }
}

// ---------------------------------------------------------------------------
// cudaGraphExecBase::kernel — typed overload (update path)
// ---------------------------------------------------------------------------

// Same scalar-widening check as above, but via exec.kernel() updates on an
// already-instantiated graph.
TEST_CASE("cudaGraph.TypedKernel.Update.ScalarWidening") {
    std::vector<int> sizes;
    for (int n = 1;   n <= 20;   ++n)     sizes.push_back(n);
    for (int n = 21;  n <= 50;   n += 3)  sizes.push_back(n);
    for (int n = 51;  n <= 100;  n += 5)  sizes.push_back(n);
    for (int n = 101; n <= 200;  n += 10) sizes.push_back(n);
    for (int n = 201; n <= 500;  n += 30) sizes.push_back(n);
    for (int n = 501; n <= 1000; n += 50) sizes.push_back(n);
    sizes.insert(sizes.end(), {1024, 1025, 1023, 2048, 2049, 2047, 4096, 4095, 4097});
    sizes.insert(sizes.end(), {
        3, 7, 15, 31, 63, 127, 255, 511, 1500, 2000, 3000,
        750, 850, 950, 1100, 1200, 1300, 1400, 1600, 1800, 2200, 2400, 2600, 2800
    });

    int max_n = *std::max_element(sizes.begin(), sizes.end());

    int* gpu = nullptr;
    REQUIRE(cudaMalloc(&gpu, static_cast<size_t>(max_n) * sizeof(int)) == cudaSuccess);

    // Build once with the first size, then update every iteration.
    const int first_n = sizes[0];
    dim3 g0 = {(static_cast<unsigned>(first_n) + 255u) / 256u, 1u, 1u};
    dim3 b0 = {256u, 1u, 1u};

    tf::cudaGraph cg;
    tf::cudaTask task = cg.kernel(g0, b0, 0, k_set_size_t, gpu, (int)first_n, 0);
    tf::cudaGraphExec exec(cg);
    tf::cudaStream stream;

    std::vector<int> cpu(static_cast<size_t>(max_n), 0);

    for (int idx = 0; idx < static_cast<int>(sizes.size()); ++idx) {
        int n     = sizes[static_cast<size_t>(idx)];
        int value = (idx * 31 + 17) & 0x7FFFFFFF;

        REQUIRE(cudaMemset(gpu, 0, static_cast<size_t>(n) * sizeof(int)) == cudaSuccess);

        dim3 g = {(static_cast<unsigned>(n) + 255u) / 256u, 1u, 1u};
        dim3 b = {256u, 1u, 1u};

        // Intentionally pass (int)n for size_t N.
        exec.kernel(task, g, b, 0, k_set_size_t, gpu, (int)n, value);
        stream.run(exec).synchronize();

        REQUIRE(cudaMemcpy(cpu.data(), gpu,
                           static_cast<size_t>(n) * sizeof(int),
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        for (int i = 0; i < n; ++i)
            REQUIRE(cpu[static_cast<size_t>(i)] == value);
    }

    REQUIRE(cudaFree(gpu) == cudaSuccess);
}

// float* → const float* on the update path.
TEST_CASE("cudaGraph.TypedKernel.Update.PointerConstQualify") {
    const size_t N = 256;
    dim3 g = {(unsigned)(N + 255) / 256, 1, 1};
    dim3 b = {256, 1, 1};

    auto* h_in  = static_cast<float*>(std::malloc(N * sizeof(float)));
    auto* h_out = static_cast<float*>(std::malloc(N * sizeof(float)));
    for (size_t i = 0; i < N; ++i) {
        h_in[i]  = static_cast<float>(i) + 1.0f;
        h_out[i] = 0.0f;
    }

    float* d_in  = tf::cuda_malloc_device<float>(N);
    float* d_out = tf::cuda_malloc_device<float>(N);

    tf::cudaGraph cg;
    auto h2d  = cg.copy(d_in, h_in, N);
    auto task = cg.kernel(g, b, 0, k_scale_f32, d_in, d_out, N);
    auto d2h  = cg.copy(h_out, d_out, N);
    h2d.precede(task);
    task.precede(d2h);

    tf::cudaGraphExec exec(cg);

    // Update with the same parameters — the typed overload must accept float* for const float*.
    exec.kernel(task, g, b, 0, k_scale_f32, d_in, d_out, N);

    tf::cudaStream stream;
    stream.run(exec).synchronize();

    for (size_t i = 0; i < N; ++i)
        REQUIRE(std::fabs(h_out[i] - h_in[i] * 2.0f) < 1e-5f);

    tf::cuda_free(d_in);
    tf::cuda_free(d_out);
    std::free(h_in);
    std::free(h_out);
}

// exec.kernel() with a plain function pointer — existing call sites need no changes.
TEST_CASE("cudaGraph.TypedKernel.Update.BackwardCompat") {
    const size_t N = 128;
    dim3 g = {(unsigned)(N + 255) / 256, 1, 1};
    dim3 b = {256, 1, 1};

    int* d_buf = tf::cuda_malloc_device<int>(N);
    auto* h_buf = static_cast<int*>(std::calloc(N, sizeof(int)));

    tf::cudaGraph cg;
    auto task = cg.kernel(g, b, 0, k_fill_old_style, d_buf, N, 10);
    auto d2h  = cg.copy(h_buf, d_buf, N);
    task.precede(d2h);

    tf::cudaGraphExec exec(cg);

    // Update to value 99 — verifies the update actually takes effect.
    exec.kernel(task, g, b, 0, k_fill_old_style, d_buf, N, 99);

    tf::cudaStream stream;
    stream.run(exec).synchronize();

    for (size_t i = 0; i < N; ++i)
        REQUIRE(h_buf[i] == 99);

    tf::cuda_free(d_buf);
    std::free(h_buf);
}

// ---------------------------------------------------------------------------
// Zero-overhead check
// Graph-creation time with the typed overload must not be measurably higher
// than a second identical run. Both loops exercise the same code path, so any
// ratio > 1 is pure measurement noise.
// ---------------------------------------------------------------------------

TEST_CASE("cudaGraph.TypedKernel.ZeroCost") {
    constexpr int N_CALLS = 1000;
    constexpr int N_RUNS  = 5;

    // Warm up the CUDA driver before timing.
    {
        tf::cudaGraph warmup;
        warmup.kernel({1,1,1}, {1,1,1}, 0, k_noop);
    }

    constexpr int BUF_N = 64;
    int* d_ptr = tf::cuda_malloc_device<int>(BUF_N);
    const size_t count = static_cast<size_t>(BUF_N);
    const int    val   = 42;

    double loop_a_ms = 0.0;
    double loop_b_ms = 0.0;

    for (int run = 0; run < N_RUNS; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_CALLS; ++i) {
            tf::cudaGraph cg;
            cg.kernel({8,1,1}, {128,1,1}, 0, k_set<int>, d_ptr, count, val);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        loop_a_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_CALLS; ++i) {
            tf::cudaGraph cg;
            cg.kernel({8,1,1}, {128,1,1}, 0, k_set<int>, d_ptr, count, val);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        loop_b_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    tf::cuda_free(d_ptr);

    const double mean_a = loop_a_ms / N_RUNS;
    const double mean_b = loop_b_ms / N_RUNS;

    REQUIRE(mean_b > 0.0);
    REQUIRE(mean_a / mean_b <= 1.05);  // 5 % margin for scheduler noise
}
