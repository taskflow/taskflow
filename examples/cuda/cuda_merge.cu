#include <taskflow/cudaflow.hpp>

int main(int argc, char* argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./cuda_merge N\n";
    std::exit(EXIT_FAILURE);
  }

  unsigned N = std::atoi(argv[1]);
  
  // gpu data
  auto da = tf::cuda_malloc_shared<int>(N);
  auto db = tf::cuda_malloc_shared<int>(N);
  auto dc = tf::cuda_malloc_shared<int>(N + N);

  // host data
  std::vector<int> ha(N), hb(N), hc(N + N);

  for(unsigned i=0; i<N; i++) {
    da[i] = ha[i] = rand()%100;
    db[i] = hb[i] = rand()%100;
  }
  
  std::sort(da, da+N);
  std::sort(db, db+N);
  std::sort(ha.begin(), ha.end());
  std::sort(hb.begin(), hb.end());

  // --------------------------------------------------------------------------
  // GPU merge
  // --------------------------------------------------------------------------

  auto beg = std::chrono::steady_clock::now();

  // allocate the buffer
  auto bufsz = tf::cuda_merge_buffer_size<tf::cudaDefaultExecutionPolicy>(N, N);
  tf::cudaScopedDeviceMemory<std::byte> buf(bufsz);

  tf::cuda_merge(tf::cudaDefaultExecutionPolicy{}, 
    da, da+N, db, db+N, dc, tf::cuda_less<int>{}, buf.data()
  );
  cudaStreamSynchronize(0);
  auto end = std::chrono::steady_clock::now();

  std::cout << "GPU merge: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";
  
  // --------------------------------------------------------------------------
  // CPU merge
  // --------------------------------------------------------------------------
  beg = std::chrono::steady_clock::now();
  std::merge(ha.begin(), ha.end(), hb.begin(), hb.end(), hc.begin());
  end = std::chrono::steady_clock::now();
  
  std::cout << "CPU merge: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";

  // --------------------------------------------------------------------------
  // verify the result
  // --------------------------------------------------------------------------

  //for(unsigned i=0; i< N; i++) {
  //  printf("a[%u]=%d, b[%u]=%d\n", i, a[i], i, b[i]);
  //}
  //printf("\n");

  //for(unsigned i=0; i<N+N; i++) {
  //  printf("c[%u]=%d\n", i, c[i]);
  //}
  
  for(size_t i=0; i<N; i++) {
    if(dc[i] != hc[i]) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";

  cudaDeviceSynchronize();

};
