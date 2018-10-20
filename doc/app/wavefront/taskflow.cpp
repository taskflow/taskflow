#include <cmath>
#include <taskflow/taskflow.hpp> 

int M = 40000, N = 40000;
int B = 160;
int MB = (M/B) + (M%B>0);
int NB = (N/B) + (N%B>0);

double **matrix {nullptr};

// nominal operations
double calc(double v0, double v1) {
  return (v0 == v1) ? std::pow(v0/v1, 4.0f) : std::max(v0,v1);
}

// initialize the matrix
void init_matrix(){
  matrix = new double *[M];
  for ( int i = 0; i < M; ++i ) matrix[i] = new double [N];
  for(int i=0; i<M; ++i){
    for(int j=0; j<N ; ++j){
      matrix[i][j] = i*N + j;
    }   
  }
}

// destroy the matrix
void destroy_matrix() {
  for ( int i = 0; i < M; ++i ) {
    delete [] matrix[i];
  }
  delete [] matrix;
}

// wavefront computing
void wavefront() {

  tf::Taskflow tf;

  std::vector<std::vector<tf::Task>> node(MB);

  for(auto &n : node){
    for(size_t i=0; i<NB; i++){
      n.emplace_back(tf.placeholder());
    }
  }
  
  matrix[M-1][N-1] = 0;
  for( int i=MB; --i>=0; ) {
    for( int j=NB; --j>=0; ) {
      node[i][j].work(
        [i=i, j=j]() {
          int start_i = i*B;
          int end_i = (i*B+B > M) ? M : i*B+B;
          int start_j = j*B;
          int end_j = (j*B+B > N) ? N : j*B+B;
          for ( int ii = start_i; ii < end_i; ++ii ) {
            for ( int jj = start_j; jj < end_j; ++jj ) {
              double v0 = ii == 0 ? 0 : matrix[ii-1][jj];
              double v1 = jj == 0 ? 0 : matrix[ii][jj-1];
              matrix[ii][jj] = ii==0 && jj==0 ? 1 : calc(v0,v1);
            }
          }
        }
      );
      if(j+1 < NB) node[i][j].precede(node[i][j+1]);
      if(i+1 < MB) node[i][j].precede(node[i+1][j]);
    }
  }

  tf.wait_for_all();
}

// main function
int main(int argc, char *argv[]) {

  init_matrix();

  auto beg = std::chrono::high_resolution_clock::now();
  wavefront();
  auto end = std::chrono::high_resolution_clock::now();
  
  std::cout << "Taskflow wavefront elapsed time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() 
            << " ms\n"
            << "result: " << matrix[M-1][N-1] << std::endl;

  destroy_matrix();

  return 0;
}




