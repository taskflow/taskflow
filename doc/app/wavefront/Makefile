tbb_include=/home/twhuang/PhD/Code/tbb/include/
tbb_library=/home/twhuang/PhD/Code/tbb/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_release/
taskflow_include=/home/twhuang/PhD/Code/cpp-taskflow/

default:
	g++ seq.cpp -O2 -o seq
	g++ omp.cpp -O2 -fopenmp -o omp
	g++ taskflow.cpp -O2 -I $(taskflow_include) -std=c++1z -lpthread -o taskflow
	g++ -std=c++17 -O2 tbb.cpp -o tbb -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt

clean:
	rm omp seq taskflow tbb
