tbb_include=/home/twhuang/PhD/Code/tbb/include/
tbb_library=/home/twhuang/PhD/Code/tbb/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_release/
taskflow_include=/home/twhuang/PhD/Code/cpp-taskflow/

default:
	g++ seq.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17  -o seq
	g++ omp.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17 -fopenmp -o omp
	g++ taskflow.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17 -lpthread -o taskflow
	g++ -std=c++17 -O2 tbb.cpp -o tbb -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt

clean:
	rm omp seq taskflow tbb
