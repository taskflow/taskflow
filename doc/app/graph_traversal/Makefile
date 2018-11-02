tbb_include=/home/twhuang/PhD/Code/tbb/include/
tbb_library=/home/twhuang/PhD/Code/tbb/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_release/
taskflow_include=/home/twhuang/PhD/Code/cpp-taskflow/

default:
	g++ -c seq.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17
	g++ -c omp.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17 -fopenmp
	g++ -c taskflow.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17 
	g++ -c tbb.cpp -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17
	g++ -o run main.cpp omp.o tbb.o taskflow.o -O2 -I $(taskflow_include) -I $(tbb_include) -L $(tbb_library) -ltbb -lpthread -lrt -std=c++17 -fopenmp

clean:
	rm -rf run *.o
