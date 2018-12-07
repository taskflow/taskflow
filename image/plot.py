import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
costs = [5326, 1608, 920]

plt.subplot(131)
x = np.arange(3)
#plt.plot(costs, label='v1 (OpenMP)', color='black')
plt.bar(x, costs)
plt.title('Development Cost', fontsize=22)
plt.xlabel('task model', fontsize=18)
plt.ylabel('$USD', fontsize=16)
plt.xticks(x, ('OpenMP', 'TBB', 'Cpp-Taskflow'))
plt.legend()


# Fixing random state for reproducibility
raw = [line.strip() for line in open("graph_algorithm.txt")]

graph_sizes = []
omp_runtimes = []
tbb_runtimes = []
ctf_runtimes = []

# extract the output data
i = 0
for line in raw:
  token = line.split();
  assert(len(token) == 4)
  graph_sizes.append(int(token[0]))
  omp_runtimes.append(float(token[1]))
  tbb_runtimes.append(float(token[2]))
  ctf_runtimes.append(float(token[3]))
  i = i+1

plt.subplot(132)
plt.plot(graph_sizes, omp_runtimes, label='OpenMP')
plt.plot(graph_sizes, tbb_runtimes, label='TBB')
plt.plot(graph_sizes, ctf_runtimes, label='Cpp-Taskflow')
plt.title('Graph Algorithm', fontsize=22)
plt.xlabel('graph size (|V|+|E| ~ # tasks)', fontsize=18)
plt.ylabel('cpu runtime (ms)', fontsize=16)
#plt.xticks(np.arange(len(graph_sizes)), graph_sizes)
plt.legend(fontsize=16)



# Fixing random state for reproducibility
raw = [line.strip() for line in open("matrix_operation.txt")]

graph_sizes = []
omp_runtimes = []
tbb_runtimes = []
ctf_runtimes = []

# extract the output data
i = 0
for line in raw:
  token = line.split();
  assert(len(token) == 4)
  graph_sizes.append(int(token[0]))
  omp_runtimes.append(float(token[1]))
  tbb_runtimes.append(float(token[2]))
  ctf_runtimes.append(float(token[3]))
  i = i+1

plt.subplot(133)
plt.plot(graph_sizes, omp_runtimes, label='OpenMP')
plt.plot(graph_sizes, tbb_runtimes, label='TBB')
plt.plot(graph_sizes, ctf_runtimes, label='Cpp-Taskflow')
plt.title('Matrix Operation', fontsize=22)
plt.xlabel('partition count (# tasks)', fontsize=18)
plt.ylabel('cpu runtime (ms)', fontsize=16)
#plt.xticks(np.arange(len(graph_sizes)), graph_sizes)
plt.legend(fontsize=16)


plt.show()



