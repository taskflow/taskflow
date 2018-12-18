import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

plt.subplots_adjust(hspace=0.23)

# dev cost for wavefront
costs = [1507, 872, 680, 306]

plt.subplot(221)
x = np.arange(4)
#plt.plot(costs, label='v1 (OpenMP)', color='black')
plt.bar(x, costs)
plt.title('Dev Cost (matrix wavefront)', fontsize=22)
#plt.xlabel('task model', fontsize=18)
plt.ylabel('$USD', fontsize=16)
plt.xticks(x, ('OpenMP', 'TBB', 'Cpp-Taskflow', 'seq'), fontsize=14)
plt.legend()


# Dev cost for graph traversal
costs = [5326, 1384, 920, 306]

plt.subplot(222)
x = np.arange(4)
#plt.plot(costs, label='v1 (OpenMP)', color='black')
plt.bar(x, costs)
plt.title('Dev Cost (graph traversal)', fontsize=22)
#plt.xlabel('task model', fontsize=18)
plt.ylabel('$USD', fontsize=18)
plt.xticks(x, ('OpenMP', 'TBB', 'Cpp-Taskflow', 'seq'), fontsize=14)
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

plt.subplot(224)
plt.plot(graph_sizes, omp_runtimes, marker='.', label='OpenMP')
plt.plot(graph_sizes, tbb_runtimes, marker='^', label='TBB')
plt.plot(graph_sizes, ctf_runtimes, marker='x', label='Cpp-Taskflow')
plt.title('Runtime (graph traversal)', fontsize=22)
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

plt.subplot(223)
plt.plot(graph_sizes, omp_runtimes, marker='.', label='OpenMP')
plt.plot(graph_sizes, tbb_runtimes, marker='^', label='TBB')
plt.plot(graph_sizes, ctf_runtimes, marker='x', label='Cpp-Taskflow')
plt.title('Runtime (matrix wavefront)', fontsize=22)
plt.xlabel('partition count (# tasks)', fontsize=18)
plt.ylabel('cpu runtime (ms)', fontsize=16)
#plt.xticks(np.arange(len(graph_sizes)), graph_sizes)
plt.legend(fontsize=16)


plt.show()



