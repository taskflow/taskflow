import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
costs = [5326, 1408, 920]

plt.subplot(121)
x = np.arange(3)
#plt.plot(costs, label='v1 (OpenMP)', color='black')
plt.bar(x, costs)
plt.title('Development Cost', fontsize=22)
plt.xlabel('task model', fontsize=22)
plt.ylabel('$USD', fontsize=22)
plt.xticks(x, ('OpenMP', 'TBB', 'Cpp-Taskflow'))
plt.legend()

# Fixing random state for reproducibility
raw = [line.strip() for line in open("dataset")]

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


plt.subplot(122)
plt.plot(graph_sizes, omp_runtimes, label='OpenMP')
plt.plot(graph_sizes, tbb_runtimes, label='TBB')
plt.plot(graph_sizes, ctf_runtimes, label='Cpp-Taskflow')
plt.title('Runtime versus Graph Size', fontsize=22)
plt.xlabel('graph size (|V|+|E|)', fontsize=22)
plt.ylabel('cpu runtime (ms)', fontsize=22)
#plt.xticks(np.arange(len(graph_sizes)), graph_sizes)
plt.legend(fontsize=22)
plt.show()

