import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
memcpy_data = pd.read_csv('memcpy_times.csv', header=None, names=['swarm_size', 'avg_time', 'std_dev'])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot GPU data
plt.errorbar(memcpy_data['swarm_size'], memcpy_data['avg_time'], yerr=memcpy_data['std_dev'], 
             fmt='o-', label='cudaMemcpy', capsize=5, markersize=5, color='green')

# # Plot CPU data
# plt.errorbar(cpu_data['swarm_size'], cpu_data['avg_time'], yerr=cpu_data['std_dev'], 
#              fmt='o-', label='CPU', capsize=5, markersize=5, color='orange')

# Add title and labels
# plt.title('Performance Comparison: GPU vs CPU')
plt.xlabel('Swarm Size')
plt.ylabel('Average Time (ms)')
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()