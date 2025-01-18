import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
gpu_data = pd.read_csv('gpu_update_times.csv', header=None, names=['swarm_size', 'avg_time', 'std_dev'])
cpu_data = pd.read_csv('cpu_update_times.csv', header=None, names=['swarm_size', 'avg_time', 'std_dev'])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot GPU data
plt.errorbar(gpu_data['swarm_size'], gpu_data['avg_time'], yerr=gpu_data['std_dev'], 
             fmt='o-', label='GPU', capsize=5, markersize=5, color='blue')

# Plot CPU data
plt.errorbar(cpu_data['swarm_size'], cpu_data['avg_time'], yerr=cpu_data['std_dev'], 
             fmt='o-', label='CPU', capsize=5, markersize=5, color='orange')

# Add title and labels
# plt.title('Performance Comparison: GPU vs CPU')
plt.xlabel('Swarm Size')
plt.ylabel('Average Time (microseconds)')
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()