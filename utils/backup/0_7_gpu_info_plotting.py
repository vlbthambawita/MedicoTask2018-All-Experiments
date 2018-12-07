import matplotlib.pyplot as plt
import pandas
import sys
import os
import time


###################################################
gpu_info_file = sys.argv[1] # full path to the csv file
train_or_test = sys.argv[2] # training or testing

#gpu_info_file = 'data_v2/pytorch_gpu_info/7_1_pytorch_resnet18_testing_v5.py.csv'

data_dir = 'data_v2'
plot_dir  = data_dir + '/pytorch_plots'
gpu_mem_plot_name = train_or_test + "_memplot.png"
gpu_util_plot_name = train_or_test + "_utilization.png"

#####################################################
print("GPu infor plotting =", train_or_test)

data = pandas.read_csv(gpu_info_file)

gpu_utilization = data.iloc[:, 2]  # Gpu utilization
gpu_memory_utilization = data.iloc[:, 3]  # GPU memory utilization

####################################################

def plot_and_save(df, plot_name):
    pie = df.plot(title=plot_name, colormap='jet')
    fig_name = pie.get_figure()
    fig_name.savefig(os.path.join(plot_dir,plot_name))




plot_and_save(gpu_memory_utilization, gpu_mem_plot_name)

plt.figure() ## Must - other wise only one plot has two line

plot_and_save(gpu_utilization, gpu_util_plot_name)

print("GPU plotiing done = ", train_or_test)

