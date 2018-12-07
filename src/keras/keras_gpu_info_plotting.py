

#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
import matplotlib as mpl
mpl.use('Agg')  # This has to run before pyplot import

import matplotlib.pyplot as plt
import pandas
import sys
import os
import time


###################################################
gpu_info_file = sys.argv[1] # full path to the csv file
train_or_test = sys.argv[2] # training or testing
main_python_code_file_name = sys.argv[3]  # Tha main python file name
main_dir = sys.argv[4]  # main data directory


gpu_info_types = ['index',
 ' timestamp',
 ' utilization.gpu [%]',
 ' utilization.memory [%]',
 ' temperature.gpu',
 ' power.draw [W]',
 ' clocks.current.sm [MHz]',
 ' clocks.current.memory [MHz]',
 ' clocks.current.graphics [MHz]']

gpu_info_types_selected = gpu_info_types[2:4]  # selecting only gpu utilization and memory utilization




#gpu_info_file = 'data_v2/pytorch_gpu_info/7_1_pytorch_resnet18_testing_v5.py.csv'

data_dir = main_dir
plot_dir  = data_dir + '/keras_plots'
gpu_mem_plot_name = train_or_test + "_memplot.png"
gpu_util_plot_name = train_or_test + "_utilization.png"

#####################################################
print("GPu infor plotting =", train_or_test)

data = pandas.read_csv(gpu_info_file)  # collecting whole data

#########################################################
no_of_gpus = pandas.value_counts(data['index']).shape[0]  # No of GPUs recorded

#  separating data into separate data frames for each GPU
gpu_data_list = []  # Dataframe list for storing data of each GPU
for i in range(no_of_gpus):
    print(i)
    gpu_data_list.append(data.loc[data['index'] == i])

##########################################################
## Plotiing - seperate
####################################################

for info_type in gpu_info_types_selected:  # selecting information type for plotting
    for gpu in gpu_data_list:  # dataframe list with each gpu informations
        # print(gpu)
        # print(info_type)
        temp_data = gpu[info_type].reset_index(drop=True)
        fig = temp_data.plot(title=info_type)
        fig.legend(list(range(no_of_gpus)))
        fig.set_xlabel("Time (s)")

        ## Saving figure
        fig_to_save = fig.get_figure()
        plot_name = main_python_code_file_name + "_" + train_or_test + info_type + '.png'
        fig_to_save.savefig(os.path.join(plot_dir, plot_name)) ## Want to change for a good name
        # print(temp_data)

    plt.figure()  # making new figure


print("GPU plotiing done = ", train_or_test)

