import os
import shutil


#  ######################################################
#  ## Generating data sets from the original data set
#  ######################################################

#original_dataset_dir = '/home/vajira/simula/Datasets/kvasir_v2_preprocessed_borders_navbox_removed'  # original data location path
#data_dir = 'data'  # location to put sub samples of data (making small data set from large one)
#base_dir = 'data/medical_image_v1'  # change this one to create new small datasets

def make_folder_structure(base_dir, main_classification_types_dirs):

    train_dir_list = []
    validation_dir_list = []
    test_dir_list = []


    if not os.path.exists(base_dir):
        os.mkdir(base_dir)  # new directory to make directory structure and load data

    main_directory_names = ['train', 'validation', 'test']

    for main_dir_name in main_directory_names:
        dir = os.path.join(base_dir, main_dir_name)
        if not os.path.exists(dir):
            os.mkdir(dir)

        for main_classification_types_dir in main_classification_types_dirs:
            type_dir = os.path.join(dir, main_classification_types_dir)

            if main_dir_name == 'train':
                train_dir_list.append(type_dir)

            elif main_dir_name == 'validation':
                validation_dir_list.append(type_dir)

            elif main_dir_name == 'test':
                test_dir_list.append(type_dir)

            if not os.path.exists(type_dir):
                os.mkdir(type_dir)

    return train_dir_list, validation_dir_list, test_dir_list



def load_data_to_folder(original_data_folder, # original data folder
                        base_dir,  # folder to create training, validation and testing data
                        main_classification_types,  # main class types of the problem
                        size_of_training,  # number of images to training set
                        size_of_validation,  # number of images to validation set
                        size_of_testing):  # number of images to testing set

    for class_type in main_classification_types:
        src_dir = os.path.join(original_data_folder, class_type)
        files = os.listdir(src_dir)

        training_files = files[0:size_of_training]
        validation_files = files[size_of_training:size_of_training + size_of_validation]
        testing_files = files[size_of_training + size_of_validation:size_of_training + size_of_validation + size_of_testing]

        print(len(files))

        # copying training images
        for file in training_files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(base_dir + '/train/' + class_type, file)
            if not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)

        # copying validaton images
        for file in validation_files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(base_dir + '/validation/' + class_type, file)
            if not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)

        # copying testing images
        for file in testing_files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(base_dir + '/test/' + class_type, file)
            if not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)

        #training_dir = os.path.join(base_dir, 'train')



