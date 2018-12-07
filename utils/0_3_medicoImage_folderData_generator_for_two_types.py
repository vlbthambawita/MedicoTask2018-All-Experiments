import os
import shutil
import math


base_dir = '/home/vajira/simula/code_samples/mediaEval_2018_structured_v2/data/CVC-356'


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
#test_dir = os.path.join(base_dir, 'test')


main_classification_types = [
                            #'stool-plenty',
                            'polyps',
                            #'colon-clear',
                            #'retroflex-rectum',
                            #'dyed-lifted-polyps',
                            #'instruments',
                            #'normal-pylorus',
                            #'stool-inclusions',
                            #'blurry-nothing',
                            #'out-of-patient',
                            'non-polyps',
                            #'retroflex-stomach',
                            #'ulcerative-colitis',
                            #'polyps',
                            #'dyed-resection-margins',
                            #'normal-cecum'
                            ]

# original data location path
original_data_folder_name = '/home/vajira/simula/code_samples/mediaEval_2018_structured_v2/data/CVC-356/all'


train_size = 70  # 70%
validation_size = 30 #  30 %
test_size = 0  # 0 %


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


########################################################################

def load_data_to_folder(original_data_folder,  # original data folder
                        base_dir,  # folder to create training, validation and testing data
                        main_classification_types,  # main class types of the problem
                        size_of_training_percentage,  # percentage %
                        size_of_validation_percentage,  # percentage %
                        size_of_testing_percentage):  # percentage %

    for class_type in main_classification_types:
        src_dir = os.path.join(original_data_folder, class_type)
        files = os.listdir(src_dir)

        size_of_training = math.floor(len(files) * (size_of_training_percentage / 100))
        size_of_validation = math.ceil(len(files) * (size_of_validation_percentage / 100))
        size_of_testing = math.ceil(len(files) * (size_of_testing_percentage / 100))

        print("Total files =", len(files))
        print("Training files=", size_of_training)
        print("Validation files=", size_of_validation)
        print("Testing files=", size_of_testing)

        training_files = files[0:size_of_training]
        validation_files = files[size_of_training:size_of_training + size_of_validation]
        testing_files = files[
                        size_of_training + size_of_validation:size_of_training + size_of_validation + size_of_testing]

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

        # training_dir = os.path.join(base_dir, 'train')


make_folder_structure(base_dir, main_classification_types)

load_data_to_folder(original_data_folder_name, base_dir, main_classification_types, train_size, validation_size, test_size)

print("data loading finised..")