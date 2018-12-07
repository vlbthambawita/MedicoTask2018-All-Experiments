#  Testing the saved model for accuracy and loss
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator




base_dir = 'data'
test_dir = os.path.join(base_dir, 'test')
model_dir = 'my_models'  # location of models
model_string = 'medical_v1_fine_tunned2018-07-04 12:47.h5'

model_fname = os.path.join(model_dir, model_string)
## Image proprocessing

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode='categorical')

## loading saved model
model = load_model(model_fname)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=80)
print('modle loaded')

print('test acc:', test_acc)
print('test loss:', test_loss)




