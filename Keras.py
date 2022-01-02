from keras.models import load_model
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np



model = load_model('cancerCNN.h5')
config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
img = image.load_img('./upload\person1946_bacteria_4875.jpeg', target_size=(40, 40),color_mode="grayscale")
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
#img_preprocessed = preprocess_input(img_batch)


for i in range(0,len(img_batch[0][0])):
    img_batch[0][0][i][0] = img_batch[0][0][i][0]*(1/255)




      
prediction = model.predict(img_batch)

print(prediction)
