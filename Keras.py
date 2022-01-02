from keras.models import load_model
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np



model = load_model('cancerCNN.h5')
print(model.summary())
img = image.load_img('./upload\person1946_bacteria_4875.jpeg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_preprocessed)

print(prediction)
