from flask import Flask,request
from flask.templating import render_template
from keras.models import load_model
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

UPLOAD_FOLDER = './upload'




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER







global model
model = load_model('cancerCNN.h5')







def ProcessInput(image):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    return classes
        








@app.route("/",methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        print(path)
        img = image.load_img(path, target_size=(224, 224))
        return render_template('result.html',DATA = ProcessInput(img))


    return render_template('index.html')


if __name__ == '__main__':
    app.run()



if __name__ == "__main__":
    app.run()
