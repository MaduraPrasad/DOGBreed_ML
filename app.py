from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import os
import pandas as pd

app = Flask(__name__)

# Load the CNN model
cnn_model = load_model('C:/Users/User/Downloads/cnn_model.h5')


# Load the labels
# Assuming 'labels.csv' is in the same directory as the app
labels = pd.read_csv('labels.csv')
labels['new_id'] = labels['id'] + '.jpg'

# Define the classes
classes = list(labels.sort_values(by='breed')['breed'].unique())

# Function to process the uploaded image
def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Route for handling image upload and prediction
@app.route('/', methods=['GET', 'POST'])
def fun_home():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

# Function to process image and make prediction
def predict():
    imagefile = request.files['imagefile']
    image_filename = secure_filename(imagefile.filename)
    image_path = os.path.join("static/images", image_filename)
    imagefile.save(image_path)

    # Process the image and make a prediction
    img_array = process_image(image_path)
    prediction = cnn_model.predict(img_array)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    # Render the result page with the prediction
    return render_template('index.html', prediction=predicted_class, image_filename=image_filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
