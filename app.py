import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__, template_folder='.')  # Ensures HTML files are found in the same directory

# Load the trained model
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo==0:
	    return "No Brain Tumor"    
    elif classNo==1:
	    return "Yes Brain Tumor"

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


# Serve static files (CSS, JS, Images) from the same directory
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('.', filename)


# Main index route
@app.route('/')
def index():
    return render_template('index.html')


# Import page route
@app.route('/import')
def import_page():
    return render_template('import.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)