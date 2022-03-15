from flask import Flask, render_template, request
import tensorflow as tf
import keras
from gevent.pywsgi import WSGIServer
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import efficientnet.keras as efn
from datetime import datetime

model = keras.models.load_model(
    r"C:\Users\tobis\Downloads\WPy64-3920\notebooks\efnB0_V1.h5", compile=False)

# dimensions of our images
img_width, img_height = 224, 224

def predictLicensePlate(licenseplate):
    img = imread(licenseplate)
    img = resize(img,(224,224))
    img = img*1./255
    img = np.reshape(img,[1,224,224,3])

    classes = model.predict(img)
    #YOUR_RESULTING_ARRAY = np.argmax(classes, axis=2).astype(int)
    characters = '0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
    return "".join(np.array(list(characters))[np.argmax(classes, axis=2).astype(int)].flat)
    
app = Flask(__name__)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")                                                                                                                                                                                                                                                        
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		licenseplate = "static/" + img.filename	
		img.save(licenseplate)

		p = predictLicensePlate(licenseplate)

	return render_template("index.html", prediction = p, licenseplate = licenseplate)

if __name__ == '__main__':
    app.run(debug=True)
