import io
import numpy as np
import flask
import keras

from decouple import config
from PIL import Image


DEBUG = config('DEBUG', default=False)

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/api/mnist/predict', methods=['POST'])
def mnist_predict():
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))
            
            if image.mode != 'L':
                image = image.convert('L')

            image = image.resize((28, 28))
            image = keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            with open('model.json', 'r') as f:
                model_json = f.read()

            model = keras.models.model_from_json(model_json)

            model.load_weights('weights_mnist.hdf5')
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            data['metrics'] = model.predict(image).tolist()

    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(debug=DEBUG)
