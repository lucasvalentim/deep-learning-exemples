import io
import numpy as np
import flask

from decouple import config
from PIL import Image
from exemples.mnist import mnist


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
            image = mnist.prepare_image(image)

            model = mnist.load_production_model()
            preds = model.predict(image)
            
            data['predictions'] = mnist.decode_predictions(preds[0])
            data['success'] = True

    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(debug=DEBUG)
