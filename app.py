from flask import Flask, url_for, send_from_directory, request, jsonify
from flask_cors import CORS
import logging, os
from werkzeug import secure_filename
from inference import predict
import argparse
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)
CORS(app)


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def getImage(path):
    with open(path, 'rb') as img_file:
        img = img_file.read()

    return img

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def predict(image):
    frozen_model_filename = "frozen_model.pb"
    graph = load_graph(frozen_model_filename)

    x = graph.get_tensor_by_name('prefix/input_image_as_bytes:0')
    y = graph.get_tensor_by_name('prefix/prediction:0')
    allProbs = graph.get_tensor_by_name('prefix/probability:0')

    img = image#getImage(image)
    print('hello type{}'.format(type(img)))

    with tf.Session(graph=graph) as sess:
        (y_out, probs_output) = sess.run([y,allProbs], feed_dict={
            x: [img]
        })
        # print(y_out)
        # print(allProbsToScore(probs_output))

        return {
            "predictions": [{
                "ocr": str(y_out),
                "confidence": probs_output
            }]
        }

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route('/upload/', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['image']:
        img = request.files['image']
        

        in_memory_file = io.BytesIO()
        img.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8).tobytes()
        prediction = predict(data)
        print(type(data))
        # print(prediction)
        # # remove the file so it will prevent from storage overload
        # os.remove(saved_path)
        return jsonify({'text':prediction})
    else:
        return jsonify({'text':"This is not an image"})

if __name__ == '__main__':
    app.run(debug=True)
