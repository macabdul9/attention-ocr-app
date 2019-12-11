import argparse
import tensorflow as tf
import numpy as np

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

def getImage(path):
    with open(path, 'rb') as img_file:
        img = img_file.read()
    return img

def predict(image):
    frozen_model_filename = "frozen_model.pb"
    graph = load_graph(frozen_model_filename)

    x = graph.get_tensor_by_name('prefix/input_image_as_bytes:0')
    y = graph.get_tensor_by_name('prefix/prediction:0')
    allProbs = graph.get_tensor_by_name('prefix/probability:0')

    img = getImage(image)

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
        };



