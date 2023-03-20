from cam_utils.model_utils import get_inference_model
import numpy as np
import json
import os


def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic


def predict_cam(frames, model_path):
    # interpreter = tf.lite.Interpreter(model_path)
    # found_signatures = list(interpreter.get_signature_list().keys())
    # prediction_fn = interpreter.get_signature_runner("serving_default")
    # output = prediction_fn(inputs=frames)
    # sign = np.argmax(output["outputs"])
    model = get_inference_model(model_path)
    # print('frames.shape', frames.shape)
    pred = model(frames.astype(np.float32))['outputs']
    # print('pred.shape', pred.shape)
    sign = np.argmax(pred, -1)
    # print('sign.shape', sign.shape)
    label_index = read_dict(r"cam_utils\chess_classes_0_to_63.json")
    index_label = dict([(label_index[key], key) for key in label_index])
    print(index_label)
    return index_label[sign[0]]
