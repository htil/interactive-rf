import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# in linux --> export TF_CPP_MIN_LOG_LEVEL="3"
# in windows --> setx TF_CPP_MIN_LOG_LEVEL "3"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

DROP_Z = False

NUM_FRAMES1 = 15  # 15 for feature_data.npy
SEGMENTS = 3

num_face = 468
num_lhand = 21
num_pose = 33
num_rhand = 21

LEFT_HAND_OFFSET = num_face
POSE_OFFSET = LEFT_HAND_OFFSET + num_lhand
RIGHT_HAND_OFFSET = POSE_OFFSET + num_pose

# average over the entire face, and the entire 'pose'
averaging_sets = [[0, 468], [POSE_OFFSET, 33]]

lip_landmarks = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + num_lhand))
all_pose_landmarks = list(range(POSE_OFFSET, POSE_OFFSET + num_pose))
pose_relavant_landmarks = [11, 13, 15, 17, 19, 21,
                           12, 14, 16, 18, 20, 22]
pose_landmarks = [all_pose_landmarks[p] for p in pose_relavant_landmarks]
right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + num_rhand))

# point_landmarks = [item for sublist in [lip_landmarks, left_hand_landmarks, pose_landmarks, right_hand_landmarks]
#                    for item in sublist]
point_landmarks = [item for sublist in [lip_landmarks, left_hand_landmarks, right_hand_landmarks] for item in sublist]
LANDMARKS1 = len(point_landmarks) + len(averaging_sets)

if DROP_Z:
    INPUT_SHAPE1 = (NUM_FRAMES1, LANDMARKS1 * 2)
else:
    INPUT_SHAPE1 = (NUM_FRAMES1, LANDMARKS1 * 3)

FLAT_INPUT_SHAPE1 = (INPUT_SHAPE1[0] + 2 * (SEGMENTS + 1)) * INPUT_SHAPE1[1]


def tf_nan_mean(x, axis=0):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) /\
           tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)


def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))


def flatten_means_and_stds1(x, axis=0):
    # Get means and stds
    x_mean = tf_nan_mean(x, axis=0)
    x_std  = tf_nan_std(x,  axis=0)

    x_out = tf.concat([x_mean, x_std], axis=0)
    x_out = tf.reshape(x_out, (1, INPUT_SHAPE1[1]*2))
    x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
    return x_out


class FeatureGen(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureGen, self).__init__()

    def call(self, x_in):
        if DROP_Z:
            x_in = x_in[:, :, 0:2]
        x_list = [tf.expand_dims(tf_nan_mean(x_in[:, av_set[0]:av_set[0] + av_set[1], :], axis=1), axis=1) for av_set in
                  averaging_sets]
        x_list.append(tf.gather(x_in, point_landmarks, axis=1))
        x = tf.concat(x_list, 1)

        x_padded = x
        for i in range(SEGMENTS):
            p0 = tf.where(((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) != 0), 1, 0)
            p1 = tf.where(((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) == 0), 1, 0)
            paddings = [[p0, p1], [0, 0], [0, 0]]
            x_padded = tf.pad(x_padded, paddings, mode="SYMMETRIC")
        x_list = tf.split(x_padded, SEGMENTS)
        x_list = [flatten_means_and_stds1(_x, axis=0) for _x in x_list]

        x_list.append(flatten_means_and_stds1(x, axis=0))

        # Resize only dimension 0. Resize can't handle nan, so replace nan with that dimension's
        # avg value to reduce impact.
        x = tf.image.resize(tf.where(tf.math.is_finite(x), x, tf_nan_mean(x, axis=0)), [NUM_FRAMES1, LANDMARKS1])
        x = tf.reshape(x, (1, INPUT_SHAPE1[0] * INPUT_SHAPE1[1]))
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x_list.append(x)
        x = tf.concat(x_list, axis=1)
        return x


class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model
    """

    def __init__(self, feature_model, islr_model):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = feature_model
        self.islr_model = islr_model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """

        #         inputs = self.input_layer()(tf.cast(inputs, dtype=tf.float32))
        x = self.prep_inputs(inputs)
        #         print(x.shape)
        outputs = self.islr_model(x)

        #         outputs  = tf.concat([_model(x) for _model in self.islr_fold_models], axis=0)

        #         # Compute the weighted sum and the sum of the weights and compute the weighted mean
        #         outputs = tf.reduce_sum(tf.multiply(outputs, self.model_weights), axis=0)
        #         outputs = tf.divide(outputs, tf.reduce_sum(self.model_weights, axis=0))

        # Return a dictionary with the output tensor
        return {'outputs': outputs}


def get_inference_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path, compile=False)
    inference_model = TFLiteModel(FeatureGen(), loaded_model)
    return inference_model
