from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import sys


#######################################################################################################################
# Fill in here all necessary information for the grad-cam analysis of your frozen graph model. This implementation
# assumes that the model inputs were scaled to [-1,1] for the model training. Other rescaling can be used by just
# changing the 'preprocess_input' , the 'unscale_input' function and the 'unpreprocess_input' function.


image_shape = [700, 700]
num_classes = 5
model_str = sys.argv[1]
print(model_str)
data_path = sys.argv[2]
print(data_path)

activation_tensor_gradcam = 'cf/StatefulPartitionedCall/vgg16/block5_conv3/Relu:0'  #insert here (last) convolutional layer
input_tensor = 'x:0'
output_tensor = "cf/StatefulPartitionedCall/dense_1/MatMul:0"

write_path = data_path
store_format = 'png'


#######################################################################################################################


def preprocess_input(img):
    """Adds 4-th dimension to image tensor and scales image values from [0,255] to [-1,1].
    Args:
        img (numpy.ndarray[float]): image to preprocess.

    Returns:
        numpy.ndarray[float]: preprocessed image tensor.
    """
    img = np.expand_dims(img, axis=0)
    img = ((img / 127.5) - 1)
    return img


def unscale_input(img):
    """Reverses scaling of image values from [-1,1] to [0,255].
    Args:
        img (numpy.ndarray[float]): image to scale

    Returns:
        numpy.ndarray[float]: unscaled image
    """
    return ((img + 1) * 127.5)


def unpreprocess_input(img):
    """Reverses prepocessing of images by squeezing first dimension of the image
    tensor and scaling of image values from [-1,1] to [0,255].
    Args:
        img (numpy.ndarray[float]): image to unpreprocess.

    Returns:
        numpy.ndarray[float]: unprepocessed image.
    """
    img = np.squeeze(img, axis=0)
    img = np.interp(img, (img.min(), img.max()), (0, 255))
    return img.astype('uint8')


def grad_cam(g,
             sess,
             image,
             category_index,
             nb_classes,
             target_size,
             conv_tensor_name,
             input_tensor_name='x:0',
             output_tensor_name='cf/StatefulPartitionedCall/dense_1/MatMul:0'):
    """Implementation of Grad-CAM (https://arxiv.org/abs/1610.02391) for frozen graph models based on tf1.
    In short this function produces a heatmap with values corresponding to weighted (the weights represent the averaged
    importance of a feature map of the defined convolutional layer wrt to the choosen class) feature map values of
    the defined convolutional layer.
    Args:
        g (tensorflow.Graph): graph of the model we are working with.
        sess (tensorflow.Session): tensorflow session we are working with.
        image (numpy.ndarray[float]): the image we want to apply gradcam on.
        category_index (numpy.ndarray[int]): the class index we want our heatmap for.
        nb_classes (int): number of classes.
        target_size (list of int): list with len=2 defining the size we want our heatmap in.
        conv_tensor_name (str): name of the tensor corresponding to the concolutional layer we want to use for gradcam
            (quote from gradcam paper: "We find that Grad-CAM maps become progressively worse as we move to earlier
            convolutional layers as they have smaller receptive fields and only focus on less semantic local features."
            --> use e.g. last concolutional layer).
        input_tensor_name (str): name of the input tensor of the model.
        output_tensor_name (str): name of the output tensor of the model.

    Returns:
        (numpy.ndarray[int], numpy.ndarray[float]): with
            np.uint8(cam): map for visualizing with e.g. matplotlib.
            heatmap: tensor with shape (1, target_size[0], target_size[1]) containing heat values from 0 (not important)
            to 1 (very important) for every pixel wrt to the defined class.
    """
    one_hot = tf.sparse_to_dense(category_index, [nb_classes], 1.0)
    signal = tf.multiply(g.get_tensor_by_name(output_tensor_name), one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, g.get_tensor_by_name(conv_tensor_name))[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run(
        [g.get_tensor_by_name(conv_tensor_name), norm_grads],
        feed_dict={g.get_tensor_by_name(input_tensor_name): image})
    output = output[0]
    grads_val = grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.array(PIL.Image.fromarray(cam).resize(target_size, PIL.Image.BILINEAR), dtype=np.float32)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cm = plt.get_cmap('coolwarm')
    cam = cm(np.uint8(255 * heatmap))
    cam = (cam[:, :, :3] * 255).astype(np.uint8)
    cam = np.float32(cam) + np.float32(unscale_input(image[0, :]))
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


list_input_imgs = [name for name in os.listdir(data_path) if
                   not os.path.isdir(data_path + '/' + name)]

with tf.gfile.FastGFile(model_str, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

g = tf.get_default_graph()
sess = tf.Session()


for counter, filename in enumerate(list_input_imgs):
    img = np.array(
        PIL.Image.open(tf.gfile.Open(data_path + '/' + filename, 'rb')).convert('RGB').resize(image_shape,
                                                                                              PIL.Image.BILINEAR),
        dtype=np.float32)
    preprocessed_input = preprocess_input(img)
    predictions = sess.run(g.get_tensor_by_name(output_tensor),
                           feed_dict={g.get_tensor_by_name(input_tensor): preprocessed_input})
    print(predictions)
    predicted_class = np.argmax(predictions)


    cam, heatmap = grad_cam(g=g, sess=sess, image=preprocessed_input, category_index=predicted_class,
                            conv_tensor_name=activation_tensor_gradcam,
                            input_tensor_name=input_tensor, output_tensor_name=output_tensor,
                            nb_classes=num_classes, target_size=image_shape)


    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(cam)
    axarr[1].imshow(img.astype(np.uint8))
    if not os.path.exists(write_path + '/grad-cam-images'):
        os.makedirs(write_path + '/grad-cam-images')
    plt.savefig(write_path + '/grad-cam-images/' + 'grad-cam-image_' + str(counter) + '.' + store_format)
    plt.clf()