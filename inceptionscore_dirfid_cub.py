import numpy as np
import tensorflow.compat.v1 as tf
from scipy import linalg
import os
import pickle
from inception.slim import slim
from inception.slim import ops


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir',
                           '/scratch/abhinav597project/RAT-Diffusion/FID-with-Fine-tuned-Inception/StackGAN-inception-model/model.ckpt',
                           """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string('npz_file1',
                           '/scratch/abhinav597project/RAT-Diffusion/samples.npz',
                           """Path to the first .npz file (e.g., generated images).""")

tf.app.flags.DEFINE_string('npz_file2',
                           '/scratch/abhinav597project/samples.npz',
                           """Path to the second .npz file (e.g., real images).""")

tf.app.flags.DEFINE_integer('num_classes', 50, """Number of classes.""")
tf.app.flags.DEFINE_integer('splits', 10, """Number of splits.""")
tf.app.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.app.flags.DEFINE_integer('gpu', 0, "The ID of GPU to use")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(sigma2 + eps * np.eye(sigma2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def preprocess_npz(npz_file):
    data = np.load(npz_file)['arr_0']
    processed_images = []
    for img in data:
        img = tf.image.resize(img, [299, 299])  # Resize to (299, 299)
        img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
        processed_images.append(img)
    return tf.stack(processed_images)  # Return a stacked tensor


def inference(images, num_classes, for_training=False, restore_logits=True, scope=None):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
    }

    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d], stddev=0.1, activation=tf.nn.relu, batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
                images,
                dropout_keep_prob=0.8,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope)
    auxiliary_logits = endpoints['mixed_8x8x2048b']
    return logits, auxiliary_logits


def get_inception_features(sess, images, pred_op):
    bs = FLAGS.batch_size
    images = sess.run(images)  # Ensure images is evaluated to a NumPy array
    num_images = images.shape[0]  # Get the number of images
    preds = []
    for i in range(0, num_images, bs):
        batch = images[i:i + bs]  # Slice as a NumPy array
        pred = sess.run(pred_op, feed_dict={'inputs:0': batch})
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    mu = np.mean(preds, axis=0)
    sigma = np.cov(preds, rowvar=False)
    return mu, sigma


def main(unused_argv=None):
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device(f"/gpu:{FLAGS.gpu}"):
                num_classes = FLAGS.num_classes + 1

                inputs = tf.placeholder(tf.float32, [None, 299, 299, 3], name='inputs')
                logits, pred_op = inference(inputs, num_classes)

                pred_op = ops.avg_pool(pred_op, pred_op.shape[1:3], padding='VALID', scope='pool')
                pred_op = ops.flatten(pred_op, scope='flatten')

                variable_averages = tf.train.ExponentialMovingAverage(0.9999)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)

                saver.restore(sess, FLAGS.checkpoint_dir)
                print(f"Model restored from {FLAGS.checkpoint_dir}")

                # Preprocess images from npz files
                images1 = preprocess_npz(FLAGS.npz_file1)
                images2 = preprocess_npz(FLAGS.npz_file2)

                # Get Inception features
                mu1, sigma1 = get_inception_features(sess, images1, pred_op)
                mu2, sigma2 = get_inception_features(sess, images2, pred_op)

                # Calculate FID
                fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                print(f"FID: {fid_value}")


if __name__ == '__main__':
    tf.app.run()
