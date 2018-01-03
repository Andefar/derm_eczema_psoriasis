"""
=============================
NOTE:
Some functionality in this
version is deprecated. See
conv_net.py for correct use.
=============================
"""


import time
import tensorflow as tf
import matplotlib.pyplot as plt
from data_handler import *
from spatial_transformer import transformer
from tensorflow.contrib.layers import dropout, fully_connected, convolution2d, flatten, max_pool2d

pool = max_pool2d
conv = convolution2d
dense = fully_connected
from tensorflow.python.ops.nn import relu
import os

os.environ["CUDA VISIBLE DEVICES"] = "7"

# ensuring reproducibility
np.random.seed(42)

# global params
MODE = 'train'
RESTORE = True

# dirs
save_dir = 'saved_models_stn_med_1/'
vis_path = 'views_med_1/'

# network params
H, W, C = 672, 672, 3
img_shape = (H, W, C)
img_shape_flat = H * W * C
num_classes = 2

# training params
display_step = 1
batch_size = 32  # max is 32
num_epochs = 75
best_test_accuracy = 0.0
last_improvement = 0
require_improvement = 20

# load and index data
dh = DataHandler((H, W, C), patient_based=False)

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out


def build_model(x_pl, input_width, input_height, output_dim):
    # make distributed representation of input image for localization network
    loc_l1 = pool(x_pl, kernel_size=[2, 2], scope="localization_l1")
    loc_l2 = conv(loc_l1, num_outputs=16, kernel_size=[3, 3], stride=[1, 1], padding="SAME", scope="localization_l2")
    loc_l3 = pool(loc_l2, kernel_size=[2, 2], scope="localization_l3")
    loc_l4 = conv(loc_l3, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding="SAME", scope="localization_l4")
    loc_l4_flatten = flatten(loc_l4, scope="localization_l4-flatten")

    loc_l5 = dense(loc_l4_flatten, num_outputs=100, activation_fn=relu, scope="localization_l5")

    # set up weights for transformation (notice we always need 6 output neurons)
    with tf.name_scope("localization"):
        W_loc_out = tf.get_variable("localization_loc-out", [100, 6], initializer=tf.constant_initializer(0.0))
        initial = np.array([[0.35, 0, 0], [0, 0.35, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_loc_out = tf.Variable(initial_value=initial, name='b-loc-out')
        loc_out = tf.matmul(loc_l5, W_loc_out) + b_loc_out

    # spatial transformer
    l_trans1 = transformer(x_pl, loc_out, out_size=(H // 2, W // 2))
    l_trans1.set_shape([None, H // 2, W // 2, C])

    print("Transformer network output shape: ", l_trans1.get_shape())

    # classification network
    class_l1 = conv(l_trans1, num_outputs=64, kernel_size=[3, 3], scope="classification_l1")
    class_l2 = pool(class_l1, kernel_size=[2, 2], scope="classification_l2")
    class_l3 = conv(class_l2, num_outputs=64, kernel_size=[3, 3], scope="classification_l3")
    class_l4 = pool(class_l3, kernel_size=[2, 2], scope="classification_l4")
    class_l5 = conv(class_l4, num_outputs=128, kernel_size=[3, 3], scope="classification_l5")
    class_l6 = pool(class_l5, kernel_size=[2, 2], scope="classification_l6")
    class_l7 = conv(class_l6, num_outputs=128, kernel_size=[3, 3], scope="classification_l7")
    class_l8 = pool(class_l7, kernel_size=[2, 2], scope="classification_l8")
    class_l9 = conv(class_l8, num_outputs=256, kernel_size=[3, 3], scope="classification_l9")
    class_l10 = pool(class_l9, kernel_size=[2, 2], scope="classification_l10")
    class_l11 = conv(class_l10, num_outputs=256, kernel_size=[3, 3], scope="classification_l11")
    class_l12 = pool(class_l11, kernel_size=[2, 2], scope="classification_l12")
    class_l13 = conv(class_l12, num_outputs=512, kernel_size=[3, 3], scope="classification_l13")
    class_l14 = pool(class_l13, kernel_size=[2, 2], scope="classification_l14")
    class_l13 = conv(class_l12, num_outputs=512, kernel_size=[3, 3], scope="classification_l15")
    class_l14 = pool(class_l13, kernel_size=[2, 2], scope="classification_l16")

    dense_flatten = flatten(class_l14)
    dense_1 = dense(dense_flatten, num_outputs=2048, activation_fn=relu)
    dense_2 = dense(dense_1, num_outputs=2048, activation_fn=relu)
    dense_3 = dense(dense_2, num_outputs=512, activation_fn=relu)
    l_out = dense(dense_3, num_outputs=num_classes, activation_fn=tf.nn.softmax)

    return l_out, loc_out, l_trans1


def batch_masks(X_arg, b_size, random=False):
    num_train = len(X_arg)
    parts = []
    batch_mask = np.arange(num_train)
    if random: batch_mask = np.random.choice(num_train, num_train, replace=False)
    num_its = int(np.ceil(num_train / b_size))
    for i in range(num_its):
        if i != num_its - 1:
            parts.append(batch_mask[b_size * i:b_size * (i + 1)])
        else:
            parts.append(batch_mask[b_size * i:])
    return np.array(parts)


# define placeholder variables
X = tf.placeholder(tf.float32, [None, H, W, C], name='X')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')
lr_pl = tf.placeholder(tf.float32, shape=[], name="learning-rate")


def train_epoch(X_arg, y_arg, learning_rate, fetch, sess):
    num_batches = int(np.ceil(X_arg.shape[0] / float(batch_size)))
    costs = []
    correct = 0
    masks = batch_masks(X_arg, batch_size, True)
    for i in range(num_batches):
        X_batch_tr = X_arg[masks[i]]
        y_batch_tr = y_arg[masks[i]]
        # print(y_batch_tr)
        fetches_tr = fetch  # [train_op, cross_entropy, model, trans_params]

        # print("X_batch_tr: ",X_batch_tr.shape)
        # print("y_batch_tr: ",y_batch_tr.shape)

        feed_dict_tr = {X: X_batch_tr, y: onehot(y_batch_tr, num_classes), lr_pl: learning_rate}
        res = sess.run(fetches=fetches_tr, feed_dict=feed_dict_tr)
        cost_batch, output_train = tuple(res[1:3])
        # print("cost_batch: ",cost_batch)
        # print("output_train: ", output_train)
        # print("loc net work out\n: ", res[-1])
        costs += [cost_batch]
        # print("output_train \t",output_train)
        preds = np.argmax(output_train, axis=-1)
        # print("preds \t",preds)
        # print("labels\t",y_batch_tr)
        # print(np.equal(preds,y_batch_tr))
        # print(np.sum(np.equal(preds, y_batch_tr)))

        correct += np.sum(np.equal(preds, y_batch_tr))
    return np.mean(costs), correct / float(X_arg.shape[0])


def eval_epoch(X_arg, y_arg, learning_rate, fetch, sess, train):
    num_samples = X_arg.shape[0]
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    # if not train: print("num_samples: ", num_samples)
    # if not train: print("batch_size: ", batch_size)
    # if not train: print("num_batches: ",num_batches)
    correct = 0
    transform_list = []
    masks = batch_masks(X_arg, batch_size, train)
    # if not train: print("masks: \t",masks)
    for i in range(num_batches):
        # print("masks: ",masks[i])
        # print("y_arg: ",y_arg)
        X_batch_val = X_arg[masks[i]]
        y_batch_val = y_arg[masks[i]]
        # if not train: print("realy: \t",y_batch_val)

        fetches_val = fetch
        # print("fetch:",fetch)
        # print("train:",train)
        # val: [train_op, model, transformed]
        # test: [model, transformed]
        feed_dict_val = {X: X_batch_val, y: onehot(y_batch_val, num_classes), lr_pl: learning_rate}
        res = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)
        output_eval, transform_eval = tuple(res)[-2:]

        preds = np.argmax(output_eval, axis=-1)

        # if not train: print("preds: \t",np.argmax(output_eval, axis=-1))
        # if not train: print("labels:\t",y_batch_val)
        # if not train: print(np.sum(np.equal(preds, y_batch_val)))

        correct += np.sum(np.equal(preds, y_batch_val))
        transform_list.append(transform_eval)
    transform_eval = np.concatenate(transform_list, axis=0)
    return correct / float(X_arg.shape[0]), transform_eval[-9:]


def main():
    global best_test_accuracy
    learning_rate = 1e-5

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    # l_out, loc_out, l_trans1
    model, trans_params, transformed = build_model(X, H, W, C)

    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y * tf.log(model + 1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)

    # applying the gradients
    train_op = optimizer.minimize(cross_entropy)

    X_train, y_train, X_valid, y_valid, X_test, y_test = dh.load_data()

    print("Train: {}".format(X_train.shape))
    print("Valid: {}".format(X_valid.shape))
    print("Test: {}".format(X_test.shape))

    train_accs, valid_accs, test_accs = [], [], []
    print('Number of epochs: %s' % num_epochs)

    # for visualization purposes
    fig = plt.figure()

    # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
    # initialize the Session
    sess = tf.Session()

    # define saver object for storing and retrieving checkpoints
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,'models')  # path for the checkpoint file

    if RESTORE:
        # restore checkpoint if it exists
        try:
            print("Trying to restore last checkpoint...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
            saver.restore(sess, save_path=last_chk_path)
            print("Restored checkpoint from:", last_chk_path)
        except:
            print("Failed to restore checkpoint...")

    if MODE == 'train':
        # print("Initiating global variables...")
        sess.run(tf.global_variables_initializer())

        try:
            for n in range(num_epochs):
                # print("Epoch %d:" % n, end=', ')
                # print(X_train.shape,y_train.shape)
                # print(X_valid.shape, y_valid.shape)
                # print(X_test.shape, y_test.shape)

                train_cost, train_acc = train_epoch(X_train, y_train, learning_rate,
                                                    [train_op, cross_entropy, model, trans_params], sess)
                valid_acc, valid_transform = eval_epoch(X_valid, y_valid, learning_rate, [train_op, model, transformed],
                                                        sess, True)
                test_acc, test_transform = eval_epoch(X_test, y_test, learning_rate, [model, transformed], sess, False)
                valid_accs += [valid_acc]
                test_accs += [test_acc]
                train_accs += [train_acc]

                # learning rate annealing
                if (n + 1) % 20 == 0:
                    learning_rate = learning_rate * 0.7
                    print("New LR:", learning_rate)

                # check to see if there's an improvement
                improved_str = ''
                if test_acc > best_test_accuracy:
                    best_test_accuracy = test_acc
                    improved_str = '*'
                    saver.save(sess=sess, save_path=save_path + "_best_test_" + str(best_test_accuracy)+'_epoch_'+str(n))

                print("Epoch {}/{}, train cost {:.2}, train acc {:.2}, val acc {:.2}, test acc {:.2} - {}".format(
                    n, num_epochs, train_cost, train_acc, valid_acc, test_acc, improved_str))

                if test_transform.shape[0] >= 9 and np.max(test_transform) < 1.0 and np.min(test_transform) > 0.0:
                    thetas = test_transform[0:9].squeeze()
                    plt.clf()
                    for j in range(9):
                        plt.subplot(3, 3, j + 1)
                        plt.imshow(thetas[j])
                        plt.axis('off')
                    fig.canvas.draw()
                    plt.savefig(vis_path + 'transformed_epoch_' + str(n) + '.png', bbox_inches='tight', global_step=n)
                #else:
                #    print("Transformed not valid")

            plt.figure(figsize=(9, 9))
            plt.plot(1 - np.array(train_accs), label='Training Error')
            plt.plot(1 - np.array(valid_accs), label='Validation Error')
            plt.plot(1 - np.array(test_accs), label='Test Error')
            plt.legend(fontsize=20)
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Error', fontsize=20)
            plt.savefig(vis_path + 'Errors_plot.png', bbox_inches='tight')

        except KeyboardInterrupt:
            pass

    test_accuracy = eval_epoch(X_test, y_test, learning_rate, [model, transformed], sess, False)
    print("Test Set Accuracy: {}".format(test_accuracy[0]))


if __name__ == '__main__':
    main()
