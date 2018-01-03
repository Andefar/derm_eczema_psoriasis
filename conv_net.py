import time
import tensorflow as tf
import matplotlib.pyplot as plt
from data_handler2 import *
import os
import vgg19

os.environ["CUDA VISIBLE DEVICES"] = "6"

np.random.seed(42)

MODE = 'test'
RESTORE = True
TEST_CLASS = 'eczema'
#TEST_CLASS = 'psoriasis'

data_dir = '/scratch/derm_bal'
save_dir = 'models_conv_1/'
vis_dir = 'plots_conv_1/'

H, W, C = 224, 224, 3
img_shape = (H, W, C)
img_shape_flat = H * W * C
num_classes = 2

# training params
batch_size = 64
num_epochs = 50
best_val_accuracy = 0.0

# load
dh = DataHandler2((H, W, C), data_dir=data_dir, mode = MODE, test_class = TEST_CLASS)

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def build_model(x_pl, input_width, input_height, output_dim):

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(x_pl)

    return vgg.prob


def batch_masks(X_arg, b_size, shuffle = False):
    num_train = len(X_arg)
    parts = []
    batch_mask = np.arange(num_train)

    if shuffle: np.random.shuffle(batch_mask)

    num_its = int(np.ceil(num_train / b_size))

    for i in range(num_its):

        if i != num_its - 1:
            parts.append(batch_mask[b_size * i:b_size * (i + 1)])

        else:
            parts.append(batch_mask[b_size * i:])

    return np.array(parts)



X = tf.placeholder(tf.float32, [None, H, W, C], name='X')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')
lr_pl = tf.placeholder(tf.float32, shape=[], name="learning-rate")


def train_epoch(X_arg, y_arg, learning_rate, fetch, sess):
    num_batches = int(np.ceil(float(len(X_arg)) / float(batch_size)))

    costs = []
    correct = 0

    masks = batch_masks(X_arg, batch_size, shuffle = True)

    for i in range(num_batches):

        X_batch_tr = X_arg[masks[i]]
        y_batch_tr = y_arg[masks[i]]

        fetches_tr = fetch  # [model, cross_entropy, train_op]
        feed_dict_tr = {X: X_batch_tr, y: onehot(y_batch_tr, num_classes), lr_pl: learning_rate}
        res = sess.run(fetches=fetches_tr, feed_dict=feed_dict_tr)
        output_train, cost_batch, _  = tuple(res)
        costs += [cost_batch]
        preds = np.argmax(output_train, axis=-1)

        #print("train-probs: \t", output_train)
        #print("train-preds: \t", preds)
        #print("train-labels:\t", y_batch_tr)
        #print("train-cost_batch:", cost_batch)
        #print(np.sum(np.equal(preds, y_batch_tr)))

        correct += np.sum(np.equal(preds, y_batch_tr))

    return np.mean(costs), correct / float(len(X_arg))


def eval_epoch(X_arg, y_arg, learning_rate, fetch, sess):
    num_batches = int(np.ceil(float(len(X_arg)) / float(batch_size)))

    costs = []
    correct = 0

    masks = batch_masks(X_arg, batch_size)

    for i in range(num_batches):

        X_batch_val = X_arg[masks[i]]
        y_batch_val = y_arg[masks[i]]

        fetches_val = fetch # [model, cross_entropy]

        feed_dict_val = {X: X_batch_val, y: onehot(y_batch_val, num_classes), lr_pl: learning_rate}
        output_eval, output_cost = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)
        costs += [output_cost]
        preds = np.argmax(output_eval, axis=-1)
        #print("val-probs: \t", output_eval)
        #print("val-preds: \t", preds)
        #print("val-labels:\t", y_batch_val)
        #print(np.sum(np.equal(preds, y_batch_val)))
        #print("val-cost_batch:", output_cost)
        correct += np.sum(np.equal(preds, y_batch_val))

    return np.mean(costs) , correct / float(len(X_arg))

def test_final_model(X_arg, y_arg, fetches, sess):
    num_samples = X_arg.shape[0]
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    correct = 0
    masks = batch_masks(X_arg, batch_size)
    preds_out = []
    for i in range(num_batches):
        X_batch_val = X_arg[masks[i]]
        y_batch_val = y_arg[masks[i]]
        fetches_val = fetches # model
        feed_dict_val = {X: X_batch_val}
        output_test = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)
        preds = np.argmax(output_test, axis=-1)
        preds_out.append(preds)
        correct += np.sum(np.equal(preds, y_batch_val))

    preds_out = [item for sublist in preds_out for item in sublist]
    return correct / float(X_arg.shape[0]), preds_out

def main():
    global best_val_accuracy
    learning_rate = 1e-5

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = build_model(X, H, W, C)

    cross_entropy = -tf.reduce_sum(y * tf.log(model + 1e-8), reduction_indices=[1])
    cross_entropy = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr_pl)

    train_op = optimizer.minimize(cross_entropy)


    sess = tf.Session()


    saver = tf.train.Saver()

    save_path = os.path.join(save_dir,'models')

    if RESTORE:

        try:

            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
            saver.restore(sess, save_path=last_chk_path)
            print("Restored checkpoint from: ", last_chk_path)
        except:
            raise("Failed to restore checkpoint")

    if MODE == 'train':

        sess.run(tf.global_variables_initializer())

        train_accs, valid_accs = [], []
        train_costs, valid_costs = [], []

        X_train, y_train, X_valid, y_valid = dh.load_data()

        try:

            for epoch_num in range(num_epochs):

                train_cost, train_acc = train_epoch(X_train, y_train, learning_rate, [model, cross_entropy, train_op], sess)

                valid_cost, valid_acc = eval_epoch(X_valid, y_valid, learning_rate, [model, cross_entropy], sess)

                train_accs += [train_acc]
                valid_accs += [valid_acc]

                train_costs += [train_cost]
                valid_costs += [valid_cost]

                if (epoch_num + 1) % 20 == 0:
                    learning_rate = learning_rate * 0.7
                    #print("New LR:", learning_rate)

                improved_str = ''
                if valid_acc > best_val_accuracy:
                    best_val_accuracy = valid_acc
                    improved_str = '*'
                    saver.save(sess=sess, save_path=save_path + "_" + str(best_val_accuracy)+'_epoch_'+str(epoch_num))

                print("Epoch {}/{}, train cost {:.4}, valid cost {:.4}, train acc {:.4}, valid acc {:.4} \t {:}".format(
                    epoch_num, num_epochs, train_cost, valid_cost, train_acc, valid_acc, improved_str))

            plt.figure(figsize=(9, 9))
            plt.plot(1 - np.array(train_accs), label='Training Error')
            plt.plot(1 - np.array(valid_accs), label='Validation Error')
            plt.legend(fontsize=20)
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Error', fontsize=20)
            plt.savefig(vis_dir + 'Accuracy_plot.png', bbox_inches='tight')

            plt.figure(figsize=(9, 9))
            plt.plot(np.array(train_costs), label='Training Cost')
            plt.plot(np.array(valid_costs), label='Validation Cost')
            plt.legend(fontsize=20)
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Cost', fontsize=20)
            plt.savefig(vis_dir + 'Cost_plot.png', bbox_inches='tight')

        except KeyboardInterrupt:
            pass

    elif MODE == 'test':
        X_test, y_test = dh.load_test_data()
        test_accuracy = test_final_model(X_test, y_test, model, sess)
        print(test_accuracy)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Used %.2f seconds" % (time.time() - start_time))
