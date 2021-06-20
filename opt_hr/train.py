import os
import time
import argparse
import numpy as np
import tensorflow as tf
from loss import loss2
import cifar100_input_python as ip
from architecture import VGG
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict

train_data = unpickle('../train')
test_data = unpickle('../test')

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path

def train(base_lr, batch_sz, gpu_no):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_no

    root_path = os.path.dirname(os.path.realpath(__file__))

    log_path = create_dir(os.path.join(root_path, 'log'))

    acc_count = 0
    while acc_count < 100:
        if os.path.exists(os.path.join(log_path, 'log_test_%02d.txt' % acc_count)):
            acc_count += 1
        else:
            break
    assert acc_count < 100

    log_train_fname = 'log_train_%02d.txt' % acc_count
    log_test_fname = 'log_test_%02d.txt' % acc_count

    n_class = 100
    batch_sz = batch_sz
    batch_test = 100
    max_epoch = 75000
    lr = base_lr
    momentum = 0.9
    is_training = tf.placeholder("bool")
    lr_ = tf.placeholder("float")

    images = tf.placeholder(tf.float32, (None, 32, 32, 3))
    labels = tf.placeholder(tf.int32, (None))

    vgg = VGG()
    vgg.build(images, n_class, is_training)

    fit_loss = loss2(vgg.score, labels, n_class, 'c_entropy')
    loss_op = fit_loss
    reg_loss_list = tf.losses.get_regularization_losses()
    if len(reg_loss_list) != 0:
        reg_loss = tf.add_n(reg_loss_list)
        loss_op += reg_loss

    all_vars = tf.trainable_variables()
    train_vars = [var for var in all_vars if 'fix' not in var.name]
    update_op = tf.train.MomentumOptimizer(lr_, 0.9).minimize(loss_op, var_list=train_vars)

    predc = vgg.pred
    acc_op = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.to_int32(vgg.pred))))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print ("====================")
        print ("Log will be saved to: " + log_path)

        with open(os.path.join(log_path, log_train_fname), 'w'):
            pass
        with open(os.path.join(log_path, log_test_fname), 'w'):
            pass

        for i in range(max_epoch):
            t = i % 390
            if t == 0:
                idx = np.arange(0, 50000)
                np.random.shuffle(idx)
                train_data['data'] = train_data['data'][idx]
                train_data['fine_labels'] = np.reshape(train_data['fine_labels'], [50000])
                train_data['fine_labels'] = train_data['fine_labels'][idx]
            tr_images, tr_labels = ip.load_train(train_data, batch_sz, t)

            if i == 30000:
                lr *= 0.1
            elif i == 50000:
                lr *= 0.1
            elif i == 65000:
                lr *= 0.1          

            fit, reg, acc, _ = sess.run([fit_loss, reg_loss, acc_op, update_op],
                                                {lr_: lr, is_training: True, images: tr_images, labels: tr_labels})

            if i % 100 == 0 and i != 0:                 
                print('====iter_%d: fit=%.4f, reg=%.4f, acc=%.4f'
                    % (i, fit, reg, acc))
                with open(os.path.join(log_path, log_train_fname), 'a') as train_acc_file:
                    train_acc_file.write('====iter_%d: fit=%.4f, reg=%.4f, acc=%.4f\n'
                    % (i, fit, reg, acc))

            if i % 500 == 0 and i != 0:
                n_test = 10000
                acc = 0.0
                for j in range(int(n_test/batch_test)):
                    te_images, te_labels = ip.load_test(test_data, batch_test, j)
                    acc = acc + sess.run(acc_op, {is_training: False, images: te_images, labels: te_labels})
                acc = acc * batch_test / float(n_test)
                print('++++iter_%d: test acc=%.4f' % (i, acc))
                with open(os.path.join(log_path, log_test_fname), 'a') as test_acc_file:
                    test_acc_file.write('++++iter_%d: test acc=%.4f\n' % (i, acc))

        n_test = 10000
        acc = 0.0
        for j in range(int(n_test/batch_test)):
            te_images, te_labels = ip.load_test(test_data, batch_test, j)
            acc = acc + sess.run(acc_op, {is_training: False, images: te_images, labels: te_labels})
        acc = acc * batch_test / float(n_test)
        print('++++iter_%d: test acc=%.4f' % (i, acc))
        with open(os.path.join(log_path, log_test_fname), 'a') as test_acc_file:
            test_acc_file.write('++++iter_%d: test acc=%.4f\n' % (i, acc))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_lr', type=float, default=1e-1,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
    parser.add_argument('--gpu_no', type=str, default='1',
                    help='gpu no')

    args = parser.parse_args()

    train(args.base_lr, args.batch_size, args.gpu_no)



