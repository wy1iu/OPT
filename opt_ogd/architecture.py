from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np

class OGD(optimizer.Optimizer):
    def __init__(self, learning_rate=0.1, use_locking=False, name="OGD"):
        super(OGD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "auxiliary", self._name)
            self._zeros_slot(v, "y", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)        
        mk = self.get_slot(var, "momentum")
        wk = self.get_slot(var, "auxiliary")
        y = self.get_slot(var, "y")

        mk = mk.assign(0.9*mk-grad)
        wk = wk.assign(tf.matmul(mk,tf.transpose(var))-0.5*tf.matmul(var,tf.matmul(tf.matmul(tf.transpose(var),mk),tf.transpose(var))))
        wk_t = state_ops.assign_sub(wk, tf.transpose(wk))
        mk_t = state_ops.assign(mk,tf.matmul(wk_t,var))
        a = math_ops.minimum(lr_t, 1.0/(tf.norm(wk_t)+1e-8))
        y = y.assign(var + a * mk_t)
        y_t = state_ops.assign(y,(var + (a/2.0) * tf.matmul(wk_t,(var + y))))
        var_update = state_ops.assign_add(var, (a/2.0) * tf.matmul(wk_t,(var + y_t))) 
        return control_flow_ops.group(*[var_update, mk_t, wk_t, y_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class VGG():
    def get_conv_filter(self, shape, reg, stddev):
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('filter', shape, initializer=init,regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt      

    def get_fix_filter(self, shape, reg, stddev):
        init = tf.random_normal_initializer(stddev=stddev)
        filt = tf.get_variable('fix', shape, initializer=init)

        return filt   

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def get_rotation(self, shape):
        init = tf.constant(np.eye(shape[0]), dtype=tf.float32)
        rotate = tf.get_variable('rotation', initializer=init)
        
        return rotate 

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, ksize, n_filt, is_training, name, stride=1, 
        pad='SAME', relu=False, reg=True,  bn=True):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape1 = [ksize*ksize*n_input, n_filt]
            shape2 = [ksize*ksize*n_input, ksize*ksize*n_input]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))

            fix_filt = self.get_fix_filter(shape1, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            rotate = self.get_rotation(shape2)
            filt = tf.reshape(tf.matmul(rotate,fix_filt),shape)
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
                
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def _conv_layer_normal(self, bottom, ksize, n_filt, is_training, name, stride=1, 
        pad='SAME', relu=False, reg=True,  bn=True):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))
            
            filt = self.get_conv_filter(shape, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
                
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def build(self, rgb, n_class, is_training):
        self.wd = 5e-4

        feat = (rgb - 127.5) / 128.0

        ksize = 3
        n_layer = 2

        # 32X32
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv1_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True)
        feat = self._max_pool(feat, 'pool1')

        # 16X16
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv2_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True)
        feat = self._max_pool(feat, 'pool2')

        # 8X8
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv3_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True)
        feat = self._max_pool(feat, 'pool3')

        self.fc6 = self._conv_layer(feat, 4, 64, is_training, "fc6", bn=False, relu=False, pad='VALID',
                                    reg=True)

        self.score = self._conv_layer_normal(self.fc6, 1, n_class, is_training, "score", bn=False, relu=False, pad='VALID',
                                      reg=True)

        self.pred = tf.squeeze(tf.argmax(self.score, axis=3))

