import tensorflow as tf
import numpy as np

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
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('fix', shape, initializer=init,regularizer=regu)
        else:
            filt = tf.get_variable('fix', shape, initializer=init)

        return filt   

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def get_rotation(self, shape):
        init = tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        regu = tf.contrib.layers.l2_regularizer(self.wd)
        rotate = tf.get_variable('rotation', shape, initializer=init,regularizer=regu)
        
        return rotate    

    def householder(self, R):
        r,c = R.get_shape().as_list()
        Q = tf.eye(r)
        for cnt in range(r - 1):
            x = R[cnt:, cnt]
            e = [tf.norm(x)]
            pad = tf.zeros([r-cnt-1, ])
            e = tf.concat([e,pad],0)
            u = x-e
            v = u/tf.norm(u)
            paddings = tf.constant([[cnt, 0], [cnt, 0]])
            v2 = tf.pad(tf.tensordot(v,v,axes=0), paddings, "CONSTANT")
            Q_cnt = tf.eye(r)
            Q_cnt -=2.0*v2
            R = tf.matmul(Q_cnt,R)
            Q = tf.matmul(Q,tf.transpose(Q_cnt))
        return Q

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
            orth_rotate = self.householder(rotate)
            filt = tf.reshape(tf.matmul(orth_rotate,fix_filt),shape)
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

