import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder


def Upsampling(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])


    net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net


def build_tbnet(inputs, num_classes, frontend="ResNet101",
                    is_training=True, pretrained_dir="models"):

    # The spatial stream
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)

    # The context stream
    logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(inputs, frontend,
                                                                                  pretrained_dir=pretrained_dir,
                                                                                  is_training=is_training)


    gamma1 = tf.get_variable(name='gamma1', shape=[1], initializer=tf.zeros_initializer())
    gamma2 = tf.get_variable(name='gamma2', shape=[1], initializer=tf.zeros_initializer())

    feature1 = end_points['pool4']
    feature1 = slim.conv2d(feature1, 512, kernel_size=[1, 1])
    feature1 = slim.batch_norm(feature1, fused=True)

    # The context-aware attention
    [_, h, w, filters] = feature1.shape.as_list()
    b_ = slim.conv2d(feature1, filters / 8, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    c_ = slim.conv2d(feature1, filters / 8, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    d_ = slim.conv2d(feature1, filters, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    vec_b = tf.reshape(b_, [1, -1, tf.shape(feature1)[3] / 8])
    vec_cT = tf.transpose(tf.reshape(c_, [1, -1, tf.shape(feature1)[3] / 8]), (0, 2, 1))
    bcT = tf.matmul(vec_b, vec_cT)
    sigmoid_bcT = tf.nn.sigmoid(bcT)
    vec_d = tf.reshape(d_, [1, -1, tf.shape(feature1)[3]])
    bcTd = tf.matmul(sigmoid_bcT, vec_d)
    bcTd = tf.reshape(bcTd, [1, tf.shape(feature1)[1], tf.shape(feature1)[2], tf.shape(feature1)[3]])
    net_4 = gamma1 * bcTd + feature1


    feature2 = end_points['pool5']
    feature2 = slim.conv2d(feature2, 512, kernel_size=[1, 1])
    feature2 = slim.batch_norm(feature2, fused=True)

    [_, h, w, filters] = feature2.shape.as_list()
    b_ = slim.conv2d(feature2, filters / 8, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    c_ = slim.conv2d(feature2, filters / 8, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    d_ = slim.conv2d(feature2, filters, kernel_size=[1, 1], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    vec_b = tf.reshape(b_, [1, -1, tf.shape(feature2)[3] / 8])
    vec_cT = tf.transpose(tf.reshape(c_, [1, -1, tf.shape(feature2)[3] / 8]), (0, 2, 1))
    bcT = tf.matmul(vec_b, vec_cT)
    sigmoid_bcT = tf.nn.sigmoid(bcT)
    vec_d = tf.reshape(d_, [1, -1, tf.shape(feature2)[3]])
    bcTd = tf.matmul(sigmoid_bcT, vec_d)
    bcTd = tf.reshape(bcTd, [1, tf.shape(feature2)[1], tf.shape(feature2)[2], tf.shape(feature2)[3]])
    net_5 = gamma2 * bcTd + feature2

    global_channels = tf.reduce_mean(net_5, [1, 2], keep_dims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)

    # The boundary stream
    # The global-gated convolution
    conv1 = slim.conv2d(net_4, 512, kernel_size=[1, 1])
    
    res = slim.conv2d(conv1, 512, kernel_size=[3, 3], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    res = tf.nn.relu(slim.batch_norm(res, fused=True))
    res = slim.conv2d(res, 512, kernel_size=[3, 3], stride=[1, 1], activation_fn=None, normalizer_fn=None)
    res = slim.batch_norm(res, fused=True)
    res = conv1 + res
    res = tf.nn.relu(res)
    
    net_5_scaled = Upsampling(net_5_scaled, scale=2)
    conv2 = slim.conv2d(net_5_scaled, 512, kernel_size=[1, 1])

    ggc = tf.concat([res, conv2], axis=-1)
    ggc = slim.batch_norm(ggc, fused=True)
    ggc = slim.conv2d(ggc, 512, kernel_size=[1, 1])
    ggc = tf.nn.relu(ggc)
    ggc = slim.conv2d(ggc, 512, kernel_size=[1,1])
    ggc = slim.batch_norm(ggc, fused=True)
    ggc = tf.nn.sigmoid(ggc)
    gated = res * (1+ggc)
    gated = Upsampling(gated, scale=2)

    output = slim.conv2d(gated, 512, kernel_size=[1,1])

    output_edge = slim.conv2d_transpose(gated, 128, kernel_size=[3, 3], stride=[4, 4], activation_fn=None)
    output_edge = tf.nn.relu(slim.batch_norm(output_edge))
    output_edge = slim.conv2d_transpose(output_edge, 1, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    output_edge = tf.nn.relu(slim.batch_norm(output_edge))
    output_edge = tf.nn.sigmoid(output_edge)
    output_edge = tf.reshape(output_edge, [tf.shape(output_edge)[1], tf.shape(output_edge)[2]])

    # The feature fusion
    net_5_scaled = Upsampling(net_5_scaled, 2)
    output_s_c = tf.concat([spatial_net,net_5_scaled], axis=-1)
    output_s_c = slim.batch_norm(output_s_c)
    output_s_c = slim.conv2d(output_s_c, 256, kernel_size=[1,1])

    net = FeatureFusionModule(input_1=output_s_c, input_2=output, n_filters=num_classes)
    net = Upsampling(net, scale=8)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn, output_edge



