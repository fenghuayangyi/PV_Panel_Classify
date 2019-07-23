# -*- coding: utf-8 -*-
from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

# 将图片resize成100*100
w = 100
h = 100
c = 3


# 读取图片
def read_img(path):
    """
    读取图片并进行特征抽取
    :param path:
    :return:
    """
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def read_img_valid(path):
    """
    读取单独验证图片
    :param path: 图片所在目录
    :return:
    """
    imgs = []
    imgsdir = []
    i = 0
    for im in glob.glob(path + '/*.jpg'):

        imgsdir.append(im)
        img = io.imread(im)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        i = i+1
        print('读取图片%d :%s' % (i, im))
    return np.asarray(imgs, np.float32), imgsdir


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """
    按批次取数据
    :param inputs:
    :param targets:
    :param batch_size:
    :param shuffle:
    :return:
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def model():
    """
    自定义CNN模型
    :return:
    """
    # 1.准备数据占位符 x [None, 100*100]  y_true [None, 2]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
        y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    # 第一个卷积层（100——>50)
    with tf.variable_scope("conv1"):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # 第二个卷积层(50->25)
    with tf.variable_scope("conv2"):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # 第三个卷积层(25->12)
    with tf.variable_scope("conv3"):
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # 第四个卷积层(12->6)
    with tf.variable_scope("conv4"):
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全连接层1
    with tf.variable_scope("fc1"):
        dense1 = tf.layers.dense(inputs=re1,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # 全连接层2
    with tf.variable_scope("fc2"):
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=512,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        # 进行矩阵运算得出样本的结果
        logits = tf.layers.dense(inputs=dense2,
                                 units=5,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return x, y_, logits


def conv_fc():
    """
    训练模型，并将模型保存
    :return:
    """
    path = 'E:/ProgrammeCode/IDEA/PV_Panel_Classify/0510_resize/'
    data, label = read_img(path)
    # 打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    # 将所有数据分为训练集和验证集
    ratio = 0.7
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    # 定义模型，得到输出
    x, y_true, y_predict = model()
    # 求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_predict)
    # 梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 计算准确率
    with tf.variable_scope("acc"):
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_predict, 1), tf.int32), y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 收集变量 单个数字值收集
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()
    # 定义一个合并变量的 op
    merged = tf.summary.merge_all()

    # 创建一个saver 保存模型
    saver = tf.train.Saver()
    n_epoch = 500
    batch_size = 20
    # 开启回话运行
    with tf.Session() as sess:
        # 建立events文件，然后写入项目目录下的tmp文件
        filewriter = tf.summary.FileWriter("../PV_Panel_Classify/tmp/summary/train/",
                                           graph=sess.graph)
        if os.path.exists("../PV_Panel_Classify/tmp/params/cnn_model"):
            saver.restore(sess, "../PV_Panel_Classify/tmp/params/cnn_model")
        else:
            sess.run(init_op)
        for epoch in range(n_epoch):
            start_time = time.time()
            # training
            train_loss, train_acc, n_batch = 0, 0, 0
            for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
                _, err, ac = sess.run([train_op, loss, accuracy], feed_dict={x: x_train_a, y_true: y_train_a})
                train_loss += err;
                train_acc += ac;
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            # validation
            val_loss, val_acc, n_batch = 0, 0, 0
            for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
                err, ac = sess.run([loss, accuracy], feed_dict={x: x_val_a, y_true: y_val_a})
                val_loss += err;
                val_acc += ac;
                n_batch += 1
            print("   validation loss: %f" % (val_loss / n_batch))
            print("   validation acc: %f" % (val_acc / n_batch))
        # 保存参数 PATH是保存路径
        builder = tf.saved_model.builder.SavedModelBuilder("../PV_Panel_Classify/tmp/model/cnn_model")
        # 保存整张网络及其变量
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
        builder.save()  # 完成保存
        saver.save(sess, "../PV_Panel_Classify/tmp/params/cnn_model")
        sess.close()
    return None


def call_conv_fc():
    """
    调用训练好的模型对valid_photo目录下的图像进行分类,测试模型的准确性
    valid_photo/N/*.JPG
    valid_photo/P/*.JPG
    :return:
    """
    path = '../PV_Panel_Classify/valid_photo/'
    data, imgsdir = read_img_valid(path)
    print(imgsdir)
    # init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 加了这句话就会导致图初始化
        # sess.run(init_op)
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "../PV_Panel_Classify/tmp/model/cnn_model")
        # 以加载输入变量
        input_x = sess.graph.get_tensor_by_name('data/x:0')
        # print(input_x)
        input_y = sess.graph.get_tensor_by_name('data/y_:0')
        # print(input_y)
        output = sess.graph.get_tensor_by_name('fc2/dense_1/BiasAdd:0')
        result = sess.run(tf.argmax(output, 1), {input_x: data})
        for i in range(len(result)):
            if result[i] == 0:
                print("预测结果：有覆盖物。 dir:%s" % imgsdir[i])
            else:
                print("预测结果：无故障。 dir:%s" % imgsdir[i])


if __name__ == "__main__":
    # conv_fc()
    call_conv_fc()

