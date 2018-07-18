# CycleGAN
#
# Based on:
#  - https://github.com/AYLIEN/gan-intro/blob/master/gan.py
#  - https://hardikbansal.github.io/CycleGANBlog/
#
# Dataset:
#  - http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip

import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf

# Due to this being run on Kamiak, that doesn't have _tkinter, we have to set a
# different backend otherwise it'll error
# https://stackoverflow.com/a/40931739/2698494
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

#tf.enable_eager_execution()
#tf.executing_eagerly()

#seed=42
#np.random.seed(seed)
#tf.set_random_seed(seed)

#
# Import and preview the Apple and Windows emojis dataset.
#
def _parse_function(filename, channels=4, resize=None):
    """ Get tensor from filename """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels) # RGBA images

    if resize:
        image_decoded = tf.image.resize_images(image_decoded, resize)

    # Normalize to be between -1 and 1
    image_cast = tf.cast(image_decoded, tf.float32)
    image_norm = tf.subtract(tf.divide(image_cast, [127.5]), 1)

    return image_norm

def train_input_fn(batch_size=1, channels=4, resize=None):
    """ Get tensors of training data for image sets A and B """
    apple = tf.data.Dataset.list_files("emojis/Apple/*.png").map(lambda x: _parse_function(x,channels,resize))
    windows = tf.data.Dataset.list_files("emojis/Windows/*.png").map(lambda x: _parse_function(x,channels,resize))

    apple_iter = apple.shuffle(10000).batch(batch_size).make_initializable_iterator()
    windows_iter = windows.shuffle(10000).batch(batch_size).make_initializable_iterator()
    return {
        'A': apple_iter,
        'B': windows_iter
    }

def test_input_fn(batch_size=1, channels=4, resize=None, seed=42):
    """ Get tensors of testing data for image sets A and B """
    apple = tf.data.Dataset.list_files("emojis/Test_Apple/*.png").map(lambda x: _parse_function(x,channels,resize))
    windows = tf.data.Dataset.list_files("emojis/Test_Windows/*.png").map(lambda x: _parse_function(x,channels,resize))

    apple_iter = apple.shuffle(10000,seed=seed).batch(batch_size).make_initializable_iterator()
    windows_iter = windows.shuffle(10000,seed=seed).batch(batch_size).make_initializable_iterator()
    return {
        'A': apple_iter,
        'B': windows_iter
    }

def denormalize(image):
    """ Take TensorFlow tensor and get a 8-bit (0-255) image again """
    return tf.cast(tf.multiply(tf.add(image,1),127.5), tf.uint8)

def show(titles, input_data, num=20, cols=10):
    with tf.Session() as sess:
        """ Display the first images in a given dataset as a sanity check """
        assert len(titles) == len(input_data)
        for i, (name,tensor) in enumerate(input_data.items()): # Plot each provided dataset (probably 2)
            sess.run(tensor.initializer)
            next_image = tensor.get_next()

            fig = plt.figure(figsize=(15, 3))
            plt.axis('off')
            plt.title("Dataset: "+titles[i])

            # If we don't denormalize it, the coloring will be off
            denormalized = denormalize(next_image)

            for j in range(num):
                next_image = sess.run(denormalized)[0] # Only take first image in batch

                ax = fig.add_subplot(np.ceil(num/cols), cols, j+1)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(next_image)

        plt.show()

#
# Implementing the GAN
#
def conv2d(name, inputs, num_outputs, kernel_size, stride, padding,
        stddev=0.02, activation=tf.nn.relu, batchnorm=True, training=False):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride, padding,
                                        activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))

        if batchnorm:
            conv = tf.layers.batch_normalization(conv, training=training)

        if activation is not None:
            return activation(conv)
        else:
            return conv

def deconv2d(name, inputs, num_outputs, kernel_size, stride, padding,
        stddev=0.02, activation=tf.nn.relu, batchnorm=True, training=False):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,
                                                  activation_fn=activation,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if batchnorm:
            conv = tf.layers.batch_normalization(conv, training=training)

        if activation is not None:
            return activation(conv)
        else:
            return conv

def resnet(name, inputs, num_outputs, training=False):
    with tf.variable_scope(name):
        r = tf.pad(inputs, [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
        r = conv2d("c1", r, num_outputs, 3, 1, "VALID", training=training)
        r = tf.pad(r, [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
        r = conv2d("c2", r, num_outputs, 3, 1, "VALID", activation=None, training=training)
        return tf.nn.relu(r + inputs)

class CycleGAN:
    def __init__(self,
                 num_epochs=100,
                 batch_size=128,
                 img_width=72,
                 img_height=72,
                 img_layers=4,
                 generator_residual_blocks=6,
                 log_dir="logs",
                 check_dir="models",
                 eval_images=3, # Probably has to be smaller than the batch size
                 restore=True,
                 history=True,
                 history_size=50):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.img_layers = img_layers
        self.log_dir = log_dir
        self.check_dir = check_dir
        self.eval_images = eval_images
        self.restore = restore
        self.generator_residual_blocks = generator_residual_blocks
        self.history = history
        self.history_size = history_size

    def create_generator(self, name, input_layer, training):
        l = tf.keras.layers
        ngf = 8 # Filter depth for generator
        summaries = []

        with tf.variable_scope(name):
            g_pad = tf.pad(input_layer, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            summaries.append(tf.summary.histogram("g_pad", g_pad))
            g_c1 = conv2d("c1", g_pad, ngf,   7, 1, "VALID", training=training)
            summaries.append(tf.summary.histogram("g_c1", g_c1))
            g_c2 = conv2d("c2", g_c1,  ngf*2, 3, 2, "SAME", training=training)
            summaries.append(tf.summary.histogram("g_c2", g_c2))
            g_c3 = conv2d("c3", g_c2,  ngf*4, 3, 2, "SAME", training=training)
            summaries.append(tf.summary.histogram("g_c3", g_c3))

            assert self.generator_residual_blocks > 0
            g_r = resnet("r1", g_c3, ngf*4, training=training)
            summaries.append(tf.summary.histogram("g_r1", g_r))
            for i in range(self.generator_residual_blocks-1):
                g_r = resnet("r"+str(i+2), g_r, ngf*4, training=training)
                summaries.append(tf.summary.histogram("g_r"+str(i+2), g_r))

            g_c4 = deconv2d("c4", g_r,  ngf*2,           3, 2, "SAME", training=training)
            summaries.append(tf.summary.histogram("g_c4", g_c4))
            g_c5 = deconv2d("c5", g_c4, ngf,             3, 2, "SAME", training=training)
            summaries.append(tf.summary.histogram("g_c5", g_c5))
            g_c6 = conv2d("c6",   g_c5, self.img_layers, 7, 1, "SAME",
                    activation=tf.nn.tanh, batchnorm=False)
            summaries.append(tf.summary.histogram("g_c6", g_c6))

            return g_c6, summaries

    def create_discriminator(self, name, input_layer, training):
        l = tf.keras.layers
        ndf = 16 # Filter depth for discriminator
        summaries = []

        with tf.variable_scope(name):
            d_c1 = conv2d("c1", input_layer, ndf,   4, 2, "SAME",
                    activation=tf.nn.leaky_relu, batchnorm=False)
            summaries.append(tf.summary.histogram("d_c1", d_c1))
            d_c2 = conv2d("c2", d_c1,        ndf*2, 4, 2, "SAME",
                    activation=tf.nn.leaky_relu, training=training)
            summaries.append(tf.summary.histogram("d_c2", d_c2))
            d_c3 = conv2d("c3", d_c2,        ndf*4, 4, 2, "SAME",
                    activation=tf.nn.leaky_relu, training=training)
            summaries.append(tf.summary.histogram("d_c3", d_c3))
            d_c4 = conv2d("c4", d_c3,        ndf*8, 4, 1, "SAME",
                    activation=tf.nn.leaky_relu, training=training)
            summaries.append(tf.summary.histogram("d_c4", d_c4))
            d_c5 = conv2d("c5", d_c4,        1,     4, 1, "SAME",
                    activation=None, batchnorm=False) # Radford didn't say disable batchnorm here but tutorial did it
            summaries.append(tf.summary.histogram("d_c5", d_c5))
            return d_c5, summaries

    def cyclegan_model(self):
        # Get image data directly from features or from "image" if it's a dictionary
        #assert isinstance(input_data, dict) and len(input_data) == 2, "input_data must include both images A and B"
        #image_A = input_data['A']
        #image_B = input_data['B']
        self.image_A = tf.placeholder(tf.float32,
                                      [self.batch_size, self.img_width, self.img_height, self.img_layers],
                                      name="input_A")
        self.image_B = tf.placeholder(tf.float32,
                                      [self.batch_size, self.img_width, self.img_height, self.img_layers],
                                      name="input_B")

        # Input images for evaluation, with only a few (eval_images) rather than the full batch size
        self.eval_image_A = tf.placeholder(tf.float32,
                                      [self.eval_images, self.img_width, self.img_height, self.img_layers],
                                      name="eval_input_A")
        self.eval_image_B = tf.placeholder(tf.float32,
                                      [self.eval_images, self.img_width, self.img_height, self.img_layers],
                                      name="eval_input_B")

        if self.history:
            # Actually stored here
            self.poolB = np.zeros((self.history_size, self.batch_size, self.img_height, self.img_width, self.img_layers))
            self.poolA = np.zeros((self.history_size, self.batch_size, self.img_height, self.img_width, self.img_layers))

            # Passed into TF with these placeholders
            self.hist_pool_A = tf.placeholder(tf.float32,
                                          [None, self.img_width, self.img_height, self.img_layers],
                                          name="hist_pool_A")
            self.hist_pool_B = tf.placeholder(tf.float32,
                                          [None, self.img_width, self.img_height, self.img_layers],
                                          name="hist_pool_B")

        # For keeping track of where we are in training, and restoring from checkpoints
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.iteration = tf.Variable(0, name="iteration", trainable=False)

        # Create models
        with tf.variable_scope("Model") as scope:
            # Generator on original images
            self.gen_AtoB, self.hist_summ_g_A = self.create_generator("gen_AtoB", self.image_A, training=True)
            self.gen_BtoA, self.hist_summ_g_B = self.create_generator("gen_BtoA", self.image_B, training=True)

            # Discriminator on the original real images
            self.disc_Areal, self.hist_summ_d_A = self.create_discriminator("discrim_A", self.image_A, training=True)
            self.disc_Breal, self.hist_summ_d_B = self.create_discriminator("discrim_B", self.image_B, training=True)

            scope.reuse_variables()

            # Generate from fake back to original (for cycle consistency)
            self.gen_AtoBtoA, _ = self.create_generator("gen_BtoA", self.gen_AtoB, training=True) # Reuse weights from BtoA
            self.gen_BtoAtoB, _ = self.create_generator("gen_AtoB", self.gen_BtoA, training=True) # Reuse weights from AtoB

            # Discriminators on the generated fake images
            self.disc_Afake, _ = self.create_discriminator("discrim_A", self.gen_BtoA, training=True)
            self.disc_Bfake, _ = self.create_discriminator("discrim_B", self.gen_AtoB, training=True)

            # Evaluation generator (only diff is input placeholder has batch size of self.eval_images, not self.batch_size)
            scope.reuse_variables()
            self.eval_gen_AtoB, _ = self.create_generator("gen_AtoB", self.eval_image_A, training=False)
            self.eval_gen_BtoA, _ = self.create_generator("gen_BtoA", self.eval_image_B, training=False)
            scope.reuse_variables()
            self.eval_gen_AtoBtoA, _ = self.create_generator("gen_BtoA", self.eval_gen_AtoB, training=False) # Reuse weights from BtoA
            self.eval_gen_BtoAtoB, _ = self.create_generator("gen_AtoB", self.eval_gen_BtoA, training=False) # Reuse weights from AtoB

            # Discriminators on the generated fake images in the history
            if self.history:
                scope.reuse_variables()
                self.hist_disc_Afake, _ = self.create_discriminator("discrim_A", self.hist_pool_A, training=True)
                self.hist_disc_Bfake, _ = self.create_discriminator("discrim_B", self.hist_pool_B, training=True)

        #
        # Loss functions
        #
        # Generator should by cycle consistent & we want the discriminator to output a 1, i.e. incorrect label
        cyc_loss = tf.reduce_mean(tf.abs(self.image_A-self.gen_AtoBtoA)) + \
                   tf.reduce_mean(tf.abs(self.image_B-self.gen_BtoAtoB))
        g_loss_A = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.disc_Bfake,1)) # loss for gen_AtoB
        g_loss_B = cyc_loss*10 + tf.reduce_mean(tf.squared_difference(self.disc_Afake,1)) # loss for gen_BtoA

        # Discriminator should correctly classify the original real images and the generated fake images
        if self.history:
            d_loss_A = (tf.reduce_mean(tf.square(self.hist_disc_Afake)) +
                             tf.reduce_mean(tf.squared_difference(self.disc_Areal,1)))/2
            d_loss_B = (tf.reduce_mean(tf.square(self.hist_disc_Bfake)) +
                             tf.reduce_mean(tf.squared_difference(self.disc_Breal,1)))/2
        else:
            d_loss_A = (tf.reduce_mean(tf.square(self.disc_Afake)) +
                             tf.reduce_mean(tf.squared_difference(self.disc_Areal,1)))/2
            d_loss_B = (tf.reduce_mean(tf.square(self.disc_Bfake)) +
                             tf.reduce_mean(tf.squared_difference(self.disc_Breal,1)))/2

        #
        # Variables
        #
        variables = tf.trainable_variables()
        d_A_vars = [v for v in variables if 'discrim_A' in v.name]
        g_A_vars = [v for v in variables if 'gen_A' in v.name]
        d_B_vars = [v for v in variables if 'discrim_B' in v.name]
        g_B_vars = [v for v in variables if 'gen_B' in v.name]

        #
        # Optimization
        #
        self.learningRate = tf.placeholder(tf.float32, shape=[], name="learningRate")

        # Required to update batch normalization statistics
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate, beta1=0.5)

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        #
        # Summaries for TensorBoard
        #
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

        #
        # For evaluation
        #
        # When running these in the session, change the feed_dict to feed in the evaluation
        # images rather than the training images.
        #
        realA = denormalize(self.eval_image_A)
        realB = denormalize(self.eval_image_B)
        fakeA = denormalize(self.eval_gen_BtoA)
        fakeB = denormalize(self.eval_gen_AtoB)
        cycA = denormalize(self.eval_gen_AtoBtoA)
        cycB = denormalize(self.eval_gen_BtoAtoB)

        self.eval_realA_summ = tf.summary.image("real_A", realA, self.eval_images)
        self.eval_realB_summ = tf.summary.image("real_B", realB, self.eval_images)
        self.eval_fakeA_summ = tf.summary.image("fake_A", fakeA, self.eval_images)
        self.eval_fakeB_summ = tf.summary.image("fake_B", fakeB, self.eval_images)
        self.eval_cycA_summ = tf.summary.image("cyc_A", cycA, self.eval_images)
        self.eval_cycB_summ = tf.summary.image("cyc_B", cycB, self.eval_images)

    def hist_image_pool(self, num_in_pool, new_image, pool):
        """ Shrivastava et al's history technique - we're storing batches though not 1 image """
        # Keep adding till pool is full
        if num_in_pool < self.history_size:
            pool[num_in_pool] = new_image
            return new_image
        else:
            p = random.random()

            # Use image from history 50% of time, replacing it with new image
            if p > 0.5:
                rand_id = random.randint(0, self.history_size-1)
                tmp = pool[rand_id]
                pool[rand_id] = new_image
                return tmp
            # Otherwise, use the new image this time
            else:
                return new_image

    def run(self):
        # Define the networks
        self.cyclegan_model()

        # Saving content checkpoints and summaries to disk
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_dir)

        if not os.path.exists(self.check_dir):
            os.makedirs(self.check_dir)

        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            # Get training input images
            input_data = train_input_fn(self.batch_size,
                                        channels=self.img_layers,
                                        resize=[self.img_width,self.img_height])
            image_A_iter = input_data['A']
            image_B_iter = input_data['B']

            # We start off with no images in the pool
            self.num_in_pool = 0

            # Get evaluation images
            eval_input_data = test_input_fn(self.eval_images, # "batch" is number of images we want
                                        channels=self.img_layers,
                                        resize=[self.img_width,self.img_height])
            eval_image_A_iter = eval_input_data['A']
            eval_image_B_iter = eval_input_data['B']

            # Restore from last checkpoint
            chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
            if self.restore and chkpt_fname is not None:
                saver.restore(sess, chkpt_fname)

            writer.add_graph(sess.graph)

            # Run the training
            for epoch in range(sess.run(self.global_step),self.num_epochs):
                print("Epoch:", epoch)
                saver.save(sess, os.path.join(self.check_dir, "cyclegan"), global_step=epoch)

                # Reinitialize at the beginning of each epoch
                sess.run([image_A_iter.initializer, image_B_iter.initializer])
                next_image_A = image_A_iter.get_next()
                next_image_B = image_B_iter.get_next()

                # Decaying learning rate
                if epoch < 100:
                    currentLearningRate = 0.0002
                else:
                    currentLearningRate = 0.0002 - 0.0002*(epoch-100)/100 # From tutorial

                while True:
                    iteration = sess.run(self.iteration)

                    try:
                        t = time.time()
                        image_A, image_B = sess.run([next_image_A, next_image_B])

                        # Make sure we have a full batch
                        if image_A.shape[0] != self.batch_size or image_B.shape[0] != self.batch_size:
                            print("Incorrect batch sizes:", image_A.shape[0], image_B.shape[0])
                            break

                        # Optimize gen_AtoB
                        _, generatedB, summ = sess.run([self.g_A_trainer, self.gen_AtoB, self.g_A_loss_summ], feed_dict={
                                                     self.image_A: image_A,
                                                     self.image_B: image_B,
                                                     self.learningRate: currentLearningRate
                                                 })
                        writer.add_summary(summ, iteration)

                        # Optimize discrim_B
                        feed_dict = {
                             self.image_A: image_A,
                             self.image_B: image_B,
                             self.learningRate: currentLearningRate,
                        }

                        if self.history:
                            generatedB = self.hist_image_pool(self.num_in_pool, generatedB, self.poolB)
                            feed_dict[self.hist_pool_B] = generatedB

                        _, summ = sess.run([self.d_B_trainer, self.d_B_loss_summ], feed_dict=feed_dict)
                        writer.add_summary(summ, iteration)

                        # Optimize gen_BtoA
                        _, generatedA, summ = sess.run([self.g_B_trainer, self.gen_BtoA, self.g_B_loss_summ], feed_dict={
                                                     self.image_A: image_A,
                                                     self.image_B: image_B,
                                                     self.learningRate: currentLearningRate
                                                 })
                        writer.add_summary(summ, iteration)

                        # Optimize discrim_A
                        feed_dict = {
                             self.image_A: image_A,
                             self.image_B: image_B,
                             self.learningRate: currentLearningRate
                        }

                        if self.history:
                            generatedA = self.hist_image_pool(self.num_in_pool, generatedA, self.poolA)
                            feed_dict[self.hist_pool_A] = generatedA

                        _, summ = sess.run([self.d_A_trainer, self.d_A_loss_summ], feed_dict=feed_dict)
                        writer.add_summary(summ, iteration)

                        # Log time to execute this step
                        t = time.time() - t
                        summ = tf.Summary(value=[tf.Summary.Value(tag="step_time", simple_value=t)])
                        writer.add_summary(summ, iteration)

                        # Log the weights
                        if iteration%50 == 0:
                            weight_summaries = self.hist_summ_g_A+self.hist_summ_g_B+self.hist_summ_d_A+self.hist_summ_d_B
                            summaries = sess.run(weight_summaries, feed_dict=feed_dict)

                            for s in summaries:
                                writer.add_summary(s, iteration)

                        # We've added one more image (batch) to the history
                        self.num_in_pool += 1

                    except tf.errors.OutOfRangeError:
                        break

                    # Evaluation
                    if iteration%100 == 0:
                        # Reset iterators for evaluation
                        sess.run([eval_image_A_iter.initializer, eval_image_B_iter.initializer])
                        eval_image_A, eval_image_B = sess.run([eval_image_A_iter.get_next(), eval_image_B_iter.get_next()])

                        # Make sure we have enough evaluation images (otherwise placeholder won't be filled and will error)
                        if eval_image_A.shape[0] != self.eval_images or eval_image_B.shape[0] != self.eval_images:
                            print("Incorrect evaluation batch sizes:", eval_image_A.shape[0], eval_image_B.shape[0])
                        else:
                            # Generate eval images
                            summaries = sess.run([
                                    self.eval_realA_summ, self.eval_realB_summ,
                                    self.eval_fakeA_summ, self.eval_fakeB_summ,
                                    self.eval_cycA_summ,  self.eval_cycB_summ
                                ], feed_dict={
                                    self.eval_image_A: eval_image_A,
                                    self.eval_image_B: eval_image_B
                                })

                            for s in summaries:
                                writer.add_summary(s, iteration)

                    # To see results every so often in TensorBoard
                    if iteration%100 == 0:
                        writer.flush()

                    # Increment iteration since we've finished another image
                    sess.run(tf.assign(self.iteration, iteration+1))

                # Increment global step since we've finished another epoch
                sess.run(tf.assign(self.global_step, epoch+1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--batch', default=128, type=int, help="Batch size")
    parser.add_argument('--width', default=72, type=int, help="Image width")
    parser.add_argument('--height', default=72, type=int, help="Image height")
    parser.add_argument('--channels', default=4, type=int, help="Image channels (e.g. 4 if RGBA)")
    parser.add_argument('--res', default=6, type=int, help="Number of residual blocks for generator")
    parser.add_argument('--modeldir', default="models", type=str, help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str, help="Directory for saving log files")
    parser.add_argument('--eval', default=3, type=int, help="Number of images to use for evaluation")
    parser.add_argument('--restore', dest='restore', action='store_true', help="Restore from saved checkpoints (default)")
    parser.add_argument('--no-restore', dest='restore', action='store_false', help="Do not restore from saved checkpoints")
    parser.add_argument('--history', dest='history', action='store_true', help="Use history of past generated images (default)")
    parser.add_argument('--no-history', dest='history', action='store_false', help="Do not use history of past generated images")
    parser.add_argument('--histsize', default=50, type=int, help="Number of images to use in history")
    parser.add_argument('--display', dest='display', action='store_true', help="Display samples from dataset instead of training")
    parser.set_defaults(restore=True, history=True, display=False)
    args = parser.parse_args()

    if args.display:
        show(["Train Apple", "Train Windows"], train_input_fn())
        show(["Test Apple", "Test Windows"], test_input_fn())
    else:
        CycleGAN(
            num_epochs=args.epochs,
            batch_size=args.batch,
            img_width=args.width,
            img_height=args.height,
            img_layers=args.channels,
            generator_residual_blocks=args.res,
            log_dir=args.logdir,
            check_dir=args.modeldir,
            eval_images=args.eval,
            restore=args.restore,
            history=args.history,
            history_size=args.histsize
        ).run()
