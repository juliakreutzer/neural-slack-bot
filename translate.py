# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import cPickle as pkl

import numpy as np
import tensorflow as tf

import seq2seq_model
import utils
import embedding


tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "slack-data/corpus", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "models/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(session, forward_only, vocab_size):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    with tf.Session() as sess:
        # Read data into buckets and compute their sizes.
        print("Reading training data (limit: %d)."
              % FLAGS.max_train_data_size)
        # load data and embeddings
        train_file = FLAGS.data_dir+"/corpus.txt"
        vocab_file = FLAGS.data_dir+"/vocab.pkl"
        word2id = pkl.load(open(vocab_file, "rb"))
        id2word = {v:k for (k,v) in word2id.items()}

        embeddings = embedding.Embedding(None, word2id,
                                             id2word,
                                             word2id["UNK"],
                                             word2id["PAD"],
                                             word2id["</s>"],
                                             word2id["<s>"])
        vocab_size = len(word2id)

        train_feature_vectors, train_sentences, train_labels = \
            utils.load_data(train_file, word2id, max_sent=FLAGS.max_train_data_size)

        print("vocab size: %d" % vocab_size)
        print("Training on %d instances" % len(train_labels))
        print("Maximum sentence length (train): %d" % max([len(y) for y in train_labels]))
        print("Average sentence length (train): %d" % np.mean([len(y) for y in train_labels]))

        # bucketing training data

        # equal bucket sizes
        #buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]  #pre-define buckets
        data_buckets, reordering_indexes = utils.put_in_double_buckets(
            np.asarray(train_feature_vectors),
            np.asarray(train_labels), _buckets, embeddings.PAD_id)
        bucket_sizes = [0]*len(_buckets)
        for i, indx in reordering_indexes.items():
            bucket_sizes[i] = len(indx)
        print("Bucket sizes: %s" % str(bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        buckets_scale = [
            sum(bucket_sizes[:i + 1]) / len(train_labels)
            for i in xrange(len(bucket_sizes))]

        print("Bucket scale: %s" % str(buckets_scale))

        # Create model.
        print(
            "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, vocab_size)


        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to the number of samples within
            probs = np.array(buckets_scale)/sum(buckets_scale)
            bucket_id = np.random.choice(range(len(buckets_scale)), p=probs)
            #print("Bucket %d" % bucket_id)

            # Get a batch and make a step.
            start_time = time.time()
            bucket_xs, bucket_ys, input_lens, output_lens, bucket_masks = data_buckets[bucket_id]
            # random order of samples in batch
            order = np.random.permutation(len(bucket_xs))
            batch_samples = order[:FLAGS.batch_size]
            #print("Batch samples: %s" % str(batch_samples))
            # get a batch from this bucket
            encoder_inputs = bucket_xs[batch_samples]  # TODO reverse inputs?
            decoder_inputs = bucket_ys[batch_samples]
            target_weights = bucket_masks[batch_samples]
            #print(encoder_inputs.shape, decoder_inputs.shape, target_weights.shape)  # batch x seq_len  -> transpose as input


            _, step_loss, _ = model.step(sess, encoder_inputs.transpose(), decoder_inputs.transpose(),
                                         target_weights.transpose(), bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float(
                    "inf")
                print(
                    "global step %d learning rate %.4f step-time %.2f perplexity "
                    "%.2f" % (
                        model.global_step.eval(), model.learning_rate.eval(),
                        step_time, perplexity))
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.model_dir,
                                               "translate.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
                step_time, loss = 0.0, 0.0

def decode():
    with tf.Session() as sess:

        # Load vocabularies.
        vocab_file = FLAGS.data_dir+"/vocab.pkl"
        word2id = pkl.load(open(vocab_file, "rb"))
        id2word = {v:k for (k,v) in word2id.items()}

        embeddings = embedding.Embedding(None, word2id,
                                             id2word,
                                             word2id["UNK"],
                                             word2id["PAD"],
                                             word2id["</s>"],
                                             word2id["<s>"])

        # Create model and load parameters.
        FLAGS.batch_size = 1  # We decode one sentence at a time.
        model = create_model(sess, True, len(word2id))

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            encoder_inputs, decoder_inputs, target_weights, bucket_id = utils.prepare_input_sent(sentence, embeddings, _buckets)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, np.array([encoder_inputs]).transpose(),
                                             np.array([decoder_inputs]).transpose(),
                                             np.array([target_weights]).transpose(), bucket_id, True)
            print(utils.process_output(output_logits, embeddings))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
