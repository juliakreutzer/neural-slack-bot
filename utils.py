# coding=utf-8
import numpy as np
import tensorflow as tf
import cPickle as pkl
import codecs
import embedding
from scipy import stats
import operator
import logging
import random

def load_embedding(pkl_file):
    word2id = {}
    id2word = {}
    with codecs.open(pkl_file, "rb", "utf8", "replace") as opened:
        words, vectors = pkl.load(opened)
        assert len(words) == len(vectors)
        UNK_id = words.index("<UNK>")
        PAD_id = words.index("<PAD>")
        start_id = words.index("<S>")
        end_id = words.index("</S>")
        word2id["<s>"] = start_id
        word2id["</s>"] = end_id
        for i, w in enumerate(words):
            word2id[w] = i
            id2word[i] = w
    logging.info("Loaded embeddings for %d words with dimensionality %d" % (len(words), len(vectors[0])))
    #print "Special tokens:", UNK_id, PAD_id, start_id, end_id
    emb = embedding.Embedding(vectors, word2id, id2word, UNK_id, PAD_id, end_id, start_id)
    return emb


def build_vocab(feature_file, origin, store=False):
    # FIXME not very efficient
    vocab = ["<PAD>", "<UNK>", "<s>", "</s>"]
    with codecs.open(feature_file, "r", "utf8", "replace") as qe_data:
        for line in qe_data:
            stripped = line.strip()
            if stripped == "":  # sentence end
                continue
            else:
                split_line = stripped.split()
                if origin == "tgt":
                    tokens = [split_line[3]]
                else:
                    tokens = split_line[6:8]
                for token in tokens:
                    if token not in vocab:
                        vocab.append(token)
    logging.info("Built %s vocabulary of %d words" % (origin, len(vocab)))
    if store:
        dump_file = feature_file+".vocab."+origin+".pkl"
        pkl.dump(vocab, open(dump_file, "wb"))
        logging.info("Stored %s vocabulary in %s" % (origin, dump_file))
    return vocab, 0, 1, 2, 3


def load_data(corpus_file, word2id, max_sent=0):
    """
    Given a dataset file and word2id embeddings, read them to lists
    :param feature_label_file:
    :param max_sent:
    :param word2id:
    :return:
    """
    end_id = word2id["</s>"]
    PAD_id = word2id["PAD"]
    UNK_id = word2id["UNK"]
    start_id = word2id["<s>"]
    id2word = {v:k for (k,v) in word2id.items()}
    word_embedding = embedding.Embedding(None, word2id, id2word, UNK_id, PAD_id, end_id, start_id)

    # load features and labels
    feature_vectors = []
    sentences = []
    labels = []
    with codecs.open(corpus_file, "r", "utf8", "replace") as data:
        i = 0
        for line in data:
            if i >= max_sent and max_sent > 0:
                break
            stripped = line.strip()
            tokens = stripped.split()
            vector = word_embedding.encode(tokens)
            vector.append(end_id)  # add </s> to sentence
            sentences.append(tokens)
            if i==0:
                # no previous dialogue available
                feature_vectors.append(vector)
            else:
                # input is previous sentence
                feature_vectors.append(labels[i-1])
            vector = [start_id]+vector  # re-pend start_id to decoder inputs
            labels.append(vector)
            i += 1

    logging.info("Loaded %d sentences" % len(feature_vectors))

    return feature_vectors, sentences, labels

def prepare_input_sent(sentence, embeddings, buckets):
    # Get token-ids for the input sentence.
    token_ids = embeddings.encode(sentence.lower().strip().split()) # TODO properly tokenize
    #print("word 2 ids: %s" % str(token_ids))
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(buckets))
                     if buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    #print("bucket: %d" % bucket_id)
    #print(_buckets[bucket_id])

    encoder_inputs = token_ids+[embeddings.PAD_id]*(buckets[bucket_id][0]-len(token_ids))
    #print("encoder inputs: %s" % str(encoder_inputs))
    decoder_inputs = [embeddings.start_id] + [embeddings.PAD_id]*(buckets[bucket_id][1]-1)  # only start symbol
    #print("decoder inputs: %s" % str(decoder_inputs))
    target_weights = np.ones_like(decoder_inputs)
    #print("target weights: %s" % str(target_weights))
    return encoder_inputs, decoder_inputs, target_weights, bucket_id


def process_output(logits, embeddings):
    response = ""
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in logits]
    #print("outputs: %s" % str(outputs))
    # If there is an EOS symbol in outputs, cut them at that point.
    if embeddings.end_id in outputs:
        outputs = outputs[:outputs.index(embeddings.end_id)]  
    #print("outputs: %s" % str(outputs))
    if sum([1 if i == embeddings.UNK_id else 0 for i in outputs]) > 0.5*len(outputs):
        # many unks
        unk_sents = ["i don't know.", "i have no clue.", "no idea.",
                     "i don't understand.", "ask @slackbot!",
                     "i love it!", "how do you mean?"] 
        reponse = random.choice(unk_sents)
    else:
        response = " ".join(embeddings.decode(outputs))
        response = response.replace("UNK", random.choice(embeddings.word2id.keys()))  # replace UNK by random word
        response = response.replace("PAD", "")
    return response

def pad_data(X, Y, max_len, PAD_symbol=0):
    """
    Pad data up till maximum length and create masks and lists of sentence lengths
    :param X:
    :param Y:
    :param max_len:
    :return:
    """
    #print "to pad", X[0], Y[0]
    feature_size = len(X[0][0])
    #print "feature size", feature_size
    seq_lens = []
    masks = np.zeros(shape=(len(X), max_len), dtype=int)
    i = 0
    X_padded = np.zeros(shape=(len(X), max_len, feature_size), dtype=int)
    X_padded.fill(PAD_symbol)
    Y_padded = np.zeros(shape=(len(Y), max_len), dtype=int)
    Y_padded.fill(PAD_symbol)

    for x, y in zip(X, Y):
        assert len(x) == len(y)
        seq_len = len(x)
        if seq_len > max_len:
            seq_len = max_len
        seq_lens.append(seq_len)
        for j in range(seq_len):
            masks[i][j] = 1
            X_padded[i][j] = x[j]
            Y_padded[i][j] = y[j]
        i += 1
    #print "padded", X_padded[0], seq_lens[0]
    return X_padded, Y_padded, masks, np.asarray(seq_lens)


def buckets_by_length(data, labels, buckets=20, max_len=50, mode='pad'):
    """
    :param data: numpy arrays of data
    :param labels: numpy arrays of labels
    :param buckets: list of buckets (lengths) into which to group samples according to their length.
    :param mode: either 'truncate' or 'pad':
                * When truncation, remove the final part of a sample that does not match a bucket length;
                * When padding, fill in sample with zeros up to a bucket length.
                The obvious consequence of truncating is that no sample will be padded.
    :return: a dictionary of grouped data and a dictionary of the data original indexes, both keyed by bucket, and the bin edges
    """
    input_lengths = np.array([len(s) for s in data], dtype='int')  # for dev and train (if dev given)

    maxlen = max_len if max_len > 0 else max(input_lengths) + 1

    # sort data by length
    # split this array into 'bucket' many parts, these are the buckets
    data_lengths_with_idx = [(len(s), i) for i, s in enumerate(data)]
    sorted_data_lengths_with_idx = sorted(data_lengths_with_idx, key=operator.itemgetter(0))
    bucket_size = int(np.ceil(len(data)/float(buckets)))
    logging.info("Creating %d Buckets of size %d" % (buckets, bucket_size))
    buckets_data = [sorted_data_lengths_with_idx[i:i+bucket_size] for i in xrange(0, len(sorted_data_lengths_with_idx), bucket_size)]
    bin_edges = [min(bucket[-1][0], max_len) for bucket in buckets_data]  # max len of sequence in bucket
    logging.info("bin_edges %s" % str(bin_edges))
    if bin_edges[-1] < maxlen:
        bin_edges[-1] = maxlen
    logging.info("final bin_edges %s" % str(bin_edges))
    input_bucket_index = np.zeros(shape=len(data), dtype=int)
    for bucket_idx, bucket in enumerate(buckets_data):
        for l, d_idx in bucket:
            input_bucket_index[d_idx] = bucket_idx

    # pad and bucket train data
    bucketed_data = {}
    reordering_indexes_train = {}
    for bucket in list(np.unique(input_bucket_index)):
        length_indexes = np.where(input_bucket_index == bucket)[0]
        reordering_indexes_train[bucket] = length_indexes
        maxlen = bin_edges[bucket]
        padded_data = pad_data(data[length_indexes], labels[length_indexes], max_len=maxlen)
        bucketed_data[bucket] = padded_data # in final dict, start counting by zero

    return bucketed_data, reordering_indexes_train, bin_edges


def put_in_buckets(data_array, labels, buckets, mode='pad'):
    """
    Given bucket edges and data, put the data in buckets according to their length
    :param data_array:
    :param labels:
    :param buckets:
    :return:
    """
    input_lengths = np.array([len(s) for s in data_array], dtype='int')
    input_bucket_index = [i if i<len(buckets) else len(buckets)-1 for i in np.digitize(input_lengths, buckets, right=False)]  # during testing, longer sentences are just truncated
    if mode == 'truncate':
        input_bucket_index -= 1
    bucketed_data = {}
    reordering_indexes = {}
    for bucket in list(np.unique(input_bucket_index)):
        length_indexes = np.where(input_bucket_index == bucket)[0]
        reordering_indexes[bucket] = length_indexes
        maxlen = int(np.floor(buckets[bucket]))
        padded = pad_data(data_array[length_indexes], labels[length_indexes], max_len=maxlen)
        bucketed_data[bucket] = padded  # in final dict, start counting by zero
    return bucketed_data, reordering_indexes


def put_in_double_buckets(inputs, outputs, buckets, PAD_id):
    """
    Put in bucket according to input and label length
    :param inputs:
    :param labels:
    :param buckets:
    :param mode:
    :return:
    """
    bucketed_data = {}
    reordering_indexes = {}
    i = 0
    for input, output in zip(inputs, outputs):
        for bucket_id, (input_size, output_size) in enumerate(buckets):  # first determine bucket for each sample
            if len(input) <= input_size and len(output) <= output_size:
                items_in_bucket = reordering_indexes.get(bucket_id, [])
                items_in_bucket.append(i)
                reordering_indexes[bucket_id] = items_in_bucket
                break
        i += 1
    for bucket_id, (input_size, output_size) in enumerate(buckets):  # then collect data for each bucket
        indexes = np.asarray(reordering_indexes[bucket_id])
        padded_data = pad_data_double(inputs[indexes], outputs[indexes], input_size, output_size, PAD_id)
        bucketed_data[bucket_id] = padded_data
    return bucketed_data, reordering_indexes


def pad_data_double(inputs, outputs, input_size, output_size, PAD_id):
    """
    Pad input data up till input_size, output up till output_size
    :param x:
    :param y:
    :param input_size:
    :param output_size:
    :param PAD_id:
    :return:
    """
    padded_inputs = np.zeros(shape=(len(inputs), input_size))
    padded_inputs.fill(PAD_id)
    input_lens = np.zeros(shape=len(inputs))
    i = 0
    j = 0
    for i, input in enumerate(inputs):  # sentences
        for j, w in enumerate(input):  # words
            padded_inputs[i][j] = w
        input_lens[i] = j
    padded_outputs = np.zeros(shape=(len(outputs), output_size))
    padded_outputs.fill(PAD_id)
    output_lens = np.zeros(shape=len(outputs))
    masks = np.zeros_like(padded_outputs)
    i = 0
    j = 0
    for i, output in enumerate(outputs):
        for j, w in enumerate(output):
            padded_outputs[i][j] = w
            masks[i][j] = 1
        output_lens[i] = j
    return padded_inputs, padded_outputs, input_lens, output_lens, masks
