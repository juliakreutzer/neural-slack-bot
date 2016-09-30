import re
import json
import os
import codecs
import sys
import cPickle as pkl

# load json files, filter messages (remove those with urls), tokenize (by regexp)
def tokenize_lowercase(message):
    # tokenize at ,.?!
    chars = ['.', ',', ';', '?', '!']
    for c in chars:
        if c in message:
            prefix = " "+c
            m = prefix.join(message.split(c))
            message = m
    filtered_m = message.lower().replace("  ", " ").replace("(", "").replace(")", "") # delete brackets
    return filtered_m

def split_sents(message, split_symbol):
    # insert a split symbol into message for splitting sentences
    chars = ['.', '?', '!', ')', '(']
    m = message
    for c in chars:
        if c in message:
            split_m = message.strip().split(c)
            if len(split_m) > 1:
                if sum([1 if len(part)<4 else 0 for part in split_m]) < 3:
                    prefix = c+split_symbol
                    m = prefix.join(split_m)
    if "\n" in m:
        m = m.replace("\n", split_symbol)
    return m

def get_messages(channels_dir):
    channels = os.listdir(channels_dir)
    print "Preprocessing %d channels: %s\n" % (len(channels), str(channels))
    all_messages = []
    for channel in channels:
        channel_sents = 0
        with codecs.open(channels_dir+"/"+channel, "r", "utf8") as c:
            loaded = json.load(c)
            channel_messages = loaded["messages"]
            for message in channel_messages:
                if message["type"] == "message":
                    if "subtype" in message.keys() or "bot" in message["user"].lower():
                        continue
                    message_text = message["text"]
                    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message_text)
                    mailtos = re.findall('<mailto:.*', message_text)
                    if len(urls) >= 1 or len(mailtos):
                        continue
                    tokenized = tokenize_lowercase(message_text)
                    if len(tokenized) < 3:
                        continue
                    split_symbol = " ### "
                    sents = split_sents(tokenized, split_symbol).split(split_symbol)
                    channel_sents += len(sents)
                    all_messages.extend(sents)
        print "%d sentences from channel %s" % (channel_sents, channel)
    print "total: %d sentences" % (len(all_messages))
    return all_messages

def is_mostl_numeric(token):
    """
    Checks whether the string contains at least 50% numbers
    :param token:
    :return:
    """
    a = len(token)
    for i in range(0, 10):
        token = token.replace(str(i), "")
    if len(token) < 0.5*a and len(token) != a:
        return True
    else:
        return False

def get_vocab(messages, limit):
    word2id = {"PAD": 0, "<s>":1, "</s>":2, "UNK": 3}
    counts = {}
    for m in messages:
        tokens = m.split()
        for t in tokens:
            if is_mostl_numeric(t):
                continue
            if t[0] in [u'.', u';', u'!', u'-', '.', ';', '!', '-'] and len(t) > 1:
                continue
            if len(t) < 20:  # filter out too long words
                c = counts.get(t, 0)
                counts[t] = c+1
    vocab = sorted(counts, key=counts.get, reverse=True)[:limit]
    for v in vocab:
        word2id[v] = len(word2id)
    return word2id

def write_corpus(corpus_dir, sentences, vocab):
    """
    write the sentences to a text file, dump the vocabulary
    :param corpus_dir:
    :param sentences:
    :param vocab:
    :return:
    """
    vocab_file = corpus_dir+"/vocab.pkl"
    sents_file = corpus_dir+"/corpus.txt"
    with open(vocab_file, "wb") as vf:
        pkl.dump(vocab, vf)
    with codecs.open(sents_file, "w") as sf:  #, "utf8"
        c = 0
        for sent in sentences:
            if len(sent) > 5:
                c += 1
                sf.write(sent.strip()+"\n")
    print "Dumped vocab in %s" % vocab_file
    print "Wrote corpus (%d sents.) to %s" % (c, sents_file)


def main():
    channels_dir = "slack-data/channels"
    corpus_dir = "slack-data/corpus"
    messages = get_messages(channels_dir)
    limit = 15000
    vocab = get_vocab(messages, limit)
    write_corpus(corpus_dir, messages, vocab)

if __name__ == "__main__":
    main()
