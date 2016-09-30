import time
from slackclient import SlackClient
from translate import create_model
import tensorflow as tf
import json
import cPickle as pkl
import embedding
import utils
import random

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

READ_WEBSOCKET_DELAY = 0.5
TOKEN = open("bot.token", "r").readline().strip()
BOT_CHANNEL = "mybotchannel"
BOT_NAME = "mybot"

def handle_command(command, channel, model, embeddings, metadata):
    if command.startswith("###welcome"):
        response = command[3:]
    else:
        sentence = command.strip()
        if "who are you" in sentence.lower():
            response = "read this :nerd_face: http://arxiv.org/abs/1506.05869"
        else:
            response = utils.get_translation(sess, model, sentence, embeddings, _buckets, metadata)
    slack_client.api_call("chat.postMessage", channel=channel, text=response, as_user=True)

def parse_slack_output(slack_rtm_output):
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            #print output
            if output['type'] == 'team_join':
                return "###welcome <@%s>! :%s:" % (output['user']['id'],
                                                   random.choice(EMOJIS)), BOT_CHANNEL_ID
            if output and 'text' in output:
                bot_mention = "<@%s>" % BOT_ID.upper()
                if bot_mention in output['text'] \
                        or output['channel'] == BOT_CHANNEL_ID:
                    if output['user'] != BOT_ID.upper():
                        # bot reacts to direct mentions and everything in its own channel, but not to his own texts
                        return output['text'].replace(bot_mention, "").strip().lower(), output['channel']
    return None, None

if __name__ == "__main__":

    # usage:
    # python run_bot.py --num_layers 1 --size 200  --data_dir slack-data/ --model_dir models

    slack_client = SlackClient(TOKEN)

    api_call = slack_client.api_call("users.list")
    if api_call.get('ok'):
        # retrieve all users so we can find our bot
        users = api_call.get('members')
        for user in users:
            if 'name' in user and user.get('name') == BOT_NAME:
                BOT_ID = user.get('id')
    api_call = slack_client.api_call("channels.list")
    for c in api_call.get('channels'):
        if c['name'] == BOT_CHANNEL:
            BOT_CHANNEL_ID = c['id']
    api_call = slack_client.api_call("emoji.list")
    if api_call.get('ok'):
        EMOJIS = api_call.get('emoji').keys()
        print EMOJIS
    # load the tf model
    with tf.Session() as sess:
        # Load slack metadata
        metadata = None
        with open("metadata.json", "r") as m:
            metadata = json.load(m)

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
        model = create_model(sess, True, len(word2id))

        if slack_client.rtm_connect():
            print "%s running: id %s, token %s" % (BOT_NAME, BOT_ID, TOKEN)
            while True:
                command, channel = parse_slack_output(slack_client.rtm_read())
                if command and channel:
                    handle_command(command, channel, model, embeddings, metadata)
                time.sleep(READ_WEBSOCKET_DELAY)
        else:
            print "%s failed" % BOT_NAME
