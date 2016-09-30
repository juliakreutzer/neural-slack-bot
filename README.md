### Neural Slack Bot ###

Train a neural dialogue bot on your slack conversations and integrate it into your slack team. Mostly fun project, inspired by Vinyals & Le's work on "A Neural Conversational Model" (https://arxiv.org/abs/1506.05869). Data pre-processing is done using heuristics. Having loose constraints on your conversational data, the bot is guaranteed to produce (more or less fluent) nonsense. Enjoy!

1. **Download slack conversations:**

   Follow the instructions here: https://gist.github.com/Chandler/fb7a070f52883849de35
   This will allow you to download the complete history of your team's public channels. For this you need an API token for your user, that you can request here: https://api.slack.com/docs/oauth-test-tokens. Store this token in the root directory of this project in a file 'token.id'.

2. **Preprocess:** 

   `python preprocess.py`
   
   Reads the slack conversations from `slack-data/channels/`, splits sentences, tokenizes and filters (heuristically!), and stores final corpus and its vocabulary in `slack-data/corpus/`.

3. **Train the bot:**

   `python translate.py`
   
   Train a neural translation system to predict the reponse for each message. Parameters need to be tuned. The default parameters are only recommended if you have a lot of data. Otherwise, start with smaller `size` and `num_layers`. The `data_dir` should point to the directory where the slack corpus was stored. Create a directory for the checkpointing of the models and specify its location with `model_dir`.
   
   The model is tensorflow's standard neural machine translation model, learn more about it in the official tutorial: https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html. The optimizer is changed to Adam. 
   
   Each `steps_per_checkpoint` steps, the model is saved and perplexity on the training set is reported. Since the model uses greedy decoding, perplexities around 3-5 are required to produce fluent output.
   Training on a GPU speeds up training, but the code runs also on CPU.

4. **Test the bot:** 

   `python translate.py --decode`
   
   With the same parameters as the training, load a model by this command and try the interactive decoding part to make sure your bot is capable of sensible conversations.

5. **Create the bot:** 

   Create or choose a channel in your slack team to integrate your bot. This is where the bot reacts to every incoming message by anyone. Let's call this channel `BOT_CHANNEL`.
   
   Register a bot in your slack team's app integrations and store its token in a file `bot.token` in the root directory of this repository. (Build -> Make a Custom Integration -> Bots, pick a Username `BOT_NAME`) 

6. **Deploy the bot:**

   `python run_bot.py --batch_size 1`
   
   First, set the constants `BOT_NAME` and `BOT_CHANNEL` in the first lines of `run_bot.py`to your chosen bot name and channel.
   Run this code with the same parameters that were used for training, except for `batch_size=1`.
   If the code doesn't report any failure, your bot is running!

7. **Chat with the bot:**

   In slack, move to the `BOT_CHANNEL` and write a message. Your bot should automatically reply.
   In addition to that, the bot reacts on mentions, so addressing `@BOT_NAME` will make it respond.

8. **Get creative:**

   This is a very basic and simple bot. Try to improve it by e.g. move away from greedy decoding, find good learning parameter settings, increase your corpus with movie dialogues, make the bot respond to more messages, let bots talk with each other, etc. Slack's RTM API (https://api.slack.com/rtm) offers a wide range of options to make your bot interact with other users. Have fun!
   
