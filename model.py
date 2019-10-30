
import torch
import torch.nn as nn

torch.manual_seed(1)

from NER.utils import argmax, prepare_sequence, log_sum_exp

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BaseBiLSTM_CRF(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class LayerCharCNN(nn.Module):
    """ LayerCharCNN implements character-level convolutional 1D layer.
    source: https://github.com/achernodub/targer/blob/master/src/layers/layer_char_cnn.py
    """

    def __init__(self, char_embeddings_dim, filter_num, char_window_size, word_len):
        super().__init__()
        self.word_len = word_len
        self.char_embeddings_dim = char_embeddings_dim
        self.char_cnn_filter_num = filter_num
        self.char_window_size = char_window_size
        self.output_dim = char_embeddings_dim * filter_num
        self.conv1d = nn.Conv1d(in_channels=char_embeddings_dim,
                                out_channels=char_embeddings_dim * filter_num,
                                kernel_size=char_window_size,
                                groups=char_embeddings_dim)

    def _get_char_embeds(sentence):

        max_seq_len = len(sentence)

        input_tensor = torch.zeros(1, max_seq_len, self.word_len, dtype=torch.long)
        for n, curr_char_seq in enumerate(char_sequences):
            curr_seq_len = len(curr_char_seq)
            curr_char_seq_tensor = self.char_seq_indexer.get_char_tensor(curr_char_seq,
                                                                         self.word_len)  # curr_seq_len x word_len
            input_tensor[n, :curr_seq_len, :] = curr_char_seq_tensor
        char_embeddings_feature = self.embeddings(input_tensor)

        char_ixs = torch.tensor([[self.char_to_ix[ch] for ch in word] for word in sentence], dtype=torch.long)
        char_ixs = self.char_embeds(char_ixs)  #
        char_embeds = self.char_cnn()  # batch_num x max_seq_len x char_embeddings_dim x word_len

    def forward(self, char_embeddings_feature):  # batch_num x max_seq_len x char_embeddings_dim x word_len
        batch_num, max_seq_len, char_embeddings_dim, _ = char_embeddings_feature.shape
        max_pooling_out = torch.zeros(batch_num,
                                      max_seq_len,
                                      self.char_cnn_filter_num * self.char_embeddings_dim,
                                      dtype=torch.float)
        for k in range(max_seq_len):
            max_pooling_out[:, k, :], _ = torch.max(self.conv1d(char_embeddings_feature[:, k, :, :]), dim=2)

        return max_pooling_out  # shape: batch_num x max_seq_len x filter_num*char_embeddings_dim


char_layer = LayerCharCNN(char_embeddings_dim=20, filter_num=1, char_window_size=2, word_len=5)
words = torch.randn(1, 7, 20, 5)  # 7 words by max 5 chars, with len of char emb 20
char_layer(words).size()