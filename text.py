"""Text processing.

The :class:`TextEncoder` class is the main feature of this module, the helper
functions were used in earlier examples and should be phased out.
"""

from collections import Counter, namedtuple

import numpy as np
import theano
from theano import tensor as T

Encoded = namedtuple('Encoded', ['sequence', 'unknown'])

class TextEncoder(object):
    def __init__(self,
                 max_vocab=None,
                 min_count=None,
                 vocab=None,
                 sub_encoder=None,
                 special=('<S>', '</S>', '<UNK>')):
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.sub_encoder = sub_encoder
        self.special = special

        if isinstance(vocab, Counter):
            self.vocab = None
            self.counter = vocab
        elif vocab is not None:
            self.vocab = vocab
            self.counter = None
        else:
            self.vocab = None
            self.counter = Counter()

    def done(self):
        if self.vocab is None:
            if self.max_vocab is not None:
                self.vocab = self.special + tuple(
                        s for s,_ in self.counter.most_common(self.max_vocab))
            elif min_count is not None:
                self.vocab = self.special + tuple(
                        s for s,n in self.counter.items() if n >= self.min_count)
            else:
                self.vocab = self.special + tuple(self.counter.keys())
            self.counter = None

        self.index = {s:i for i,s in enumerate(self.vocab)}

    def fields(self):
        return ('surface', len(self))

    def __str__(self):
        if self.sub_encoder is None:
            return '{}({})'.format(self.__class__.__name__, len(self))
        else:
            return '{}({}, {})'.format(self.__class__.__name__,
                                       len(self), str(self.sub_encoder))

    def __repr__(self):
        return str(self)

    def __getitem__(self, x):
        return self.index.get(x, self.index.get('<UNK>'))

    def __len__(self):
        return len(self.vocab)

    def encode_sequence(self, sequence, max_length=None, dtype=np.int32):
        """
        returns:
            an Encoded namedtuple, with the following fields:
            sequence --
                numpy array of symbol indices.
                Negative values index into the unknowns list,
                while positive values index into the encoder lexicon.
            unknowns --
                list of unknown tokens as Encoded(seq, None) tuples,
                or None if no subencoder.
        """
        start = (self.index['<S>'],) if '<S>' in self.index else ()
        stop = (self.index['</S>'],) if '</S>' in self.index else ()
        unk = self.index.get('<UNK>')
        unknowns = None if self.sub_encoder is None else []
        def encode_item(x):
            idx = self.index.get(x)
            if idx is None:
                if unknowns is None:
                    return unk
                else:
                    encoded_unk = self.sub_encoder.encode_sequence(x)
                    unknowns.append(encoded_unk)
                    return -len(unknowns)
            else:
                return idx
        encoded = tuple(idx for idx in list(map(encode_item, sequence))
                        if idx is not None)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            out = start + encoded + stop
        else:
            out = start + encoded[:max_length-(len(start)+len(stop))] + stop
        return Encoded(np.asarray(out, dtype=dtype), unknowns)

    def decode_sentence(self, encoded):
        start = self.index.get('<S>')
        stop = self.index.get('</S>')
        return [''.join(self.sub_encoder.decode_sentence(
                    encoded.unknown[-x-1]))
                if x < 0 else self.vocab[x]
                for x in encoded.sequence
                if x not in (start, stop)]

    def pad_sequences(self, encoded_sequences,
                      max_length=None, pad_right=True, dtype=np.int32):
        """
        arguments:
            encoded_sequences -- a list of Encoded(encoded, unknowns) tuples.
        """
        if not encoded_sequences:
            # An empty matrix would mess up things, so create a dummy 1x1
            # matrix with an empty mask in case the sequence list is empty.
            m = np.zeros((1 if max_length is None else max_length, 1),
                         dtype=dtype)
            mask = np.zeros_like(m, dtype=np.bool)
            return m, mask

        length = max((len(x[0]) for x in encoded_sequences))
        length = length if max_length is None else min(length, max_length)

        m = np.zeros((length, len(encoded_sequences)), dtype=dtype)
        mask = np.zeros_like(m, dtype=np.bool)

        all_unknowns = []
        for i,pair in enumerate(encoded_sequences):
            encoded, unknowns = pair
            if unknowns is not None:
                unk_offset = len(all_unknowns)
                encoded = [idx - unk_offset if idx < 0 else idx
                           for idx in encoded]
                all_unknowns.extend(unknowns)

            if pad_right:
                m[:len(encoded),i] = encoded
                mask[:len(encoded),i] = 1
            else:
                m[-len(encoded):,i] = encoded
                mask[-len(encoded):,i] = 1

        if self.sub_encoder is None:
            return m, mask
        else:
            char, char_mask = self.sub_encoder.pad_sequences(all_unknowns)
            return m, mask, char, char_mask

    def decode_padded(self, m, mask, char=None, char_mask=None):
        if char is not None:
            unknowns = list(map(
                ''.join, self.sub_encoder.decode_padded(char, char_mask)))
        start = self.index.get('<S>')
        stop = self.index.get('</S>')
        return [[unknowns[-x-1] if x < 0 else self.vocab[x]
                 for x,b in zip(row,row_mask)
                 if bool(b) and x not in (start, stop)]
                for row,row_mask in zip(m.T,mask.T)]

    def split_unk_outputs(self, outputs, outputs_mask):
        # Compute separate mask for character level (UNK) words
        # (with symbol < 0).
        charlevel_mask = outputs_mask * T.lt(outputs, 0)
        charlevel_indices = T.nonzero(charlevel_mask.T)
        # shortlisted words directly in word level decoder,
        # but char level replaced with unk
        unked_outputs = (1 - charlevel_mask) * outputs
        unked_outputs += charlevel_mask * T.as_tensor(
            self.index['<UNK>'])
        return unked_outputs, charlevel_indices


class TwoThresholdTextEncoder(TextEncoder):
    def __init__(self,
                 max_vocab=None,
                 vocab=None,
                 sequences=None,
                 sub_encoder=None,
                 special=('<S>', '</S>', '<UNK>'),
                 low_thresh=None):
        super().__init__(max_vocab=max_vocab, vocab=vocab,
                         sequences=sequences, sub_encoder=sub_encoder,
                         special=special)
        self.low_thresh = min(low_thresh + len(special), len(self))
        assert self.sub_encoder is not None

    def __str__(self):
        return 'TwoThresholdTextEncoder(%d, %d, %s)' % (
            self.low_thresh, len(self), str(self.sub_encoder))

    def encode_sequence(self, sequence, max_length=None, dtype=np.int32):
        """
        returns:
            an Encoded namedtuple, with the following fields:
            sequence --
                numpy array of symbol indices.
                Negative values index into the unknowns list,
                while positive values index into the encoder lexicon.
            unknowns --
                list of unknown tokens as Encoded(seq, None) tuples,
                or None if no subencoder.
        """
        start = (self.index['<S>'],) if '<S>' in self.index else ()
        stop = (self.index['</S>'],) if '</S>' in self.index else ()
        unk = self.index.get('<UNK>')
        unknowns = []
        def encode_item(x):
            idx = self.index.get(x)
            if idx is not None and idx <= self.low_thresh:
                low_idx = idx
            else:
                low_idx = None
            if low_idx is None:
                encoded_unk = self.sub_encoder.encode_sequence(x)
                unknowns.append(encoded_unk)
                low_idx = -len(unknowns)
                if idx is None:
                    idx = low_idx
            return idx
        encoded = tuple(idx for idx in list(map(encode_item, sequence))
                        if idx is not None)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            out = start + encoded + stop
        else:
            out = start + encoded[:max_length-(len(start)+len(stop))] + stop
        return Encoded(np.asarray(out, dtype=dtype), unknowns)

    def split_unk_outputs(self, outputs, outputs_mask):
        # Compute separate mask for character level (UNK) words
        # (with symbol < 0 or > self.low_thresh).
        charlevel_mask = outputs_mask * T.lt(outputs, 0)
        # lower threshold used for indexing tensor for charlevel
        lthr = T.as_tensor(self.low_thresh)
        low_charlevel_mask = charlevel_mask + (outputs_mask * T.gt(outputs, lthr))
        low_charlevel_indices = T.nonzero(low_charlevel_mask.T)
        # higher threshold used for
        # shortlisted words directly in word level decoder,
        # but char level replaced with unk
        unked_outputs = (1 - charlevel_mask) * outputs
        unked_outputs += charlevel_mask * T.as_tensor(
            self.index['<UNK>'])
        return unked_outputs, low_charlevel_indices
