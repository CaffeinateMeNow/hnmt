from collections import Counter, namedtuple, OrderedDict
from text import Encoded, TextEncoder, TwoThresholdTextEncoder 
from utils import *

import numpy as np
import theano

MiniBatch = namedtuple('MiniBatch',
    ['src',             # (tokens, mask, chars, charmask)
     'tgt',             # (tokens, mask, chars, charmask)
     'aux',             # (aux0, aux1, ...) or None
    ])
# each of the fields contains a sequence with one element per token
# the main 'sequence' is the sequence form
# FIXME: split morph tags into multiple fields? lang-specific
#   Finnish: Number, Case, Person, Mood, Tense
Finnpos = namedtuple('Finnpos',
    ['surface', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])
Aux = namedtuple('Aux',
    ['surface', 'logf', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])

# julistan    _   julistaa    [POS=VERB]|[VOICE=ACT]|[MOOD=INDV]|[TENSE=PRESENT]|[PERS=SG1]   _   
        
def finnpos_helper(split):
    columns = list(zip(*split))
    surface = columns[0]
    lemmas = columns[2]
    pos = columns[3]
    num = columns[4]
    case = columns[5]
    pers = columns[6]
    mood = columns[7]
    tense = columns[8]
    return Finnpos(surface, lemmas, pos, num, case, pers, mood, tense)
            
def finnpos_reader(filename):
    def reader():
        raw = []    
        for line in open_file(filename):
            line = line.strip()
            if len(line) == 0:
                # empty lines indicate sentence breaks
                yield finnpos_helper(raw)
                raw = []
                continue
            if line[0] == '#':
                # comments start with hash
                continue
            raw.append(line.split('\t'))
    return reader


class LogFreqEncoder(object):
    def __init__(self,
                 vocab=None,
                 use_boundaries=True):
        self.use_boundaries = use_boundaries

        if vocab is not None:
            # vocab should be a Counter or similar
            self.vocab = vocab
        else:
            self.vocab = Counter()

    def count(self, sequence, raw=False):
        if not raw:
            sequence = sequence.lemma
        for lemma in sequence:
            self.vocab[lemma] += 1

    def done(self):
        # note that vocabulary is intentionally not truncated
        # this keeps actual frequencies of rare lemmas
        # but the softmax doesn't get any bigger
        mostfreq, _ = self.vocab.most_common(1)[0]
        self.max_val = self[mostfreq]

    def fields(self):
        return ('logf', len(self))

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

    def __repr__(self):
        return str(self)

    def __getitem__(self, token):
        return int(np.log(self.vocab[token] + 1))

    def __len__(self):
        return self.max_val + 1

    def encode_sequence(self, sequence, max_length=None, dtype=np.int32, raw=True):
        start = (0,) if self.use_boundaries else ()
        stop = (0,) if self.use_boundaries else ()
        encoded = tuple(self[symbol] for symbol in sequence)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            out = start + encoded + stop
        else:
            out = start + encoded[:max_length-(len(start)+len(stop))] + stop
        return Encoded(np.asarray(out, dtype=dtype), None)

    def pad_sequences(self, sequences,
                      max_length=None, pad_right=True, dtype=np.int32):
        if not sequences:
            # An empty matrix would mess up things, so create a dummy 1x1
            # matrix with an empty mask in case the sequence list is empty.
            m = np.zeros((1 if max_length is None else max_length, 1),
                         dtype=dtype)
            mask = np.zeros_like(m, dtype=np.bool)
            return m, mask
        encoded_sequences = [
                self.encode_sequence(sequence, max_length)
                for sequence in sequences]
        length = max(map(len, encoded_sequences))
        length = length if max_length is None else min(length, max_length)

        m = np.zeros((length, len(sequences)), dtype=dtype)
        mask = np.zeros_like(m, dtype=np.bool)

        for i,encoded in enumerate(encoded_sequences):
            if pad_right:
                m[:len(encoded),i] = encoded
                mask[:len(encoded),i] = 1
            else:
                m[-len(encoded):,i] = encoded
                mask[-len(encoded):,i] = 1

        return m, mask

    def decode_sentence(self, encoded, no_boundary=False):
        try:
            seq = encoded.sequence
        except AttributeError:
            seq = encoded
        if self.use_boundaries and not no_boundary:
            seq = seq[1:]
        #return [str(np.exp(x) - 1) for x in seq]
        return [str(x) for x in seq]

    def decode_padded(self, m, mask, raw=True):
        result = []
        for row, row_mask in zip(m.T, mask.T):
            decoded_row = []
            for x, b in zip(row, row_mask):
                decoded_row.append(str(x))
            result.append(decoded_row)
        return result


class FinnposEncoder(object):
    def __init__(self,
                 max_vocab=None,
                 max_lemma_vocab=None,
                 min_char_count=0,
                 overlap=0):
        self.max_vocab = max_vocab
        self.max_lemma_vocab = max_lemma_vocab
        self.overlap = overlap
        char_encoder = TextEncoder(min_count=min_char_count)
        self.sub_encoders = OrderedDict((
            ('surface', TwoThresholdTextEncoder(
                max_vocab=max_vocab,
                overlap=overlap,
                sub_encoder=char_encoder)),
            ('logf', LogFreqEncoder()),
            ('lemma', TextEncoder(max_vocab=self.max_lemma_vocab)),
            ('pos', TextEncoder()),
            ('num', TextEncoder()),
            ('case', TextEncoder()),
            ('pers', TextEncoder()),
            ('mood', TextEncoder()),
            ('tense', TextEncoder())))

    @property
    def sub_encoder(self):
        return self.sub_encoders['surface'].sub_encoder

    def count(self, tpl, raw=False):
        self.sub_encoders['surface'].count(tpl.surface, raw=True)
        self.sub_encoders['logf'].count(tpl.lemma, raw=True)
        self.sub_encoders['lemma'].count(tpl.lemma, raw=True)
        self.sub_encoders['pos'].count(tpl.pos, raw=True)
        self.sub_encoders['num'].count(tpl.num, raw=True)
        self.sub_encoders['case'].count(tpl.case, raw=True)
        self.sub_encoders['pers'].count(tpl.pers, raw=True)
        self.sub_encoders['mood'].count(tpl.mood, raw=True)
        self.sub_encoders['tense'].count(tpl.tense, raw=True)

    def done(self):
        for subenc in self.sub_encoders.values():
            subenc.done()

    def fields(self):
        return (
            ('surface', len(self.sub_encoders['surface'])),
            ('logf',    len(self.sub_encoders['logf'])),
            ('lemma',   len(self.sub_encoders['lemma'])),
            ('pos',     len(self.sub_encoders['pos'])),
            ('num',     len(self.sub_encoders['num'])),
            ('case',    len(self.sub_encoders['case'])),
            ('pers',    len(self.sub_encoders['pers'])),
            ('mood',    len(self.sub_encoders['mood'])),
            ('tense',   len(self.sub_encoders['tense'])),
            )

    def __str__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(key, sub) for (key, sub)
                                     in self.sub_encoders.items()))

    def __repr__(self):
        return str(self)

    def __getitem__(self, token):
        return self.sub_encoders['surface'][token]

    def __len__(self):
        # length of main vocabulary
        return len(self.sub_encoders['surface'])

    def encode_sequence(self, fields, max_length=None, dtype=np.int32):
        surf = self.sub_encoders['surface'].encode_sequence(fields.surface,
             max_length=max_length, dtype=dtype, raw=True)
        logf = self.sub_encoders['logf'].encode_sequence(fields.lemma,
             max_length=max_length, dtype=dtype, raw=True)
        lemma = self.sub_encoders['lemma'].encode_sequence(fields.lemma,
             max_length=max_length, dtype=dtype, raw=True)
        pos = self.sub_encoders['pos'].encode_sequence(fields.pos,
             max_length=max_length, dtype=dtype, raw=True)
        num = self.sub_encoders['num'].encode_sequence(fields.num,
             max_length=max_length, dtype=dtype, raw=True)
        case = self.sub_encoders['case'].encode_sequence(fields.case,
             max_length=max_length, dtype=dtype, raw=True)
        pers = self.sub_encoders['pers'].encode_sequence(fields.pers,
             max_length=max_length, dtype=dtype, raw=True)
        mood = self.sub_encoders['mood'].encode_sequence(fields.mood,
             max_length=max_length, dtype=dtype, raw=True)
        tense = self.sub_encoders['tense'].encode_sequence(fields.tense,
             max_length=max_length, dtype=dtype, raw=True)
        return Aux(surf, logf, lemma, pos, num, case, pers, mood, tense)

    def pad_sequences(self, encoded_sequences,
                      max_length=None, pad_right=True, dtype=np.int32, pad_chars=False):
        padded_surf = self.sub_encoders['surface'].pad_sequences(
            [x.surface for x in encoded_sequences],
            max_length=max_length, pad_right=pad_right, dtype=dtype)
        length = padded_surf[0].shape[0]
        n_batch = padded_surf[0].shape[1]
        out = Aux(*[
            np.zeros((length, n_batch), dtype=dtype)
            for _ in Aux._fields])

        for i,tpl in enumerate(encoded_sequences):
            for j,encoded in enumerate(tpl):
                encoded = encoded.sequence      # throw away empty unknowns
                if pad_right:
                    out[j][:len(encoded),i] = encoded
                else:
                    out[j][-len(encoded):,i] = encoded
        return padded_surf + (out,)

    def decode_sentence(self, encoded):
        return Aux(
            self.sub_encoders['surface'].decode_sentence(encoded.surface),
            self.sub_encoders['logf'].decode_sentence(encoded.lemma, no_boundary=True),
            self.sub_encoders['lemma'].decode_sentence(encoded.lemma),
            self.sub_encoders['pos'].decode_sentence(encoded.pos),
            self.sub_encoders['num'].decode_sentence(encoded.num),
            self.sub_encoders['case'].decode_sentence(encoded.case),
            self.sub_encoders['pers'].decode_sentence(encoded.pers),
            self.sub_encoders['mood'].decode_sentence(encoded.mood),
            self.sub_encoders['tense'].decode_sentence(encoded.tense),
        )

    def decode_padded(self, m, mask, chars, char_mask, *aux):
        _, lemma, pos, num, case, pers, mood, tense = aux
        result = (
            self.sub_encoders['surface'].decode_padded(m, mask, chars, char_mask, raw=True),
            self.sub_encoders['logf'].decode_padded(lemma, mask, raw=True),
            self.sub_encoders['lemma'].decode_padded(lemma, mask, raw=True),
            self.sub_encoders['pos'].decode_padded(pos, mask, raw=True),
            self.sub_encoders['num'].decode_padded(num, mask, raw=True),
            self.sub_encoders['case'].decode_padded(case, mask, raw=True),
            self.sub_encoders['pers'].decode_padded(pers, mask, raw=True),
            self.sub_encoders['mood'].decode_padded(mood, mask, raw=True),
            self.sub_encoders['tense'].decode_padded(tense, mask, raw=True),
        )
        return [Aux(*x) for x in zip(*result)]

    def split_unk_outputs(self, *args):
        return self.sub_encoders['surface'].split_unk_outputs(*args)
