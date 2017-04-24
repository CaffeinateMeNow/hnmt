import collections
import itertools
import numpy as np
import random
import pickle

from finnpos import *
from utils import *

LineLengths = collections.namedtuple('LineLengths',
    ['idx', 'src_len', 'trg_len'])
LineStatistics = collections.namedtuple('LineStatistics',
    ['idx', 'shard', 'group', 'idx_in_group', 'src_len', 'trg_len', 'src_unks', 'trg_unks'])

class SplitNode(object):
    def __init__(self, threshold, left, right, trg=False):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.trg = trg

    def decide(self, linelens):
        val = linelens.trg_len if self.trg else linelens.src_len
        if val < self.threshold:
            return self.left.decide(linelens)
        else:
            return self.right.decide(linelens)

    def __repr__(self):
        return 'S({}:{}, {}, {})'.format(
            'trg' if self.trg else 'src',
            self.threshold,
            repr(self.left),
            repr(self.right))

class LeafNode(object):
    def __init__(self, group_idx):
        self.group_idx = group_idx

    def decide(self, linelens):
        return self.group_idx

    def __repr__(self):
        return 'L({})'.format(self.group_idx)


class ShardedData(object):
    def __init__(self,
                 corpus,
                 src_lines,
                 trg_lines,
                 src_encoder,
                 trg_encoder,
                 src_format,
                 trg_format,
                 src_max_len=600,
                 trg_max_len=600,
                 src_max_word_len=50,
                 trg_max_word_len=50,
                 max_lines_per_shard=500000,
                 min_lines_per_group=128,
                 min_saved_padding=4096,
                 file_fmt='{corpus}.shard{shard:03}.group{group:03}.pickle',
                 vocab_file_fmt='{corpus}.vocab.pickle'):
        self.corpus = corpus
        # callables, yielding tokenized lines
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        # single new-style encoder per side
        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.src_format = src_format
        self.trg_format = trg_format
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.src_max_word_len = src_max_word_len
        self.trg_max_word_len = trg_max_word_len
        self.min_lines_per_group = min_lines_per_group
        self.max_lines_per_shard = max_lines_per_shard
        self.min_saved_padding = min_saved_padding
        self.file_fmt = file_fmt
        self.vocab_file_fmt = vocab_file_fmt
        # first LineLengths
        self.line_lens = []
        # later LineStatistics
        self.line_statistics = []
        self.n_shards = None
        self.shard_indices = None
        # decision tree
        self.padding_group_thresholds = None
        self.n_groups = 0

    def prepare_data(self):
        self.collect_statistics()
        self.encode()

    def collect_statistics(self):
        print('*** First pass: collecting statistics')
        for (i, (src, trg)) in enumerate(safe_zip(self.src_lines(),
                                                  self.trg_lines())):
            src_len = len(src.surface)
            trg_len = len(trg.surface)
            # filter out too long lines
            if src_len > self.src_max_len:
                continue
            if trg_len > self.trg_max_len:
                continue
            # filter out lines with too long words
            if max(len(word) for word in src.surface) > self.src_max_word_len:
                continue
            if max(len(word) for word in trg.surface) > self.trg_max_word_len:
                continue
            # total line count => shard sizes/num
            # length distribution => thresholds for padding groups
            self.line_lens.append(LineLengths(i, src_len, trg_len))
            # token counts => vocabulary index (encoder)
            self.src_encoder.count(src)
            self.trg_encoder.count(trg)
        print('*** pre-assigning sentences to shards')
        # preassign sentences to shards by random draw without replacement
        self.n_shards = int(np.ceil(len(self.line_lens) / self.max_lines_per_shard))
        lines_per_shard = int(np.ceil(len(self.line_lens) / self.n_shards))
        self.shard_indices = [j for i in range(self.n_shards) for j in [i] * lines_per_shard]
        random.shuffle(self.shard_indices)
        print('*** choosing thresholds')
        # choose thresholds for padding groups
        self.padding_group_thresholds = self.choose_thresholds(self.line_lens, trg=False)
        # decide vocabularies for encoders
        self.src_encoder.done()
        self.trg_encoder.done()

    def choose_thresholds(self, lines, trg):
        if trg:
            lenfunc = lambda x: x.trg_len
        else:
            lenfunc = lambda x: x.src_len
        # sort lengths
        lines = sorted(lines, key=lenfunc)
        lens = np.array([lenfunc(x) for x in lines])
        # select threshold to maximize reduced padding waste
        waste = lens[-1] - lens
        savings = np.arange(len(lens)) * waste
        mid = np.argmax(savings)
        # criteria for split
        split_ok = True
        if savings[mid] < self.min_saved_padding:
            # savings are not big enough
            split_ok = False
        # size limit is average over shards, doesn't guarantee limit
        threshold = lens[mid]
        below = np.sum(lens < threshold)
        above = np.sum(lens > threshold)
        if min(below, above) < self.min_lines_per_group * self.n_shards:
            # too small group
            split_ok = False
        if split_ok:
            print('splitting {} at {}'.format('target' if trg else 'source', threshold))
            left = self.choose_thresholds(lines[:mid], not trg)
            right = self.choose_thresholds(lines[mid:], not trg)
            return SplitNode(threshold, left, right, trg)
        else:
            leaf = LeafNode(self.n_groups)
            self.n_groups += 1
            return leaf


    def encode(self):
        print('*** Second pass: shard, encode, pad and save data')
        for shard in range(self.n_shards):
            print('** shard: {}'.format(shard))
            lines_in_shard = {line.idx: line 
                              for (line, sid)
                              in zip(self.line_lens, self.shard_indices)
                              if sid == shard}
            encoded = [list() for _ in range(self.n_groups)]
            # one pass over the data per shard
            for (i, (src, trg)) in enumerate(safe_zip(self.src_lines(),
                                                      self.trg_lines())):
                line = lines_in_shard.get(i, None)
                if line is None:
                    # drops too long and lines belonging to other shards
                    continue
                # choose padding group by lengths
                group = self.padding_group_thresholds.decide(line)
                # encode
                src_enc = self.src_encoder.encode_sequence(src)
                trg_enc = self.trg_encoder.encode_sequence(trg)
                # also track number of unks
                self.line_statistics.append(LineStatistics
                    (line.idx, shard, group, len(encoded[group]),
                     line.src_len, line.trg_len,
                     len(src_enc.surface.unknown),
                     len(trg_enc.surface.unknown)))
                encoded[group].append((src_enc, trg_enc))
            # pad and concatenate groups
            for (group, pairs) in enumerate(encoded):
                group_file_name = self.file_fmt.format(
                    corpus=self.corpus,
                    shard=shard,
                    group=group)
                if len(pairs) == 0:
                    print('shard {} group {} was empty'.format(shard, group))
                    with open(group_file_name, 'wb') as fobj:
                        pickle.dump([],
                                    fobj, protocol=pickle.HIGHEST_PROTOCOL)
                    continue
                srcs, trgs = zip(*pairs)
                padded_src = self.src_encoder.pad_sequences(srcs)
                del srcs
                padded_trg = self.trg_encoder.pad_sequences(trgs)
                del trgs
                # save encoded and padded data
                n_src_unks = None
                n_trg_unks = None
                if len(padded_src) > 2:
                    n_src_unks = sum(len(x) for x in padded_src[2])
                if len(padded_trg) > 2:
                    n_trg_unks = sum(len(x) for x in padded_trg[2])
                print('saving shard {} group {} with len ({}, {}) unks ({}, {}) in file {}'.format(
                    shard, group, padded_src[0].shape, padded_trg[0].shape,
                    n_src_unks, n_trg_unks, group_file_name))
                with open(group_file_name, 'wb') as fobj:
                    pickle.dump([padded_src, padded_trg],
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                del padded_src
                del padded_trg
            del encoded
        # save encoders and stats
        self.line_statistics = dict(
            (shard, list(lines)) for (shard, lines) in 
            itertools.groupby(
                sorted(self.line_statistics, key=lambda x: x.shard),
                lambda x: x.shard))
        config = {
            'corpus_name': self.corpus,
            'shard_file_fmt': self.file_fmt,
            'shard_n_groups': self.n_groups,
            'max_source_length': self.src_max_len,
            'max_target_length': self.trg_max_len,
            'max_source_word_length': self.src_max_word_len,
            'max_target_word_length': self.trg_max_word_len,
            'source_tokenizer': 'char' if self.src_format == 'char' else 'space',
            'target_tokenizer': 'char' if self.trg_format == 'char' else 'space',
            'src_encoder': self.src_encoder,
            'trg_encoder': self.trg_encoder,
            'src_format': self.src_format,
            'trg_format': self.trg_format,
        }
        with open(self.vocab_file_fmt.format(corpus=self.corpus), 'wb') as fobj:
            pickle.dump(
                [config, self.line_statistics],
                fobj, protocol=pickle.HIGHEST_PROTOCOL)

def instantiate_mb(group, indices, encoder):
    # start with placeholders for word-level
    out = [None, None]
    unk_offsets = np.zeros((1, len(indices)), dtype=np.int32)

    if len(group) >= 3:
        # character-level is padded just-in-time
        flat = []
        current_unk_offset = 0
        for (i, idx) in enumerate(indices):
            flat.extend(group[2][idx])
            unk_offsets[0,i] = current_unk_offset
            current_unk_offset += len(group[2][idx])
        char, char_mask = encoder.sub_encoder.pad_sequences(flat)
        out.extend((char, char_mask))

    # word-level after determining unk_offsets
    extracted = np.array(group[0][:, indices], dtype=np.int32)
    unk_mask = (extracted < 0)
    extracted += unk_mask * -unk_offsets
    out[0] = extracted
    # word-level is simply indexed
    out[1] = group[1][:, indices]

    if len(group) >= 4:
        # all fields in aux should be word-level
        out.extend(aux_m[:, indices]
                   for (field, aux_m) in zip(group[3]._fields, group[3])
                   if field != 'surface')
    return out

def iterate_sharded_data(config, shard_line_stats, budget_func):
    epoch = 0   # FIXME: integrate with old epoch counting
    while True:
        print('Starting epoch {}'.format(epoch))
        shards = list(shard_line_stats.keys())
        random.shuffle(shards)
        for shard in shards:
            print('Loading shard {}...'.format(shard))
            # load in the data of the shard
            groups = []
            for group in range(config['shard_n_groups']):
                group_file_name = config['shard_file_fmt'].format(
                    corpus=config['corpus_name'],
                    shard=shard,
                    group=group)
                with open(group_file_name, 'rb') as fobj:
                    groups.append(pickle.load(fobj))
            # randomize indices belonging to shard
            lines = list(shard_line_stats[shard])
            random.shuffle(lines)
            # build minibatches group-wise
            minibatches = [list() for _ in range(config['shard_n_groups'])]
            for line in lines:
                if budget_func(minibatches[line.group], line):
                    # group would become overfull according to budget
                    print('yielding mb from shard {} group {}'.format(shard, line.group))
                    # instantiate mb (indexing into full padding group tensors)
                    indices = np.array([line.idx_in_group
                                        for line in minibatches[line.group]])
                    src = instantiate_mb(
                        groups[line.group][0], indices, config['src_encoder'])
                    trg = instantiate_mb(
                        groups[line.group][1], indices, config['trg_encoder'])
                    #print('src shapes (after indexing): ', [m.shape for m in src])
                    #print('trg shapes (after indexing): ', [m.shape for m in trg])
                    # yield it and start a new one
                    yield (src, trg)
                    minibatches[line.group] = []
                # otherwise extend the minibatch
                minibatches[line.group].append(line)
            for (mb, group) in zip(minibatches, groups):
                # yield the unfinished minibatches
                indices = np.array([line.idx_in_group for line in mb])
                src = instantiate_mb(group[0], indices, config['src_encoder'])
                trg = instantiate_mb(group[1], indices, config['trg_encoder'])
                # yield it and start a new one
                yield (src, trg)
        epoch += 1


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Prepare data for HNMT')

    parser.add_argument('corpus', type=str,
        metavar='corpus',
        help='name of corpus')
    parser.add_argument('source', type=str,
        metavar='FILE',
        help='name of source language file')
    parser.add_argument('target', type=str,
        metavar='FILE',
        help='name of target language file')

    parser.add_argument('--source-format', type=str, default='hybrid',
            choices=('char', 'hybrid', 'finnpos'),
            help='type of preprocessing for source text '
            'default "%(default)s"')
    parser.add_argument('--target-format', type=str, default='hybrid',
            choices=('char', 'hybrid', 'finnpos'),
            help='type of preprocessing for target text '
            'default "%(default)s"')
    parser.add_argument('--max-source-length', type=int, default=600,
            metavar='N',
            help='maximum length of source sentence (in tokens) '
            'default "%(default)s"')
    parser.add_argument('--max-target-length', type=int, default=600,
            metavar='N',
            help='maximum length of target sentence (in tokens) '
            'default "%(default)s"')
    parser.add_argument('--max-source-word-length', type=int, default=50,
            metavar='N',
            help='maximum length of source word in chars '
            'default "%(default)s"')
    parser.add_argument('--max-target-word-length', type=int, default=50,
            metavar='N',
            help='maximum length of target word in chars '
            'default "%(default)s"')
    parser.add_argument('--min-char-count', type=int, default=2,
            metavar='N',
            help='drop all characters with count < N in training data '
            'default "%(default)s"')
    parser.add_argument('--source-vocabulary', type=int, default=10000,
            metavar='N',
            help='maximum size of source word-level vocabulary '
            'default "%(default)s"')
    parser.add_argument('--target-vocabulary', type=int, default=10000,
            metavar='N',
            help='maximum size of target word-level vocabulary '
            'default "%(default)s"')
    parser.add_argument('--lemma-vocabulary', type=int, default=10000,
            metavar='N',
            help='size of lemma vocabulary for aux task '
            'default "%(default)s"')
    parser.add_argument('--hybrid-vocabulary-overlap', type=int, default=0,
            metavar='N',
            help='overlap vocabularies of word and character-level decoder '
            'during training with all except this number of least frequent words '
            'default "%(default)s"')
    parser.add_argument('--max-lines-per-shard', type=int, default=1000000,
            metavar='N',
            help='Approximate size of the shards, in sentences. '
            'default "%(default)s"')
    parser.add_argument('--min-lines-per-group', type=int, default=128,
            metavar='N',
            help='Do not split a padding group if the result would contain '
            'less than this number of sentences. '
            'default "%(default)s"')
    parser.add_argument('--min-saved-padding', type=int, default=2048,
            metavar='N',
            help='Do not split a padding group if the result would save '
            'less than this number of timesteps of wasted padding. '
            'default "%(default)s"')
    parser.add_argument('--shard-group-filenames', type=str,
            default='{corpus}.shard{shard:03}.group{group:03}.pickle',
            help='Template string for sharded file names. '
            'Use {corpus}, {shard} and {group}. '
            'default "%(default)s"')
    parser.add_argument('--vocab-filename', type=str,
            default='{corpus}.vocab.pickle',
            help='Template string for vocabulary file name. '
            'Use {corpus}. default "%(default)s"')

    args = parser.parse_args()

    # type of encoders depends on format
    if args.source_format == 'char':
        src_reader = tokenize(args.source, 'char', False)
        # TextEncoder from all chars
        src_encoder = TextEncoder(
            #sequences=[token for sent in src_reader() for token in sent],
            min_count=args.min_char_count)
    elif args.source_format == 'hybrid':
        src_reader = tokenize(args.source, 'space', False)
        src_char_encoder = TextEncoder(
            #sequences=[token for sent in src_reader() for token in sent],
            min_count=args.min_char_count,
            special=())
        src_encoder = TextEncoder(
            #sequences=src_reader(),
            max_vocab=args.source_vocabulary,
            sub_encoder=src_char_encoder)
    elif args.source_format == 'finnpos':
        src_reader = finnpos_reader(args.source)
        # FinnposEncoder does the lot
        src_encoder = FinnposEncoder(
            #sequences=src_reader(),
            max_vocab=args.source_vocabulary,
            max_lemma_vocab=args.lemma_vocabulary)

    if args.target_format == 'char':
        trg_reader = tokenize(args.target, 'char', False)
        # TextEncoder from all chars
        trg_encoder = TextEncoder(
            #sequences=[token for sent in trg_reader() for token in sent],
            min_count=args.min_char_count)
    elif args.target_format == 'hybrid':
        trg_reader = tokenize(args.target, 'space', False)
        trg_char_encoder = TextEncoder(
            #sequences=[token for sent in trg_reader() for token in sent],
            min_count=args.min_char_count,
            special=())
        if args.hybrid_vocabulary_overlap <= 0:
            trg_encoder = TextEncoder(
                #sequences=trg_reader(),
                max_vocab=args.target_vocabulary,
                sub_encoder=trg_char_encoder)
        else:
            trg_encoder = TwoThresholdTextEncoder(
                #sequences=trg_reader(),
                max_vocab=args.target_vocabulary,
                overlap=args.hybrid_vocabulary_overlap,
                sub_encoder=trg_char_encoder)
    elif args.target_format == 'finnpos':
        trg_reader = finnpos_reader(args.target)
        # FinnposEncoder does the lot
        trg_encoder = FinnposEncoder(
            #sequences=trg_reader(),
            max_vocab=args.target_vocabulary,
            max_lemma_vocab=args.lemma_vocabulary,
            overlap=args.hybrid_vocabulary_overlap)

    sharded = ShardedData(
        args.corpus,
        src_reader,
        trg_reader,
        src_encoder,
        trg_encoder,
        src_format=args.source_format,
        trg_format=args.target_format,
        src_max_len=args.max_source_length,
        trg_max_len=args.max_target_length,
        src_max_word_len=args.max_source_word_length,
        trg_max_word_len=args.max_target_word_length,
        max_lines_per_shard=args.max_lines_per_shard,
        min_lines_per_group=args.min_lines_per_group,
        min_saved_padding=args.min_saved_padding,
        file_fmt=args.shard_group_filenames,
        vocab_file_fmt=args.vocab_filename)
    sharded.prepare_data()

if __name__ == '__main__':
    main()
