import collections
import itertools
import numpy as np
import random
import cPickle as pickle

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
                 src_max_len=600,
                 trg_max_len=600,
                 max_lines_per_shard=1000000,
                 min_lines_per_group=128,
                 min_saved_padding=2048,
                 file_fmt='{corpus}.shard{shard:03}.group{group:03}.pickle',
                 vocab_file_fmt='{corpus}.vocab.pickle'):
        self.corpus = corpus
        # callables, yielding tokenized lines
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        # single new-style encoder per side
        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
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
        # first pass
        for (i, (src, trg)) in enumerate(safe_zip(self.src_lines(),
                                                  self.trg_lines())):
            src_len = len(src.surface)
            trg_len = len(trg.surface)
            # filter out too long lines
            if src_len > self.src_max_len:
                continue
            if trg_len > self.trg_max_len:
                continue
            # total line count => shard sizes/num
            # length distribution => thresholds for padding groups
            self.line_lens.append(LineLengths(i, src_len, trg_len))
            # token counts => vocabulary index (encoder)
            self.src_encoder.count(src)
            self.trg_encoder.count(trg)
        # preassign sentences to shards by random draw without replacement
        self.n_shards = int(np.ceil(len(self.line_lens) / self.max_lines_per_shard))
        lines_per_shard = int(np.ceil(len(self.line_lens) / self.n_shards))
        self.shard_indices = [j for i in range(self.n_shards) for j in [i] * lines_per_shard]
        random.shuffle(self.shard_indices)
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
        if min(mid, len(lens) - mid) < self.min_lines_per_group:
            # too small group
            split_ok = False
        if split_ok:
            threshold = lens[mid]
            left = self.choose_thresholds(lines[:mid], not trg)
            right = self.choose_thresholds(lines[mid:], not trg)
            return SplitNode(threshold, left, right, trg)
        else:
            leaf = LeafNode(self.n_groups)
            self.n_groups += 1
            return leaf


    def encode(self):
        # second pass
        for shard in range(self.n_shards):
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
                     len(src_enc.surface.unknowns),
                     len(trg_enc.surface.unknowns)))
                encoded[group].append((src_enc, trg_enc))
            # pad and concatenate groups
            for (group, pairs) in enumerate(encoded):
                srcs, trgs = zip(*pairs)
                padded_src = self.src_encoder.pad_sequences(srcs)
                padded_trg = self.trg_encoder.pad_sequences(trgs)
                # save encoded and padded data
                group_file_name = self.file_fmt.format(
                    corpus=self.corpus,
                    shard=shard,
                    group=group)
                with open(group_file_name, 'w') as fobj:
                    pickle.dump([padded_src, padded_trg],
                                protocol=pickle.HIGHEST_PROTOCOL)
        # save encoders and stats
        self.line_statistics = dict(itertools.groupby(
            sorted(self.line_statistics, key=lambda x: x.shard),
            lambda x: x.shard))
        with open(self.vocab_file_fmt.format(corpus=self.corpus), 'w') as fobj:
            pickle.dump(
                [self.corpus, self.file_fmt, self.src_encoder, self.trg_encoder,
                 self.line_statistics, self.n_groups],
                protocol=pickle.HIGHEST_PROTOCOL)


def iterate_sharded_data(vocab_file, budget_func):
    corpus, file_fmt, src_encoder, trg_encoder, line_statistics, n_groups = \
        pickle.loads(vocab_file)
    while True:
        shards = line_statistics.keys()
        random.shuffle(shards)
        for shard in shards:
            # load in the data of the shard
            groups = [pickle.load(file_fmt.format(
                                  corpus=corpus,
                                  shard=shard,
                                  group=group))
                      for group in range(n_groups)]
            # randomize indices belonging to shard
            lines = list(line_statistics[shard])
            random.shuffle(lines)
            # build minibatches group-wise
            minibatches = [list() for _ in range(n_groups)]
            for line in lines:
                if budget_func(minibatches[line.group], line):
                    # group would become overfull according to budget
                    # instantiate mb (indexing into full padding group tensors)
                    indices = np.array([line.idx_in_group
                                        for line in minibatches[line.group]])
                    src = [m[:, indices] for m in groups[line.group][0]]
                    trg = [m[:, indices] for m in groups[line.group][1]]
                    # yield it and start a new one
                    yield (src, trg)
                    groups[line.group] = []
                # otherwise extend the minibatch
                minibatches[line.group].append(line)
            for (mb, group) in zip(minibatches, groups):
                # yield the unfinished minibatches
                indices = np.array([line.idx_in_group for line in mb])
                src = [m[:, indices] for m in group[0]]
                trg = [m[:, indices] for m in group[1]]
                # yield it and start a new one
                yield (src, trg)


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

    parser.add_argument('--source-format', type=str,
            choices=('char', 'hybrid', 'finnpos'),
            help='type of preprocessing for source text')
    parser.add_argument('--target-format', type=str,
            choices=('char', 'hybrid', 'finnpos'),
            help='type of preprocessing for target text')
    parser.add_argument('--max-source-length', type=int,
            metavar='N',
            help='maximum length of source sentence (in tokens)')
    parser.add_argument('--max-target-length', type=int,
            metavar='N',
            help='maximum length of target sentence (in tokens)')
    parser.add_argument('--max-source-word-length', type=int,
            metavar='N',
            help='maximum length of source word in chars')
    parser.add_argument('--max-target-word-length', type=int,
            metavar='N',
            help='maximum length of target word in chars')
    parser.add_argument('--min-char-count', type=int,
            metavar='N',
            help='drop all characters with count < N in training data')
    parser.add_argument('--source-vocabulary', type=int, default=10000,
            metavar='N',
            help='maximum size of source word-level vocabulary')
    parser.add_argument('--target-vocabulary', type=int, default=10000,
            metavar='N',
            help='maximum size of target word-level vocabulary')
    parser.add_argument('--lemma-vocabulary', type=int, default=10000,
            metavar='N',
            help='size of lemma vocabulary for aux task')
    parser.add_argument('--hybrid-vocabulary-overlap', type=int, default=0,
            metavar='N',
            help='overlap vocabularies of word and character-level decoder '
            'during training with all except this number of least frequent words')
    parser.add_argument('--max-lines-per-shard', type=int, default=1000000,
            metavar='N',
            help='Approximate size of the shards, in sentences.')
    parser.add_argument('--min-lines-per-group', type=int, default=128,
            metavar='N',
            help='Do not split a padding group if the result would contain '
            'less than this number of sentences.')
    parser.add_argument('--min-saved-padding', type=int, default=2048,
            metavar='N',
            help='Do not split a padding group if the result would save '
            'less than this number of timesteps of wasted padding.')
    parser.add_argument('--filenames', type=str,
                default='{corpus}.shard{shard:03}.group{group:03}.pickle',
                help='Template string for sharded file names. '
                'Use {corpus}, {shard} and {group}.')
    parser.add_argument('--filenames', type=str,
                default='{corpus}.vocab.pickle',
                help='Template string for vocabulary file name. '
                'Use {corpus}.')

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
        max_lines_per_shard=args.max_lines_per_shard,
        min_lines_per_group=args.min_lines_per_group,
        min_saved_padding=args.min_saved_padding,
        file_fmt=args.file_fmt,
        vocab_file_fmt=args.vocab_file_fmt)
    sharded.prepare_data()
