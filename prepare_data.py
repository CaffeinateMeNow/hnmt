import collections
import itertools
import numpy as np
import random

LineLengths = collections.namedtuple('LineLengths',
    ['idx', 'src_len', 'tgt_len'])
LineStatistics = collections.namedtuple('LineStatistics',
    ['idx', 'shard', 'src_len', 'tgt_len', 'n_unk', 'group'])

class SplitNode(object):
    def __init__(self, threshold, left, right, tgt=False):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.tgt = tgt

    def decide(self, linelens):
        val = linelens.tgt_len if self.tgt else linelens.src_len
        if val < self.threshold:
            return self.left.decide(linelens)
        else:
            return self.right.decide(linelens)

    def __repr__(self):
        return 'S({}:{}, {}, {})'.format(
            'tgt' if self.tgt else 'src',
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
                 src_lines,
                 tgt_lines,
                 src_encoders,
                 tgt_encoders,
                 src_max_len=600,
                 tgt_max_len=600,
                 min_lines_per_group=128,
                 max_lines_per_shard=1000000,
                 min_saved_padding=2048):
        # callables, yielding tokenized lines
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        # list of functions: [type] -> encoder
        self.src_encoders = src_encoders
        self.tgt_encoders = tgt_encoders
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.min_lines_per_group = min_lines_per_group
        self.max_lines_per_shard = max_lines_per_shard
        self.min_saved_padding = min_saved_padding
        # first LineLengths, later LineStatistics
        self.line_stats = []
        self.src_token_counts = [collections.Counter()
                                 for _ in self.src_encoders]
        self.tgt_token_counts = [collections.Counter()
                                 for _ in self.tgt_encoders]
        self.n_shards = None
        self.shard_indices = None
        # decision tree
        self.padding_group_thresholds = None
        self.current_group = 0

    def collect_statistics(self):
        # first pass
        for (i, (src, tgt)) in enumerate(safe_zip(self.src_lines(),
                                                  self.tgt_lines())):
            src_len = len(src.surface)
            tgt_len = len(tgt.surface)
            # filter out too long lines
            if src_len > self.src_max_len:
                continue
            if tgt_len > self.tgt_max_len:
                continue
            # total line count => shard sizes/num
            # length distribution => thresholds for padding groups
            self.line_stats.append(LineLengths(i, src_len, tgt_len))
            # token counts => vocabulary index (encoder)
            for (field, counter) in zip(src, self.src_token_counts):
                counter.update(field)
            for (field, counter) in zip(tgt, self.tgt_token_counts):
                counter.update(field)
        # preassign sentences to shards by random draw without replacement
        self.n_shards = int(np.ceil(len(self.line_stats) / self.max_lines_per_shard))
        lines_per_shard = int(np.ceil(len(self.line_stats) / self.n_shards))
        self.shard_indices = [j for i in range(self.n_shards) for j in [i] * lines_per_shard]
        random.shuffle(self.shard_indices)
        # choose thresholds for padding groups
        self.padding_group_thresholds = self.choose_thresholds(self.line_stats, tgt=False)

    def choose_thresholds(self, lines, tgt):
        if tgt:
            lenfunc = lambda x: x.tgt_len
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
            left = self.choose_thresholds(lines[:mid], not tgt)
            right = self.choose_thresholds(lines[mid:], not tgt)
            return SplitNode(threshold, left, right, tgt)
        else:
            leaf = LeafNode(self.current_group)
            self.current_group += 1
            return leaf


    def encode(self):
        #- second pass
        #    - one pass per shard
        #        - choose padding group by lengths
        #        - encode, pad and concatenate
        #        - also track number of unks
        pass

    def prepare_data(self):
        self.collect_statistics()
        self.encode()


def safe_zip(*iterables):
    iters = [iter(x) for x in iterables]
    sentinel = object()
    for (j, tpl) in enumerate(itertools.zip_longest(*iterables, fillvalue=sentinel)):
        for (i, val) in enumerate(tpl):
            if val is sentinel:
                raise ValueError('Column {} was too short. '
                    'Row {} (and later) missing.'.format(i, j))
        yield tpl
