import dataclasses
from typing import List, Optional, Tuple, Union

import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
# import tokenizations
import torch

from benepar import ptb_unescape
from benepar.parse_base import BaseInputExample
from benepar.decode_chart import get_labeled_spans
import transliterate
import random

from rules_dist import silver_trees_js_correlation

@dataclasses.dataclass
class ParsingExample(BaseInputExample):
    """A single parse tree and sentence."""

    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree] = None
    _pos: Optional[List[Tuple[str, str]]] = None

    def leaves(self):
        if self.tree is not None:
            return self.tree.leaves()
        elif self._pos is not None:
            return [word for word, tag in self._pos]
        else:
            return None

    def pos(self):
        if self.tree is not None:
            return self.tree.pos()
        else:
            return self._pos

    def without_gold_annotations(self):
        return dataclasses.replace(self, tree=None, _pos=self.pos())


class Treebank(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @property
    def trees(self):
        return [x.tree for x in self.examples]

    @property
    def sents(self):
        return [x.words for x in self.examples]

    @property
    def tagged_sents(self):
        return [x.pos() for x in self.examples]
    
    def remove(self, i):
        del self.examples[i]

    def filter_by_length(self, max_len):
        return Treebank([x for x in self.examples if len(x.leaves()) <= max_len])

    def without_gold_annotations(self):
        return Treebank([x.without_gold_annotations() for x in self.examples])
    
    def combine_two_bank(self, extra_bank):
        if extra_bank is not None:
            self.examples += extra_bank.examples


# def read_text(text_path):
#     sents = []
#     sent = []
#     end_of_multiword = 0
#     multiword_combined = ""
#     multiword_separate = []
#     multiword_sp_after = False
#     with open(text_path) as f:
#         for line in f:
#             if not line.strip() or line.startswith("#"):
#                 if sent:
#                     sents.append(([w for w, sp in sent], [sp for w, sp in sent]))
#                     sent = []
#                     assert end_of_multiword == 0
#                 continue
#             fields = line.split("\t", 2)
#             num_or_range = fields[0]
#             w = fields[1]

#             if "-" in num_or_range:
#                 end_of_multiword = int(num_or_range.split("-")[1])
#                 multiword_combined = w
#                 multiword_separate = []
#                 multiword_sp_after = "SpaceAfter=No" not in fields[-1]
#                 continue
#             elif int(num_or_range) <= end_of_multiword:
#                 multiword_separate.append(w)
#                 if int(num_or_range) == end_of_multiword:
#                     _, separate_to_combined = tokenizations.get_alignments(
#                         multiword_combined, multiword_separate
#                     )
#                     have_up_to = 0
#                     for i, char_idxs in enumerate(separate_to_combined):
#                         if i == len(multiword_separate) - 1:
#                             word = multiword_combined[have_up_to:]
#                             sent.append((word, multiword_sp_after))
#                         elif char_idxs:
#                             word = multiword_combined[have_up_to : max(char_idxs) + 1]
#                             sent.append((word, False))
#                             have_up_to = max(char_idxs) + 1
#                         else:
#                             sent.append(("", False))
#                     assert int(num_or_range) == len(sent)
#                     end_of_multiword = 0
#                     multiword_combined = ""
#                     multiword_separate = []
#                     multiword_sp_after = False
#                 continue
#             else:
#                 assert int(num_or_range) == len(sent) + 1
#                 sp = "SpaceAfter=No" not in fields[-1]
#                 sent.append((w, sp))
#     return sents


def load_trees(const_path, text_path=None, text_processing="default"):
    """Load a treebank.

    The standard tree format presents an abstracted view of the raw text, with the
    assumption that a tokenizer and other early stages of the NLP pipeline have already
    been run. These can include formatting changes like escaping certain characters
    (e.g. -LRB-) or transliteration (see e.g. the Arabic and Hebrew SPMRL datasets).
    Tokens are not always delimited by whitespace, and the raw whitespace in the source
    text is thrown away in the PTB tree format. Moreover, in some treebanks the leaves
    of the trees are lemmas/stems rather than word forms.

    All of this is a mismatch for pre-trained transformer models, which typically do
    their own tokenization starting with raw unicode strings. A mismatch compared to
    pre-training often doesn't affect performance if you just want to report F1 scores
    within the same treebank, but it raises some questions when it comes to releasing a
    parser for general use: (1) Must the parser be integrated with a tokenizer that
    matches the treebank convention? In fact, many modern NLP libraries like spaCy train
    on dependency data that doesn't necessarily use the same tokenization convention as
    constituency treebanks. (2) Can the parser's pre-trained model be merged with other
    pre-trained system components (via methods like multi-task learning or adapters), or
    must it remain its own system because of tokenization mismatches?

    This tree-loading function aims to build a path towards parsing from raw text by
    using the `text_path` argument to specify an auxiliary file that can be used to
    recover the original unicode string for the text. Parser layers above the
    pre-trained model may still use gold tokenization during training, but this will
    possibly help make the parser more robust to tokenization mismatches.

    On the other hand, some benchmarks involve evaluating with gold tokenization, and
    naively switching to using raw text degrades performance substantially. This can
    hopefully be addressed by making the parser layers on top of the pre-trained
    transformers handle tokenization more intelligently, but this is still a work in
    progress and the option remains to use the data from the tree files with minimal
    processing controlled by the `text_processing` argument to clean up some escaping or
    transliteration.

    Args:
        const_path: Path to the file with one tree per line.
        text_path: (optional) Path to a file that provides the correct spelling for all
            tokens (without any escaping, transliteration, or other mangling) and
            information about whether there is whitespace after each token. Files in the
            CoNLL-U format (https://universaldependencies.org/format.html) are accepted,
            but the parser also accepts similarly-formatted files with just three fields
            (ID, FORM, MISC) instead of the usual ten. Text is recovered from the FORM
            field and any "SpaceAfter=No" annotations in the MISC field.
        text_processing: Text processing to use if no text_path is specified:
            - 'default': undo PTB-style escape sequences and attempt to guess whitespace
                surrounding punctuation
            - 'arabic': guess that all tokens are separated by spaces
            - 'arabic-translit': undo Buckwalter transliteration and guess that all
                tokens are separated by spaces
            - 'chinese': keep all tokens unchanged (i.e. do not attempt to find any
                escape sequences), and assume no whitespace between tokens
            - 'hebrew': guess that all tokens are separated by spaces
            - 'hebrew-translit': undo transliteration (see Sima'an et al. 2002) and
                guess that all tokens are separated by spaces

    Returns:
        A list of ParsingExample objects, which have the following attributes:
            - `tree` is an instance of nltk.Tree
            - `words` is a list of strings
            - `space_after` is a list of booleans
    """
    reader = BracketParseCorpusReader("", [const_path])
    trees = reader.parsed_sents()

    if text_path is not None:
        # sents = read_text(text_path)
        assert False, ""
    elif text_processing in ("arabic-translit", "hebrew-translit"):
        translit = transliterate.TRANSLITERATIONS[
            text_processing.replace("-translit", "")
        ]
        sents = []
        for tree in trees:
            words = [translit(word) for word in tree.leaves()]
            sp_after = [True for _ in words]
            sents.append((words, sp_after))
    elif text_processing in ("arabic", "hebrew"):
        sents = []
        for tree in trees:
            words = tree.leaves()
            sp_after = [True for _ in words]
            sents.append((words, sp_after))
    elif text_processing == "chinese":
        sents = []
        for tree in trees:
            words = tree.leaves()
            sp_after = [False for _ in words]
            sents.append((words, sp_after))
    elif text_processing == "default":
        sents = []
        for tree in trees:
            words = ptb_unescape.ptb_unescape(tree.leaves())
            sp_after = ptb_unescape.guess_space_after(tree.leaves())
            sents.append((words, sp_after))
    else:
        raise ValueError(f"Bad value for text_processing: {text_processing}")

    assert len(trees) == len(sents)
    treebank = Treebank(
        [
            ParsingExample(tree=tree, words=words, space_after=space_after)
            for tree, (words, space_after) in zip(trees, sents)
        ]
    )
    for example in treebank:
        assert len(example.words) == len(example.leaves()), (
            "Constituency tree has a different number of tokens than the CONLL-U or "
            "other file used to specify reversible tokenization."
        )
    return treebank


def read_raw_text(text_path):
    snts = []
    for line in open(text_path, 'r', encoding='utf-8'):
        origin_words = line.strip().split()
        words = ptb_unescape.ptb_unescape(origin_words)
        sp_after = ptb_unescape.guess_space_after(origin_words)
        _pos = [(w, '') for w in origin_words]
        snts.append((words, sp_after, _pos))
        
    return snts


def load_raw_snts(path_in: Union[str, List]):
    if type(path_in) == str:
        path_list = [path_in]
    else:
        path_list = path_in

    snts = []
    for path in path_list:
        snts.extend(read_raw_text(path))
            
    raw_data = Treebank(
        [
            ParsingExample(tree=None, words=words, space_after=space_after, _pos=_pos) for words, space_after, _pos in snts
        ]
    )
    return raw_data


def choose_topK_GRs(snt_tree_score_dic, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, topK, remove_large_after_select: int, accord: int, slice=1000):
    # accord_dict = {0: "nonT", 1: "T", 2: "all" 3: "confidence", 4: "the best two combination"}
    print(f"0: T, 1: nonT, 2: all 3: confidence, 4: best two combination : {accord}")
    if accord <= 2:
        trees = [item['tree'] for item in snt_tree_score_dic]
        tree_js_correlation_scrs = silver_trees_js_correlation(trees, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, slice=slice, accord=accord)
        i = -1
        for item, (tree, jsscore) in zip(snt_tree_score_dic, tree_js_correlation_scrs):
            i += 1
            assert tree == item['tree']
            scale_num = len(get_labeled_spans(item["tree"]))
            item['score'] = [item['score'] / scale_num, jsscore / scale_num]
            item["id"] = i

        large_good_to_bad_list = sorted(snt_tree_score_dic, key=lambda x: x['score'][1], reverse=False)
    
    elif accord == 3:
        for i, item in enumerate(snt_tree_score_dic):
            item["score"] = item["score"] / len(get_labeled_spans(item["tree"]))
            item["id"] = i

        large_good_to_bad_list = sorted(snt_tree_score_dic, key=lambda x: x['score'], reverse=True)

    else:
        trees = [item['tree'] for item in snt_tree_score_dic]
        tree_js_correlation_scrs = silver_trees_js_correlation(trees, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, slice=slice, accord=2)
        i = -1
        for item, (tree, jsscore) in zip(snt_tree_score_dic, tree_js_correlation_scrs):
            i += 1
            assert tree == item['tree']
            scale_num = len(get_labeled_spans(item["tree"]))
            item['score'] = [item['score'] / scale_num, jsscore / scale_num]
            item["id"] = i
        all_large_good_to_bad_list = sorted(snt_tree_score_dic, key=lambda x: x["score"][1], reverse=False)[:10000]
        large_good_to_bad_list = sorted(all_large_good_to_bad_list, key=lambda x: x["score"][0], reverse=True)

    pseudo_trees, choosed_snts, ids = [], [], []
    assert remove_large_after_select >= topK, ""


    for item in large_good_to_bad_list[:remove_large_after_select]:
        if len(pseudo_trees) < topK:
            pseudo_trees.append(item["tree"])
        choosed_snts.append(item["sent"])
        ids.append(item["id"])
    
    return pseudo_trees, choosed_snts, ids

def load_as_trees(pseudo_trees, choosed_snts):
    snts = []
    for s in choosed_snts:
        words = ptb_unescape.ptb_unescape(s)
        sp_after = ptb_unescape.guess_space_after(s)
        snts.append((words, sp_after))

    treebank = Treebank(
        [
            ParsingExample(tree=tree, words=words, space_after=space_after)
            for tree, (words, space_after) in zip(pseudo_trees, snts)
        ]
    )
    return treebank

def remove_topK_snts(raw_snts: List[ParsingExample], choosed_snts, ids):
    id_snt = [(i, snt) for i, snt in zip(ids, choosed_snts)]
    del_id_snts = sorted(id_snt, key=lambda item: item[0], reverse=True)
    for i, snt in del_id_snts:
        assert snt == raw_snts[i].words
        raw_snts.remove(i)
    return raw_snts
