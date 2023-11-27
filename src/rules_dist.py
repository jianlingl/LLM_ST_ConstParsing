from trees import Tree, load_tree_from_str
from statistic_GRs import is_prob_GRs, convert_cnt2prob, cal_JS, is_prob_GRs, get_rules_from_path
from tqdm import tqdm
import numpy as np
from collections import Counter


def map_rules_2array(*dict_list):
    unique_rules = set()
    assert len(*dict_list) >= 2
    dict_list = dict_list[0]
    for rules_dict in dict_list:
        unique_rules = unique_rules | set(list(rules_dict.keys()))
    A, B, C = [], [], []
    if len(dict_list) == 3:
        for rule in unique_rules:
            A.append(dict_list[0].get(rule, 0))
            B.append(dict_list[1].get(rule, 0))
            C.append(dict_list[2].get(rule, 0))
        return [np.array(A), np.array(B), np.array(C)]
    elif len(dict_list) == 2:
        for rule in unique_rules:
            A.append(dict_list[0].get(rule, 0))
            B.append(dict_list[1].get(rule, 0))
        return [np.array(A), np.array(B)]
    else:
        return None

def get_updated_prob_rules(tree, Src_cnt_nonT_GRs, Src_cnt_T_GRs):
    tree_str = tree.pformat(margin=1e100)
    tree = load_tree_from_str(tree_str, top_del=True)
    # get rules
    small_nonT_rules, small_T_rules, small_all_rules = {}, {}, {}
    tree.rules_seperate(small_nonT_rules, small_T_rules)
    # update all small rules with src big GRs
    small_nonT_rules = dict(Counter(small_nonT_rules) + Counter(Src_cnt_nonT_GRs))
    small_T_rules = dict(Counter(small_T_rules) + Counter(Src_cnt_T_GRs))
    small_all_rules = dict(Counter(small_nonT_rules) + Counter(small_T_rules))

    # prob the cnt
    assert not is_prob_GRs(Src_cnt_nonT_GRs) and not is_prob_GRs(small_nonT_rules), "should be not prob"
    tgt_nonT_rules = convert_cnt2prob(small_nonT_rules)
    tgt_T_rules = convert_cnt2prob(small_T_rules)
    tgt_all_rules = convert_cnt2prob(small_all_rules)
    return [tgt_nonT_rules, tgt_T_rules, tgt_all_rules]

def load_tree_prob_updated_rules(tree_list, src_GR_list, accord):
    # prob the src GRs with cnt
    Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs = src_GR_list
    assert accord <= 2, ""
    src_GRs = convert_cnt2prob(src_GR_list[accord].copy())

    if len(tree_list) == 1:
        tgt_rules = get_updated_prob_rules(tree_list[0], Src_cnt_nonT_GRs, Src_cnt_T_GRs)[accord]
    else:
        tgt_rules = [get_updated_prob_rules(tree, Src_cnt_nonT_GRs, Src_cnt_T_GRs)[accord] for tree in tree_list]

    return src_GRs, tgt_rules

def silver_trees_js_correlation(trees, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, slice=500, accord=2):
    def data_slice(trees, src_GR_list, slice):
        return [[trees[i: i+slice], src_GR_list] for i in range(0, len(trees), slice)]

    data_list = data_slice(trees, [Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs], slice)
    tree_js_correlation_scrs = []
    for data in tqdm(data_list):
        tree_js_correlation_scr = trees_js_correlation_serial(data, accord)
        tree_js_correlation_scrs.extend(tree_js_correlation_scr)
    return tree_js_correlation_scrs

def trees_js_correlation_serial(data_and_GRs, accord):
    assert len(data_and_GRs) == 2, ""
    trees, src_GR_list = data_and_GRs

    # cal by slice batch
    Src_GRs, tgt_rules =  load_tree_prob_updated_rules(trees, src_GR_list, accord)
    score_list = cal_JS(Src_GRs, tgt_rules)
    trees_GRs_correlation_scrs = [(tree, score) for tree, score in zip(trees, score_list)]
    return trees_GRs_correlation_scrs

def update_src_GRs(pseudo_treebank, Src_cnt_nonT_GRs: dict, Src_cnt_T_GRs: dict):
    assert not is_prob_GRs(Src_cnt_nonT_GRs) and not is_prob_GRs(Src_cnt_T_GRs)
    for tree in pseudo_treebank:
        tree_str = tree.pformat(margin=1e100)
        tree = load_tree_from_str(tree_str, top_del=True)
        tree.rules_seperate(Src_cnt_nonT_GRs, Src_cnt_T_GRs)
    
    src_cnt_all_GRs = Src_cnt_nonT_GRs.copy()
    src_cnt_all_GRs.update(Src_cnt_T_GRs)

    return Src_cnt_nonT_GRs, Src_cnt_T_GRs, src_cnt_all_GRs



# if __name__ == "__main__":
#     # for test
#     from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
#     reader = BracketParseCorpusReader("", [""])
#     tgt_trees = reader.parsed_sents()
#     src_nonT_rules, src_T_rules, src_all_rules = get_rules_from_path("", prob=False, sample=500)
#     trees_GRs_correlation_scrs = silver_trees_js_correlation(tgt_trees, src_nonT_rules, src_T_rules, src_all_rules, slice=1000, accord=0)
#     print(len(trees_GRs_correlation_scrs))