from typing import List
from trees import Tree, load_tree_from_str, load_treebank
import numpy as np
import scipy.stats, random
from collections import Counter

def convert_cnt2prob(rules0: dict):
    rules = rules0.copy()
    cnt = sum(list(rules.values()))
    for gr, num in rules.items():
        rules[gr] = num / cnt
    return rules

def get_3rules(treebank: List[Tree], prob=True):
    no_terminal_rules, terminal_rules = {}, {}
    for t in treebank:
        t.rules_seperate(no_terminal_rules, terminal_rules)

    all_rules = no_terminal_rules.copy()
    all_rules.update(terminal_rules)

    if prob:
        no_terminal_rules = convert_cnt2prob(no_terminal_rules)
        terminal_rules = convert_cnt2prob(terminal_rules)
        all_rules = convert_cnt2prob(all_rules)

    return no_terminal_rules, terminal_rules, all_rules

def is_prob_GRs(GRs: dict):
    flag = None
    for _, v in sorted(GRs.items(), key=lambda x: x[1], reverse=True):
        if type(v) == int:
            flag = False
            break
        elif type(v) == float:
            flag = True
            break
        else:
            assert False
    return flag

def cal_JS(GRs0: dict, GRs1: dict or list):
    # note note GRs0==src GRs1==tgt->list
    def map_2array(keys, GRs0, GRs1):
        array0, array1 = [], []
        if type(GRs1) == list:
            for k in keys:
                array0.append(GRs0.get(k, 0.))
                array1.append([GRs.get(k, 0.) for GRs in GRs1])
            return np.repeat(np.expand_dims(np.array(array0), 0), len(GRs1), 0).transpose(1, 0), np.array(np.array(array1)) 
        else:
            for k in keys:
                array0.append(GRs0.get(k, 0.))
                array1.append(GRs1.get(k, 0.))
            return np.array(array0), np.array(array1)

    def JSdivergence(P, Q):
        M = (P + Q) / 2
        return 0.5 * scipy.stats.entropy(P, M) + 0.5 * scipy.stats.entropy(Q, M)

    if type(GRs1) == list:
        all_GRs1 = {}
        for GRs in GRs1:
            all_GRs1.update(GRs)
        keys = set(all_GRs1.keys()) | set(GRs0.keys())
    else:
        keys = set(GRs0.keys()) | set(GRs1.keys())
    np0, np1 = map_2array(keys, GRs0, GRs1)
    js_correlation = JSdivergence(np0, np1)
    return js_correlation.tolist()

def GRs_set_operate(GRs0, GRs1, operat: str):
    assert operat in ["&", "-"], "" #交集，差集
    # assert is_prob_GRs(GRs0) and is_prob_GRs(GRs1), ""

    def get_oprted_GRs(keys, *GRs, operat: str):
        if len(*GRs) == 2:
            assert operat in ["&"], ""
            src_GRs0, src_GRs1 = GRs[0]
            GRs0, GRs1 = {}, {}
            for k in keys:
                GRs0[k] = src_GRs0[k]
                GRs1[k] = src_GRs1[k]
            return GRs0, GRs1
        elif len(*GRs) == 1:
            assert operat in ["-"], ""
            src_GRs = GRs[0][0]
            tgt_GRs = {}
            for k in keys:
                tgt_GRs[k] = src_GRs[k]
            return tgt_GRs
        else:
            assert False

    if operat == "&":
        share_keys = set(GRs0.keys()) & set(GRs1.keys())
        share_GRs0, share_GRs1 = get_oprted_GRs(share_keys, [GRs0, GRs1], operat="&")
        return share_GRs0, share_GRs1 

    elif operat == "-":
        unique_keys_4GRs0 = set(GRs0.keys()) - set(GRs1.keys())
        unique_keys_4GRs1 = set(GRs1.keys()) - set(GRs0.keys())
        unique_GRs0, unique_GRs1 = get_oprted_GRs(unique_keys_4GRs0, [GRs0,], operat='-'), get_oprted_GRs(unique_keys_4GRs1, [GRs1], operat='-')
        return unique_GRs0, unique_GRs1
    else:
        assert False

def statistic_share_unique_prob_GRs(rules0: dict, rules1: dict, print_Top_share=False, print_Top_unique=False, print_share_info=False, demical=2):

    def print_top5_GRs(GRs: list):
        for i, item in enumerate(GRs):
            if i < 5:
                print(item)
            else:
                break

    def print_GRs_type_info(len_share):
        len_GR0, len_GR1 = len(rules0), len(rules1)
        src_share_prob, tgt_share_prob = round((len_share / len_GR0) * 100, demical), round((len_share / len_GR1) * 100, demical)
        print(f"There are {len_GR0} types of SRC GRs, {len_GR1} types of TGT GRs, SRC share prob: {src_share_prob}%, TGT share prob: {tgt_share_prob}%")

    print(f"set level GRs JS divergence: {round(cal_JS(rules0, rules1), demical)}")

    assert is_prob_GRs(rules0), "statistic should be prob type"
    # share_GR0, share_GR1 = GRs_set_operate(rules0, rules1, operat="&")
    # print(f"share GRs JS divergence: {round(cal_JS(share_GR0, share_GR1), demical)}")

    # if print_Top_share:
    #     rules_4prob_diff = {}
    #     for (k0, v0), (k1, v1) in zip(share_GR0.items(), share_GR1.items()):
    #         assert k0 == k1, ""
    #         rules_4prob_diff[k0] = (round(abs(v0-v1), 4), v0, v1)
    #     print("--------share with the biggest prob difference--------")
    #     print_top5_GRs(sorted(rules_4prob_diff.items(), key=lambda x: x[1][0], reverse=True))
    
    # unique_GR0, unique_GR1 = GRs_set_operate(rules0, rules1, operat="-")
    # print(f"unique GRs JS divergence: {round(cal_JS(unique_GR0, unique_GR1), demical)}")

    # if print_Top_unique:
    #     print("--------unique target GRs with the biggest prob--------")
    #     print_top5_GRs(sorted(unique_GR1.items(), key=lambda x: x[1], reverse=True))

    # if print_share_info:
    #     print_GRs_type_info(len(share_GR0))
    print("\n")

def statistic_share_unique_cnt_GRs(rules0: dict, rules1: dict):
    assert not is_prob_GRs(rules0), "statistic should be cnt type"
    share_GR0, share_GR1 = GRs_set_operate(rules0, rules1, operat="&")
    unique_GR0, unique_GR1 = GRs_set_operate(rules0, rules1, operat="-")
    # amount
    all_GR0_cnt, share_GR0_cnt, unique_GR0_cnt = sum(list(rules0.values())), sum(list(share_GR0.values())), sum(list(unique_GR0.values()))
    all_GR1_cnt, share_GR1_cnt, unique_GR1_cnt = sum(list(rules1.values())), sum(list(share_GR1.values())), sum(list(unique_GR1.values()))
    assert unique_GR0_cnt + share_GR0_cnt == all_GR0_cnt and unique_GR1_cnt + share_GR1_cnt == all_GR1_cnt, ""
    
    cnt_share_prob0, cnt_share_prob1 = round((share_GR0_cnt / all_GR0_cnt) * 100, 2), round((share_GR1_cnt / all_GR1_cnt) * 100, 2)
    cnt_unique_prob0, cnt_unique_prob1 = round((unique_GR0_cnt / all_GR0_cnt) * 100, 2), round((unique_GR1_cnt / all_GR1_cnt) * 100, 2)
    print(f"There are {all_GR0_cnt} SRC GRs amount, {all_GR1_cnt} cnts of TGT GRs")
    print(f"SRC share prob: {cnt_share_prob0}%, SRC unique prob: {cnt_unique_prob0}%")
    print(f"TGT share prob: {cnt_share_prob1}%, TGT unique prob: {cnt_unique_prob1}%")

    # types
    all_GR0_cnt, share_GR0_cnt, unique_GR0_cnt = len(rules0), len(share_GR0), len(unique_GR0)
    all_GR1_cnt, share_GR1_cnt, unique_GR1_cnt = len(rules1), len(share_GR1), len(unique_GR1)
    assert unique_GR0_cnt + share_GR0_cnt == all_GR0_cnt and unique_GR1_cnt + share_GR1_cnt == all_GR1_cnt, ""

    cnt_share_prob0, cnt_share_prob1 = round((share_GR0_cnt / all_GR0_cnt) * 100, 2), round((share_GR1_cnt / all_GR1_cnt) * 100, 2)
    cnt_unique_prob0, cnt_unique_prob1 = round((unique_GR0_cnt / all_GR0_cnt) * 100, 2), round((unique_GR1_cnt / all_GR1_cnt) * 100, 2)
    print(f"There are {all_GR0_cnt} SRC GRs types, {all_GR1_cnt} cnts of TGT GRs")
    print(f"SRC share prob: {cnt_share_prob0}%, SRC unique prob: {cnt_unique_prob0}%")
    print(f"TGT share prob: {cnt_share_prob1}%, TGT unique prob: {cnt_unique_prob1}%")

# Notes: all functin rules0 -> src GRs, rules1-> tgt GRs

def statistic_words(trees: List[Tree]):
    w_len_list = []
    for tree in trees:
        w_len_list.append(len(list(tree.leaves())))
    print(f"snt avg len: {round(sum(w_len_list) / len(trees), 4)}")

def get_rules_from_path(path, prob=False, words_statistic=False, sample=False, topK=None):
    all_lines = [bracket_line for bracket_line in open(path, 'r', encoding='utf-8')]

    if sample:
        assert len(all_lines) > sample, ""
        all_lines = random.sample(all_lines, sample)
    if topK is not None:
        all_lines = all_lines[:topK]
        assert len(all_lines) == topK, ""
        print(topK)
    trees = [load_tree_from_str(bracket_line, top_del=True) for bracket_line in all_lines]
    
    if words_statistic:
        statistic_words(trees)

    nonT_cnt_GRs, T_cnt_GRs, all_cnt_GRs = get_3rules(trees, prob=prob)
    return nonT_cnt_GRs, T_cnt_GRs, all_cnt_GRs

def save_3GRs(GRs_list: List[dict], folder, domain=None, GR_types=None):
    assert GR_types == ["nonT", "T", "all"] and domain in ["PTB.train", "PTB", "Dialogue", "Forum", "Law", "Literature","Review"]
    assert len(GRs_list) == len(GR_types)
    for type_, GRs in zip(GR_types, GRs_list):
        with open(folder + domain + '.' + type_, 'w', encoding='utf-8')as F:
            sorted_GRs = sorted(GRs.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_GRs:
                F.write(f"{k[0]}->{' '.join(k[1])} ||| {str(v)}" + '\n')

def get_write_GRs_2files(pathin1, pathin2, pathout, topK=None):
    print(f"read GRs of {pathin1}")
    _, _, all_GRs_1 = get_rules_from_path(pathin1, prob=False, words_statistic=False, sample=False)
    _, _, all_GRs_2 = get_rules_from_path(pathin2, prob=False, words_statistic=False, sample=False, topK=topK)
    print(len(all_GRs_1), len(all_GRs_2))
    all_GRs = dict(Counter(all_GRs_1) + Counter(all_GRs_2))
    all_GRs = convert_cnt2prob(all_GRs)
    with open(pathout, 'w', encoding='utf-8')as F:
        for k, v in sorted(all_GRs.items(), key=lambda x: x[1], reverse=True):
            F.write(f"{k[0]}->{' '.join(k[1])} ||| {str(v)}" + '\n')
    print(f"{len(all_GRs)} GRs write in {pathout}")

def get_write_GRs_1file(pathin, pathout, topK=None):
    print(f"read GRs of {pathin}")
    _, _, all_GRs = get_rules_from_path(pathin, prob=False, words_statistic=False, sample=False, topK=topK)
    all_GRs = convert_cnt2prob(all_GRs)
    with open(pathout, 'w', encoding='utf-8')as F:
        for k, v in sorted(all_GRs.items(), key=lambda x: x[1], reverse=True):
            F.write(f"{k[0]}->{' '.join(k[1])} ||| {str(v)}" + '\n')
    print(f"{len(all_GRs)} GRs write in {pathout}")

def load_trees_snts_by_num(path, tgt_tree_path, tgt_snts_path, num=30):
    tree_list, snt_list = [], []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        tree_list.append(line)
        snt_list.append(' '.join(list(load_tree_from_str(line).leaves())))
        if len(tree_list) >= num:
            break
    
    with open(tgt_tree_path, 'w', encoding='utf-8')as W1, open(tgt_snts_path, 'w', encoding='utf-8')as W2 :
        for tree, snt in zip(tree_list, snt_list):
            W1.write(tree + '\n')
            W2.write(snt + '\n')    


