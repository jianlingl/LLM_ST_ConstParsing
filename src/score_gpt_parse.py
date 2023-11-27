
# 1 对齐 删掉没有对齐的，用字典保存 key为句子，value用树，需要去报key无重复
# 2 计算分数
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from evaluate import evalb
from trees import load_tree_from_str

def load_snt_tree_dict(path):
    snt_tree_dict = dict()
    reader = BracketParseCorpusReader("", [path])
    trees = reader.parsed_sents()
    for tree in trees:
        try:
            words = tree.leaves()
            snt = ' '.join(words)
            snt_tree_dict[snt] = tree
        except:
            pass
    # assert len(snt_tree_dict) == len(trees)
    return snt_tree_dict

def fiter_the_missing(gold_ent_tree: dict, pred_snt_tree: dict, ):
    snt_set = set(gold_ent_tree.keys())
    pred_left, gold_left = dict(), dict()
    for snt, tree in pred_snt_tree.items():
        if snt in snt_set:
            try:
                load_tree_from_str(tree.pformat(margin=1e100))
                pred_left[snt] = tree
                gold_left[snt] = gold_ent_tree[snt]
            except:
                pass
    print(len(pred_left))
    return list(gold_left.values()), list(pred_left.values())

def update_pred_tag(gold_trees, pred_trees):
    g_t, p_t = [], []
    for t1, t2 in zip(gold_trees, pred_trees):
        try:
            g_t.append(load_tree_from_str(t1.pformat(margin=1e100)))
            p_t.append(load_tree_from_str(t2.pformat(margin=1e100)))
        except:
            pass
    
    modified_pt = []
    for pt, gt in zip(p_t, g_t):
        words, tags = list(pt.leaves()), list(pt.pos())
        assert words == list(gt.leaves())
        if tags != list(gt.pos()):
            assert len(words) == len(tags), ""
            leaf_tag = {w:t for w, t in zip(words, tags)}
            pt.modify_pos(leaf_tag)
            modified_pt.append(pt)
    return g_t, modified_pt


def eval(gold_path, pred_path):
    domain = pred_path.split('/')[-1].split('.')[0]
    print("====="*3, domain, "====="*3)
    gold_snt_tree, pred_ent_tree = load_snt_tree_dict(gold_path), load_snt_tree_dict(pred_path)
    gold_trees, pred_trees = fiter_the_missing(gold_snt_tree, pred_ent_tree)
    s = evalb("EVALB/", gold_trees, pred_trees, domain=domain)
    print(s)

# if __name__ == "__main__":
#     path_list = [
#         ("data/domain/dialogue.cleaned.txt", "data/domain/demo10_ffff/dialogue.trees"),
#         ("data/domain/forum.cleaned.txt", "data/domain/demo10_ffff/forum.trees"),
#         ("data/domain/law.cleaned.txt", "data/domain/demo10_ffff/law.trees"),
#         ("data/domain/literature.cleaned.txt", "data/domain/demo10_ffff/literature.trees"),
#         ("data/domain/review.cleaned.txt", "data/domain/demo10_ffff/review.trees"),
#     ]
#     for (gold_path, pred_path) in path_list:
#         eval(gold_path, pred_path)