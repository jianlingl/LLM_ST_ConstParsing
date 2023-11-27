from typing import List


class Tree:
    def __init__(self, label, children, left, right):
        self.label = label
        self.word = None if not isinstance(children, str) else children
        self.children = children if not isinstance(children, str) else None
        self.left = left
        self.right = right
        self.is_leaf = False if self.word is None else True

    def leaves(self):
        if self.is_leaf:
            yield self.word
        else:
            for child in self.children:
                yield from child.leaves()

    def pos(self):
        if self.is_leaf:
            yield self.label
        else:
            for child in self.children:
                yield from child.pos()

    def modify_pos(self, leaf_tag:dict):
        if self.is_leaf:
            assert self.word in leaf_tag.keys(), ""
            self.label = leaf_tag
        else:
            for child in self.children:
                child.modify_pos(leaf_tag)

    def span_labels(self):
        if self.is_leaf:
            res = []
        else:
            res = [self.label]
            for child in self.children:
                res += child.span_labels()
        return res

    def rules_seperate(self, r={}, t_r={}):
        if self.is_leaf:
            k = (self.label, (self.word,))
            t_r[k] = t_r.get(k, 0) + 1
        else:
            k = (self.label, tuple([child.label for child in self.children]))
            r[k] = r.get(k, 0) + 1
            for child in self.children:
                child.rules_seperate(r, t_r)
        
    def rules_dominate(self, hr={}):
        if self.label == "TOP":
            assert len(self.children) == 1, ""
            key = self.children[0].label + '->' + str([child.label for child in self.children[0].children])
            hr[key] = hr.get(key, 0) + 1
        else:
            key = self.label + '->' + str([child.label for child in self.children])
            hr[key] = hr.get(key, 0) + 1

    def get_labeled_spans(self, strip_top=True):
        if self.is_leaf:
            res = []
        else:
            if strip_top and self.label == 'TOP':
                res = []
            else:
                res = [(self.left, self.right, self.label)]

            for child in self.children:
                res += child.get_labeled_spans(strip_top)
        return res

    def linearize(self):
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])

        return '(%s %s)' % (self.label, text)


def write_tree(tree_list: List[Tree], path):
    with open(path, 'w', encoding='utf-8') as W:
        for tree in tree_list:
            tree.debinarize()
            W.write(Tree("TOP", [tree], tree.left, tree.right).linearize() + '\n')

def build_tree(tokens, idx, span_left_idx):
    idx += 1
    label = tokens[idx]
    idx += 1
    assert idx < len(tokens)

    # 若直接是）说明当前短语只有标签
    assert tokens[idx] != ')', "empty label here"
    
    if tokens[idx] == '(':
        children = []
        span_right_idx = span_left_idx
        while idx < len(tokens) and tokens[idx] == '(':
            child, idx, span_right_idx = build_tree(tokens, idx, span_right_idx)
            children.append(child)
        tree = Tree(label, children, span_left_idx, span_right_idx)

    else:
        word = tokens[idx]
        # 对特殊字符进行处理
        word = ptb_unescape([word])[0] 
        assert len(word) != 0, "after ptb none return!"
        idx += 1
        span_right_idx = span_left_idx + 1
        tree = Tree(label, word, span_left_idx, span_right_idx)
        assert tree.is_leaf
    
    assert tokens[idx] == ')'
    return tree, idx+1, span_right_idx


PTB_UNESCAPE_MAPPING = {
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}

NO_SPACE_BEFORE = {"-RRB-", "-RCB-", "-RSB-", "''"} | set("%.,!?:;")
NO_SPACE_AFTER = {"-LRB-", "-LCB-", "-LSB-", "``", "`"} | set("$#")
NO_SPACE_BEFORE_TOKENS_ENGLISH = {"'", "'s", "'ll", "'re", "'d", "'m", "'ve"}
PTB_DASH_ESCAPED = {"-RRB-", "-RCB-", "-RSB-", "-LRB-", "-LCB-", "-LSB-", "--"}

def ptb_unescape(words):
    cleaned_words = []
    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        # This un-escaping for / and * was not yet added for the
        # parser version in https://arxiv.org/abs/1812.11760v1
        # and related model releases (e.g. benepar_en2)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "（").replace("-RRB-", "）")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        cleaned_words.append(word)
    return cleaned_words

def load_treebank(path, sort=False, binarize=True, max_snt_len: int=150, top_del: bool=True):
    trees = []
    for bracket_line in open(path, 'r', encoding='utf-8'):
        t = load_tree_from_str(bracket_line, top_del=top_del)
        if len(list(t.leaves())) >= max_snt_len:
                continue

        # 二叉化
        if binarize:
            origin_t_str = t.linearize()
            t.binarize()
            trees.append(t)

            # Check the binarization and debinarization
            t_ = load_tree_from_str(t.linearize(), top_del=False)
            t_.debinarize()
            assert t_.linearize() == origin_t_str, "debinarization can not reverse to original tree"

        else:
            trees.append(t)


    print(path)
    print(len(trees))
    if sort:
        return sorted(trees, key=lambda x: len(list(x.leaves())))
    else:
        return trees

def load_tree_from_str(bracket_line: str, top_del: bool=True):
    assert bracket_line.count('(') == bracket_line.count(')')

    tokens = bracket_line.replace('(', ' ( ').replace(')',' ) ').split()
    idx, span_left_idx = 0, 0
    tree, _, _ = build_tree(tokens, idx, span_left_idx)

    if top_del:
        # 处理根节点TOP
        if len(tree.children) == 1 and tree.label == 'TOP':
            tree = tree.children[0]
        else:
            tree.label = "S"
    
    return tree


