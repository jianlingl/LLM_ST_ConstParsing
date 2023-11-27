# LLM_enhanced_ST_4ConstParsing
A cross-domain constituency parser with LLM-enhanced self-training. Based on [LLM-enhanced Self-training for Constituency Parsing](https://arxiv.org/abs/2311.02660) from EMNLP 2023.

## install
Before training the parser, please refer to requirements.txt and install the necessary packages.  And also you need download and compile EVALB.
For the detailed information about the berkeley parser, please refer to the original work: [Constituency Parsing with a Self-Attentive Encoder](https://github.com/nikitakit/self-attentive-parser/tree/master).  

## training
As for our method, the [LLM-generation](#LLM-generation) process should be introduced in each iteration, run train_LLM.sh after generating the raw language data. 
Here are the parameters that need to be modified for each iteration:
```
    --raw-path：the LLM generated raw corpus (at most 10,000 sentences).
    --start-iter: current iteration (e.g. 0)
    --iter-cnt: current stop iteration (e.g. 1, for the vanilla self-training set to 4 directly)
    --accord: 0: Token, 1: nonTerminal, 2: GRs 3: confidence, 4: GRsConf
    --topK: the count for selected pseudo-trees
    --select-record: path to save the selected pseudo-trees
    --pretrain-parser: the parser trained in the previous iteration, if it is start iteration 0, ignores this parameter
    --tab-score-path: save the test score for domains and to use this please prepare the files in test_path_list of main.py
    --pretrained-model: select a pre-train and copy the path to this parameter, we use bert-base-uncased as well as bert-large-uncased
    --model-path-base: path to save the trained parser
```
## testing
For convenience, we combine train&test together. You can also sepeartely run test followed. 





