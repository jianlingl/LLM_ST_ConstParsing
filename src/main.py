import os
import argparse
import functools
import itertools
import os.path
import time

import torch

import numpy as np

from benepar import char_lstm
from benepar import decode_chart
from benepar import nkutil
from benepar import parse_chart
import evaluate
import learning_rates
import treebanks
from rules_dist import get_rules_from_path, update_src_GRs


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def make_hparams():
    return nkutil.HParams(
        # Data processing
        max_len_train=0,  # no length limit
        max_len_dev=0,  # no length limit
        # Optimization
        batch_size=64,
        mini_batch=4,
        learning_rate=0.00005,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0.0,  # no clipping
        checks_per_epoch=4,
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3,  # establishes a termination criterion
        # CharLSTM
        use_chars_lstm=False,
        d_char_emb=64,
        char_lstm_input_dropout=0.2,
        # BERT and other pre-trained models
        use_pretrained=False,
        pretrained_model="bert-base-uncased",
        # Partitioned transformer encoder
        use_encoder=False,
        d_model=1024,
        num_layers=8,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        encoder_max_len=512,
        # Dropout
        morpho_emb_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.1,
        residual_dropout=0.2,
        # Output heads and losses
        force_root_constituent="auto",
        predict_tags=True,
        d_label_hidden=256,
        d_tag_hidden=256,
        tag_loss_scale=5.0,
    )


def self_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed), flush=True)
        np.random.seed(args.numpy_seed)

    # seed_from_numpy = np.random.randint() # 2147483648 # 961895650 7832657190
    seed_from_numpy = args.numpy_seed
    print("Manual seed for pytorch:", seed_from_numpy, flush=True)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:", flush=True)
    print(f"topK {args.topK}", flush=True)
    hparams.print()

    print("\n args for self training: ", flush=True)
    for (k, v) in args.__dict__.items():
        if k in ["pretrain_parser", "iter_cnt", "topK", "accord", "use_pseudo_GRs", "raw_path", "remove_large_after_select", "tab_score_path"]:
            print(f"{k}: {v}")
    print('\n')

    print("Loading training trees from {}...".format(args.train_path), flush=True)
    train_treebank = treebanks.load_trees(
        args.train_path, args.train_path_text, args.text_processing
    )
    if hparams.max_len_train > 0:
        train_treebank = train_treebank.filter_by_length(hparams.max_len_train)
    print("Loaded {:,} training examples.".format(len(train_treebank)), flush=True)

    print("Loading development trees from {}...".format(args.dev_path), flush=True)
    dev_treebank = treebanks.load_trees(
        args.dev_path, args.dev_path_text, args.text_processing
    )
    if hparams.max_len_dev > 0:
        dev_treebank = dev_treebank.filter_by_length(hparams.max_len_dev)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    
    test_path_list = {
        'PTB': 'data/wsj/test.txt',
        'Dialogue': 'data/domain/dialogue.cleaned.txt',
        'Forum': 'data/domain/forum.cleaned.txt',
        'Law': 'data/domain/law.cleaned.txt',
        'Literature': 'data/domain/literature.cleaned.txt',
        'Review': 'data/domain/review.cleaned.txt',
    }
    ptb_domian_bank = {}
    for domain, test_path in test_path_list.items():
        test_treebank = treebanks.load_trees(test_path)
        ptb_domian_bank[domain] = (test_treebank, test_path)

    print("Constructing vocabularies...")
    label_vocab = decode_chart.ChartDecoder.build_vocab(train_treebank.trees)
    if hparams.use_chars_lstm:
        char_vocab = char_lstm.RetokenizerForCharLSTM.build_vocab(train_treebank.sents)
    else:
        char_vocab = None

    tag_vocab = set()
    for tree in train_treebank.trees:
        for _, tag in tree.pos():
            tag_vocab.add(tag)
    tag_vocab = ["UNK"] + sorted(tag_vocab)
    tag_vocab = {label: i for i, label in enumerate(tag_vocab)}

    if hparams.force_root_constituent.lower() in ("true", "yes", "1"):
        hparams.force_root_constituent = True
    elif hparams.force_root_constituent.lower() in ("false", "no", "0"):
        hparams.force_root_constituent = False
    elif hparams.force_root_constituent.lower() == "auto":
        hparams.force_root_constituent = (
            decode_chart.ChartDecoder.infer_force_root_constituent(train_treebank.trees)
        )
        print("Set hparams.force_root_constituent to", hparams.force_root_constituent)

    # init train
    if args.pretrain_parser is not None:
        parser = parse_chart.ChartParser.from_trained(args.pretrain_parser)
        parser.cuda()
    else:
        parser = train_new_parser(0, tag_vocab, label_vocab, char_vocab, args, hparams, train_treebank, dev_treebank)

    # init test
    if args.start_iter == 0:
        print("PTB and domain tesing here ...")
        domain_score = ptb_domian_test(parser, ptb_domian_bank, args.evalb_dir, args.subbatch_max_tokens)
        tab_iter_res(args.tab_score_path, args.start_iter, domain_score)

    
    if args.iter_cnt > 0:
        # prepare for parse trainging and pseudo data selection
        raw_snts = treebanks.load_raw_snts(args.raw_path)
        Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs = get_rules_from_path(args.train_path)
        if args.start_iter > 0:
            train_treebank.combine_two_bank(treebanks.load_trees(args.select_record))
            print(f"There are {len(train_treebank.trees)} (should be {39832 + args.topK * args.start_iter}) trees in all.")

        # self training iteration
        for iter in range(1, args.iter_cnt + 1):
            if iter > args.start_iter:
                print("select pseudo data here ...")
                raw_snts, pseudo_treebank = pred_with_parser(iter, parser, raw_snts, args.subbatch_max_tokens, args.topK, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, args.remove_large_after_select, accord=args.accord, select_record=args.select_record)
                train_treebank.combine_two_bank(pseudo_treebank)

                print(f'iter {iter}th for self training...')
                parser = train_new_parser(iter, tag_vocab, label_vocab, char_vocab, args, hparams, train_treebank, dev_treebank)

                print("PTB and domain tesing here ...")
                domain_score = ptb_domian_test(parser, ptb_domian_bank, args.evalb_dir, args.subbatch_max_tokens)
                tab_iter_res(args.tab_score_path, iter, domain_score)

                # prepare for next iteration pseudo data selection
                if iter < args.iter_cnt and args.use_pseudo_GRs: # the last training do not need further select pseudo data
                    Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs = update_src_GRs(pseudo_treebank.trees, Src_cnt_nonT_GRs, Src_cnt_T_GRs)

            else:
                print(f"iter {iter}th already done !!! ")

    else:
        print('Just train an init parser as baseline')


def train_new_parser(iter, tag_vocab, label_vocab, char_vocab, args, hparams, train_treebank, dev_treebank):
    print("Initializing model...", flush=True)
    parser = parse_chart.ChartParser(
    tag_vocab=tag_vocab,
    label_vocab=label_vocab,
    char_vocab=char_vocab,
    hparams=hparams,
    )
            
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()
    else:
        print("Not using CUDA!")

    print("Initializing optimizer...")
    trainable_parameters = [
        param for param in parser.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=hparams.learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = learning_rates.WarmupThenReduceLROnPlateau(
        optimizer,
        hparams.learning_rate_warmup_steps,
        mode="max",
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience * hparams.checks_per_epoch,
        verbose=True,
    )

    clippable_parameters = trainable_parameters
    grad_clip_threshold = (
        np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm
    )

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_treebank) / hparams.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed

        dev_start_time = time.time()

        dev_predicted = parser.parse(
            dev_treebank.without_gold_annotations(),
            subbatch_max_tokens=args.subbatch_max_tokens,
            use_gold_pos=True,
        )
        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank.trees, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            ), flush=True
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_itr{}_dev={:.2f}".format(
                args.model_path_base, iter, dev_fscore.fscore
            )
            best_dev_processed = total_processed
            print("Saving new best model to {}...".format(best_dev_model_path))
            torch.save(
                {
                    "config": parser.config,
                    "state_dict": parser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                best_dev_model_path + ".pt",
            )

    data_loader = torch.utils.data.DataLoader(
        train_treebank,
        batch_size=hparams.mini_batch,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate_subbatches,
            subbatch_max_tokens=args.subbatch_max_tokens,
        ),
    )
    step, accm_steps = 0, hparams.batch_size / hparams.mini_batch
    batch_loss_value = 0.
    optimizer.zero_grad()
    for epoch in itertools.count(start=1):
        epoch_start_time = time.time()

        for batch_num, batch in enumerate(data_loader, start=1):
            step += 1
            parser.train()

            for subbatch_size, subbatch in batch:
                loss = parser.compute_loss(subbatch)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += subbatch_size
                current_processed += subbatch_size

            if step % accm_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    clippable_parameters, grad_clip_threshold
                )

                optimizer.step()
                optimizer.zero_grad()
                print(
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "grad-norm {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {}".format(
                        epoch,
                        batch_num,
                        int(np.ceil(len(train_treebank) / hparams.batch_size)),
                        total_processed,
                        batch_loss_value,
                        grad_norm,
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                )

                if current_processed >= check_every:
                    current_processed -= check_every
                    check_dev()
                    scheduler.step(metrics=best_dev_fscore)
                else:
                    scheduler.step()
                
                batch_loss_value = 0.0

        if (total_processed - best_dev_processed) > (
            (hparams.step_decay_patience + 1)
            * hparams.max_consecutive_decays
            * len(train_treebank)
        ):
            print("Terminating due to lack of improvement in dev fscore.")
            break

    return parser 

def ptb_domian_test(parser: parse_chart.ChartParser, ptb_domian_bank: dict, evalb_dir: str, subbatch_max_tokens: int):
    print("==============="*8, flush=True)
    import math
    domain_score = {}
    for domain, (test_treebank, test_path) in ptb_domian_bank.items():
        test_predicted = parser.parse(test_treebank.without_gold_annotations(), subbatch_max_tokens=subbatch_max_tokens*2, use_gold_pos=True)
        test_fscore = evaluate.evalb(evalb_dir, test_treebank.trees, test_predicted, ref_gold_path=test_path)

        domain_score[domain] = test_fscore
        print("test-fscore {} of {}".format(test_fscore, domain))
    print("==============="*8)
    return domain_score

def pred_with_parser(iter, parser: parse_chart.ChartParser, raw_snts: treebanks.Treebank, subbatch_max_tokens: int, topK: int, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, remove_large_after_select, accord=2, select_record=''):
    snt_tree_score_dic = parser.parse(raw_snts, subbatch_max_tokens=subbatch_max_tokens*2, return_predT_scores=True)
    # confidence or all GRs or nonT GRs
    print("start select ...", flush=True) 
    pseudo_trees, choosed_snts, ids = treebanks.choose_topK_GRs(snt_tree_score_dic, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, topK, remove_large_after_select, accord=accord)
    with open(select_record, 'a', encoding='utf-8')as W:
        for tree in sorted(pseudo_trees):
            W.write(tree.pformat(margin=1e100) + '\n')
    pseudo_treebank = treebanks.load_as_trees(pseudo_trees, choosed_snts)
    raw_snts = treebanks.remove_topK_snts(raw_snts, choosed_snts, ids)
    print(f"remove after {iter}th select: {remove_large_after_select}, there are {len(raw_snts)} raw sents left.")
    return raw_snts, pseudo_treebank

def tab_iter_res(tab_path, iter, domain_scores: dict, message=None):
    from tabulate import tabulate
    add_header = True if not os.path.exists(tab_path) else False
    domain = ['PTB', 'Dialogue', 'Forum', 'Law', 'Literature', 'Review']

    def str_score(f: evaluate.FScore, mode):
        if mode == 'R':
            return str(f.recall)
        elif mode == 'P':
            return str(f.precision)
        elif mode == 'F':
            return str(f.fscore)

    with open(tab_path, 'a+', encoding='utf-8') as F:
        if add_header:
             F.write(tabulate(
                 [['iter\ domain'] + domain, ]
            ) + '\n')
        assert [d for d, _ in domain_scores.items()] == domain
        F.write(tabulate(
            [[str(iter) + '_' + mode] + [str_score(f, mode) for _, f in domain_scores.items()] for mode in ["R", "P", "F"]]
        ) + '\n')

def run_test(args):
    test_path_list = {
        'PTB': 'data/wsj/test.txt',
        'Dialogue': 'data/domain/dialogue.cleaned.txt',
        'Forum': 'data/domain/forum.cleaned.txt',
        'Law': 'data/domain/law.cleaned.txt',
        'Literature': 'data/domain/literature.cleaned.txt',
        'Review': 'data/domain/review.cleaned.txt',
    }
    parser = parse_chart.ChartParser.from_trained(args.model_path)
    parser.cuda()
    domain_scores = {}
    for domain, domain_path in test_path_list.items():
        domain_bank = treebanks.load_trees(domain_path)
        predicted = parser.parse(domain_bank.without_gold_annotations(), subbatch_max_tokens=args.subbatch_max_tokens, use_gold_pos=True)
        ref_gold_path = domain_path
        test_fscore = evaluate.evalb(args.evalb_dir, domain_bank.trees, predicted, ref_gold_path=ref_gold_path, domain=domain)
        print("domain {} test-fscore {} ".format(domain, test_fscore))
        domain_scores[domain] = test_fscore
    tab_iter_res("log/score_tab/ptb_domain.scores", 1, domain_scores)

def run_pred(args):
    parser = parse_chart.ChartParser.from_trained(args.model_path)
    parser.cuda()
    raw_snts = treebanks.load_raw_snts(args.raw_path)
    pseudo_trees = parser.parse(raw_snts, subbatch_max_tokens=args.subbatch_max_tokens, return_predT_scores=False)

    # TODO run only for select data
    snt_tree_score_dic = parser.parse(raw_snts, subbatch_max_tokens=args.subbatch_max_tokens, return_predT_scores=True)
    print("start select ...", flush=True) 
    Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs = get_rules_from_path(args.train_path)
    topK, remove_large_after_select, accord = args.topK, args.remove_large_after_select, args.accord
    print("topK, remove_large_after_select, accord -->", topK, remove_large_after_select, accord)
    pseudo_trees, choosed_snts, ids = treebanks.choose_topK_GRs(snt_tree_score_dic, Src_cnt_nonT_GRs, Src_cnt_T_GRs, Src_cnt_all_GRs, topK, remove_large_after_select, accord=accord)

    # pseudo_treebank = treebanks.load_as_trees(pseudo_trees, choosed_snts)
    # raw_snts = treebanks.remove_topK_snts(raw_snts, choosed_snts, ids)
    # print(f"remove after select: {remove_large_after_select}, there are {len(raw_snts)} raw sents left.")
    # return raw_snts, pseudo_treebank

    assert len(pseudo_trees)
    with open(args.tgt_path, "w") as outfile:
        for tree in pseudo_trees:
            outfile.write("{}\n".format(tree.pformat(margin=1e100)))

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("self_train")
    subparser.set_defaults(callback=lambda args: self_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--device", type=int, default=0)
    subparser.add_argument("--numpy-seed", type=int, default=832657190)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--select-record", type=str, required=True)
    subparser.add_argument("--pretrain-parser", default=None)
    subparser.add_argument("--iter-cnt", type=int, default=4) # iter_cnt=0 mean train baseline init parser
    subparser.add_argument("--start-iter", type=int, default=0) # 0: start from train init parser, 0<x<iter_cnt: start from trained xth parser (already trained)
    subparser.add_argument("--topK", type=int, default=4000)
    subparser.add_argument("--accord", type=int, default=0) # accord_dict ={0: "T", 1: "nonT", 2: "all" 3: "confidence", 4: "the best two combination"}
    subparser.add_argument("--use-pseudo-GRs", type=bool, default=False)
    subparser.add_argument("--raw-path", default='', required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/wsj/train.txt")
    subparser.add_argument("--train-path-text", type=str)
    subparser.add_argument("--dev-path", default="data/wsj/dev.txt")
    subparser.add_argument("--remove-large-after-select", type=int, default=10000)  # mainly for vanilla
    subparser.add_argument("--dev-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000) 
    subparser.add_argument("--tab-score-path", type=str, default="log/score_tab/ptb_domain.scores") 
    subparser.add_argument("--parallelize", action="store_true")
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--evalb-dir", default="EVALB/")

    subparser = subparsers.add_parser("pred")
    subparser.set_defaults(callback=run_pred)
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--raw-path", required=True)
    subparser.add_argument("--accord", type=int, default=4)
    subparser.add_argument("--topK", type=int, default=2000)
    subparser.add_argument("--remove-large-after-select", type=int, default=10000) # mainly for vanilla
    subparser.add_argument("--train-path", default="data/wsj/train.txt")
    subparser.add_argument("--tgt-path", default='')
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
