# train
python src/main.py self_train \
    --raw-path "data/raw/Review.txt" \
    --start-iter 0 \
    --iter-cnt 1 \
    --accord 4 \
    --topK 2000 \
    --select-record "" \
    --pretrain-parser "" \
    --tab-score-path "" \
    --use-pretrained --pretrained-model "" \
    --use-encoder --num-layers 2 \
    --model-path-base ""
# test

python src/main.py test \
    --model-path "the trained parser path" >  log/test_log/test.log 2>&1 &