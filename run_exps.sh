DATA_DIR="multi-domain"
EXPS_DIR="experiments"
mkdir -p $EXPS_DIR

domain="medical"
split="test"
weight=0.1
min_bleu_thres=1.0

SRC="${DATA_DIR}/${domain}/${split}.de"
TGT="${DATA_DIR}/${domain}/${split}.en"

mkdir -p ${EXPS_DIR}/${domain}/fewshot/
mkdir -p ${EXPS_DIR}/${domain}/bm25/

python create_random_fewshot_file.py  --domain ${domain} \
    --out-prompt-file ${EXPS_DIR}/$domain/fewshot/prompts_${domain}_random_fewshot \
    --num-trials 20

for (( i = 0; i < 2; i++ )); do
    python run_generation.py --k 1  \
    --input_file "${DATA_DIR}/${domain}/${split}" \
    --prompt_file "${EXPS_DIR}/${domain}/fewshot/prompts_${domain}_random_fewshot.${i}.pkl" \
    --output_file "${EXPS_DIR}/${domain}/fewshot/output_${domain}_${split}_random_fewshot.${i}.txt"

    python run_metrics_eval.py --domain ${domain} --split ${split} \
    --source-file ${SRC} --target-file ${TGT} \
    --output-file "${EXPS_DIR}/${domain}/fewshot/output_${domain}_${split}_random_fewshot.${i}.txt" \
    --metric-class-name "BleuMetric"
done

bm25
python run_bm25_retriever.py --create_index --domain ${domain} \
    --output_index ${EXPS_DIR}/${domain}/bm25/bm25_${domain}_model 

python run_bm25_retriever.py --search --domain ${domain} --split ${split} \
    --retrieved_examples_file  ${EXPS_DIR}/${domain}/bm25/similar_examples_${domain}_${split}_src.pkl \
    --output_index ${EXPS_DIR}/${domain}/bm25/bm25_${domain}_model

python create_task_file.py --retrieved_examples_file ${EXPS_DIR}/${domain}/bm25/similar_examples_${domain}_${split}_src.pkl \
    --domain ${domain} --retrieval_method bm25 --prompt-file ${EXPS_DIR}/${domain}/bm25/prompts_${domain}_${split}_src.pkl --split ${split}

python run_generation.py --k 1  \
    --input_file "${DATA_DIR}/${domain}/${split}" \
    --prompt_file "${EXPS_DIR}/${domain}/bm25/prompts_${domain}_${split}_src.pkl" \
    --output_file "${EXPS_DIR}/${domain}/bm25/output_${domain}_${split}_src.txt"

python run_metrics_eval.py --domain ${domain} --split ${split} \
    --source-file ${SRC} --target-file ${TGT} \
    --output-file "${EXPS_DIR}/${domain}/bm25/output_${domain}_${split}_src.txt" \
    --metric-class-name "BleuMetric"

python create_recall_set_selection.py --input-prompt-file ${EXPS_DIR}/${domain}/bm25/prompts_${domain}_${split}_src.pkl \
        --output-prompt-file ${EXPS_DIR}/${domain}/bm25/prompts_${domain}_${split}_src_recall_select_${weight}_${min_bleu_thres}.pkl \
        --domain ${domain} --weight ${weight} --split ${split} --min-bleu-threshold ${min_bleu_thres} \
        --input-source-file $SRC

python run_generation.py --k 1  \
    --input_file "${DATA_DIR}/${domain}/${split}" \
    --prompt_file "${EXPS_DIR}/${domain}/bm25/prompts_${domain}_${split}_src_recall_select_${weight}_${min_bleu_thres}.pkl" \
    --output_file "${EXPS_DIR}/${domain}/bm25/output_${domain}_${split}_src_recall_select_${weight}_${min_bleu_thres}.txt"

python run_metrics_eval.py --domain ${domain} --split ${split} \
    --source-file ${SRC} --target-file ${TGT} \
    --output-file "${EXPS_DIR}/${domain}/bm25/output_${domain}_${split}_src_recall_select_${weight}_${min_bleu_thres}.txt" \
    --metric-class-name "BleuMetric"