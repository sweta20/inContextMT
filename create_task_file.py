import sys
import sacrebleu
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import random
import editdistance
from utils import FewShotSample
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.translate import meteor_score, chrf_score
BETA=3.0
from utils import *

def edit_similarity(a, b):
    return 1- (editdistance.eval(a, b)/max(len(a), len(b)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_examples_file", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--src-lang", type=str, default="de", help="Eval Source Language")
    parser.add_argument("--tgt-lang", type=str, default="en", help="Eval Target Language")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--rerank-samples", action='store_true')
    parser.add_argument("--metric-name", type=str)
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()

    
    train_src, train_tgt = get_data(args.domain, args.src_lang, args.tgt_lang, "train")
    eval_samples, _ = get_data(args.domain, args.src_lang, args.tgt_lang, args.split)


    with open(args.retrieved_examples_file, "rb") as f:
        incontext_examples = pickle.load(f)
    
    print("Loaded incontext_examples")
    prompts = {}
    for i, sample in tqdm(enumerate(eval_samples)):
        top_n_indices = incontext_examples[i]
        train_samples = []
        metric_scores = []
        for k in top_n_indices:
            if args.retrieval_method == "bm25":
                train_samples.append(FewShotSample(data={
                    "src": train_src[int(k)],
                    "tgt": train_tgt[int(k)],
                }, correct_candidates=[train_tgt[int(k)]]))
            if args.rerank_samples:
                if args.metric_name == "bleu":
                    metric_scores.append(sacrebleu.sentence_bleu(train_src[int(k)], [sample]).score )
                elif args.metric_name == "chrf":
                    metric_scores.append(chrf_score.sentence_chrf(sample, [train_src[int(k)]], beta=BETA))
                elif args.metric_name == "edit":
                    metric_scores.append(edit_similarity(train_src[int(k)], sample))
                else:
                    print("Error")
                    exit()

        if args.rerank_samples:
            sorted_indices = np.argsort(metric_scores)[::-1]
            train_samples_sorted = [train_samples[j] for j in sorted_indices]
            train_samples = train_samples_sorted
        prompts[i] = train_samples

    random_index = random.randint(0,len(eval_samples)-1)
    print(prompts[random_index][0].data, eval_samples[random_index])
    
    with open(args.prompt_file, "wb") as f:
        pickle.dump(prompts, f)

if __name__ == '__main__':
    main()