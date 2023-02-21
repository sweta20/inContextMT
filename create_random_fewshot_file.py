import os    
import pickle
import argparse
import random
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--src-lang", type=str, default="de", help="Eval Source Language")
    parser.add_argument("--tgt-lang", type=str, default="en", help="Eval Target Language")
    parser.add_argument("--out-prompt-file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--k", type=str, default=1)

    args = parser.parse_args()

    train_src, train_tgt = get_data(args.domain, args.src_lang, args.tgt_lang, "train")

    random.seed(args.seed)
    num_itrs = args.num_trials
    for j in range(num_itrs):
        random.seed(j)
        indices = random.sample(range(len(train_src)), args.k)
        prompts =  [
                    FewShotSample(data={
                    "src": train_src[ind], 
                    "tgt": train_tgt[ind]
                }, correct_candidates=[train_tgt[ind]])
                for ind in indices ]
        prompt_file=f"{args.out_prompt_file}.{j}.pkl"

        with open(prompt_file, "wb") as f:
            pickle.dump(prompts, f)


if __name__ == '__main__':
    main()