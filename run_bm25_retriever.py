import torch
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
import os
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import json

def read_json_file(file_path):
    res_json = None
    with open(file_path) as f:
        res_json = json.load(f)
    return res_json


def read_jsonl_file(file_path):
    items = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            items.append(item)

    return items

def read_file(fname, transform=lambda x: x):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(transform(line.strip()))
    return data


def get_outputs(predictions_file, lower=False, truncate=False, max_length=None):
    predictions = read_jsonl_file(predictions_file)
    outputs = []
    for i, pred in enumerate(predictions):
        cand = pred["candidates"][0][0]
        if lower:
            cand = cand.lower()
        if truncate:
            if isinstance(max_length, list):
                cand = cand[:max_length[i]]
            else:
                cand = cand[:max_length]
        outputs.append(cand)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data-dir", type=str, default='multi-domain/')
    parser.add_argument("-d", "--domain", type=str, default='medical')
    parser.add_argument("-s", "--split", type=str, default='test')
    parser.add_argument("--slang", type=str, default="de")
    parser.add_argument("--tlang", type=str, default="en")
    parser.add_argument("--lang", type=str, default="de")

    parser.add_argument("-o", "--output_index", type=str)    
    parser.add_argument("--create_index", action='store_true')
    parser.add_argument("--search", action='store_true')
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--target-file", type=str, default=None, help="output json file from running XGLM.")
    parser.add_argument("--source-file", type=str, default=None, help="Source corresponding to the outputs to estimate length truncation.")

    parser.add_argument("--retrieved_examples_file", type=str, help="path to dump retrieved examples")
    args = parser.parse_args()

    if args.create_index:
        my_dict = {args.slang: read_file(f"{args.data_dir}/{args.domain}/train.{args.slang}"),
                args.tlang: read_file(f"{args.data_dir}/{args.domain}/train.{args.tlang}")}
        for lang in [args.slang, args.tlang]:
            mt_tok = MosesTokenizer(lang=lang)
            corpus = my_dict[lang]
            tokenized_corpus = []
            for i in range(len(corpus)):
                tokenized_corpus.append(mt_tok.tokenize(corpus[i]))

            bm25 = BM25Okapi(tokenized_corpus)
            with open(f"{args.output_index}_{lang}.pkl", "wb") as f:
                pickle.dump(bm25, f)
            
    elif args.search:
        lang = args.lang
        mt_tok = MosesTokenizer(lang=lang)
        with open(f"{args.output_index}_{lang}.pkl", "rb") as f:
            bm25 = pickle.load(f)
        if args.source_file is not None:
            lengths = [len(x)*2 for x in read_file(args.source_file)]
            src = get_outputs(f"{args.target_file}", truncate=True, max_length=lengths)
        else:
            src = read_file(f"{args.data_dir}/{args.domain}/{args.split}.{lang}")
        similar_outs = {}
        for ind in tqdm(range(len(src))):
            tokenized_query = mt_tok.tokenize(src[ind]) 
            doc_scores = bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(doc_scores)[::-1][:args.top_k]
            similar_outs[ind] = top_n_indices
        
        os.makedirs(os.path.dirname(args.retrieved_examples_file), exist_ok=True)
        with open(f"{args.retrieved_examples_file}", "wb") as f:
            pickle.dump(similar_outs, f)
    else:
        print("Incorrect arguments")

if __name__ == '__main__':
    main()