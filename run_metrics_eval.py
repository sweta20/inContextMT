import os
import pickle
from tqdm import tqdm
import numpy as np
import json
import argparse
from metrics import *

def read_file(fname, transform=lambda x: x):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(transform(line.strip()))
    return data


def get_outputs(predictions_file, lower=False, truncate=False, max_length=None):
    predictions = read_file(predictions_file)
    outputs = []
    for i, cand in enumerate(predictions):
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
    parser.add_argument("-d", "--domain", type=str, default='medical')
    parser.add_argument("-s", "--split", type=str, default='test')
    parser.add_argument("--metric-class-name", type=str, default="COMETSrcMetric")
    parser.add_argument("--output-file", type=str, default=None, help="output text file from running XGLM.")
    parser.add_argument("--target-file", type=str, default=None, help="Reference corresponding to the outputs.")
    parser.add_argument("--source-file", type=str, default=None, help="Source corresponding to the outputs to estimate length truncation.")
    args = parser.parse_args()

    metric = getattr(sys.modules[__name__], args.metric_class_name)()

    src = read_file(f"{args.source_file}")
    ref = read_file(f"{args.target_file}")
    lengths = [len(x)*2 for x in src]

    outputs = get_outputs(f"{args.output_file}", truncate=True, max_length=lengths)

    print(args.metric_class_name, metric.get_score(ref, outputs)[0])

if __name__ == '__main__':
    main()