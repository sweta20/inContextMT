# inContextMT

This repository contains the code for our paper: [In-context Examples Selection for Machine Translation](https://arxiv.org/abs/2212.02437).

Abstract: Large-scale generative models show an impressive ability to perform a wide range of Natural Language Processing (NLP) tasks using in-context learning, where a few examples are used to describe a task to the model. For Machine Translation (MT), these examples are typically randomly sampled from the development dataset with a similar distribution as the evaluation set. However, it is unclear how the choice of these in-context examples and their ordering impacts the output translation quality. In this work, we aim to understand the properties of good in-context examples for MT in both in-domain and out-of-domain settings. We show that the translation quality and the domain of the in-context examples matter and that 1-shot noisy unrelated example can have a catastrophic impact on output quality. While concatenating multiple random examples reduces the effect of noise, a single good prompt optimized to maximize translation quality on the development dataset can elicit learned information from the pre-trained language model. Adding similar examples based on an n-gram overlap with the test source significantly and consistently improves the translation quality of the outputs, outperforming a strong kNN-MT baseline in 2 out of 4 out-of-domain datasets.


__Note__: The experiments reported in the paper were run using internal fairseq code, so the numbers might not exactly match in the paper but the overall trends should be the same. 

## Data

The multi-domain German-English dataset can be obtained from [here](https://github.com/roeeaharoni/unsupervised-domain-clusters). Unzip using `unzip multi_domain_new_split.zip -d multi-domain`.

## Running the code

Scripts to train the models reported in Table 2 and 3 can be found in `run_exps.sh`.

## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:

```
@article{agrawal2022context,
  title={In-context Examples Selection for Machine Translation},
  author={Agrawal, Sweta and Zhou, Chunting and Lewis, Mike and Zettlemoyer, Luke and Ghazvininejad, Marjan},
  journal={arXiv preprint arXiv:2212.02437},
  year={2022}
}
```
