from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from utils import read_file, convert_input_to_template
from transformers import pipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
model.half()
model.to(device)
model.eval()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", required=True, type=str)
	parser.add_argument("-s", "--src_lang", default="de", type=str)
	parser.add_argument("-t", "--tgt_lang", default='en', type=str)
	parser.add_argument("--prompt_file", required=True, type=str)
	parser.add_argument("--k", type=int, required=True)
	parser.add_argument("-o", "--output_file", required=True, type=str)
	parser.add_argument("-c", "--max_new_tokens", default=200, type=int)
	
	args = parser.parse_args()

	src = read_file(f"{args.input_file}.{args.src_lang}")
	trg = read_file(f"{args.input_file}.{args.tgt_lang}")
	# src_lengths = [len(src[i])*2 for i in range(len(src))]

	with open(args.prompt_file, "rb") as f:
		prompts = pickle.load(f)

	if len(prompts) != len(src): # few-shot setup
		prompts = [prompts]*len(src)

	with open(args.output_file, "w") as f:
		for i in tqdm(range(len(src))):
			context_text = convert_input_to_template(prompts[i][:args.k]) + " </s> " + src[i] + " = "
			input_ids = tokenizer.encode(context_text, return_tensors='pt').to(device)
			output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
			final_output = tokenizer.decode(output[0, input_ids.shape[1]: ], skip_special_tokens=True)    	
			f.write(final_output + "\n")

if __name__ == '__main__':
	main()
