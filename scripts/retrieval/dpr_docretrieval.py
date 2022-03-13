import argparse
import json
import os
import re
import time
from multiprocessing.pool import ThreadPool
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import datasets
from datasets import load_dataset
import jsonlines
from tqdm import tqdm
import torch
wiki = load_dataset("wiki_dpr", index_name='exact', with_embeddings=False, with_index=True, split="train", cache_dir='./cachedir')

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        for obj in reader:
            lines.append(obj)
            
    return lines

def write_json_lines(output_file_name, list_data, output_folder):
    with jsonlines.open(output_folder+ output_file_name, mode='w') as writer:
        for dataline in list_data:
            writer.write(dataline)

class DPRDoc_Retrieval:

    def __init__(self, topk=100, model_type='ftwctx'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.topk = topk
        self.model_type = model_type
#         self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
#         self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        if self.model_type=='original':
            self.q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        elif self.model_type=='ftnoctx':
            self.q_encoder = DPRQuestionEncoder.from_pretrained("nqt2e13/")
        else:#ftwctx
            print('model wctx')
            self.q_encoder = DPRQuestionEncoder.from_pretrained(
                "wctxt1e21/")

        self.q_encoder = self.q_encoder.to(self.device)
    
    def get_top(self, question, topk=5):
        question_emb = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0].detach().numpy()
        passages_scores, passages = wiki.get_nearest_examples("embeddings", question_emb, k=selftopk)
        all_passgae = ""
        for score, title, text in zip(passages_scores, passages['title'], passages['text']):
            if len(all_passgae.split(" ")) < 450:
                all_passgae += f" ({title}) {text}"
        return all_passgae

    def get_top_passages(self, question, topk=None):
        if topk is None:
            topk = self.topk
        question_emb = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt").to(self.device))[0].cpu().detach().numpy()
        passages_scores, passages = wiki.get_nearest_examples("embeddings", question_emb, k=topk)

        return passages_scores, passages

def add_retrieved_documents(args, examples):
    ret_object = DPRDoc_Retrieval(topk=args.topk, model_type=args.model_type)
    for example in tqdm(examples):
        responses = example['response_candidates']
        context = example['context']
        context_string = ' '.join(context[-2:])
        response_docs = []
        for adp, response in enumerate(responses):
            inputstring = response
            if args.use_context:
                inputstring = ' [eot] '.join(example['context'][-2:]) + ' [SEP] ' + response
            passages_scores, passages = ret_object.get_top_passages(inputstring)
#             print(response)
            pages = dict()
            for score, title, text in zip(passages_scores, passages['title'], passages['text']):
#                 print(score, title, " : ", text)       
#                 pages.append([title, score, text])
                if title not in pages:
                    pages[title] = []
                pages[title].append(text)
            response_docs.append(pages)
#             print(adp)
        example['response_docs'] = response_docs
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='path to the file containing the evaluation data', required=True)
    parser.add_argument('-o', '--preds_file', type=str, help='output file to save the results')
    parser.add_argument('--topk', type=int, help='topk results', default=100)
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--model_type', type=str, help='model type from ftwctx,ftnoctx,original')
    parser.add_argument('--output_folder', type=str, help='output file to save the results', default='output/')
#     parser.add_argument('-append', action='store_true', help='allow append to previous run', default=False)

    args = parser.parse_args()
    if args.preds_file is None:
        args.preds_file = args.model_type+str(args.topk)+'topk_dprdocs_' + args.input_file.split('/')[-1]
        if args.use_context:
            args.preds_file = args.model_type+str(args.topk)+'topk_dprdocs_wctx_' + args.input_file.split('/')[-1]

    if args.model_type=='ftwctx':
        args.use_context=True
        print('set use_context true')

    input_examples = get_json_lines(args.input_file)  # [:5]
    input_examples = input_examples[:]
    print(len(input_examples))
    add_retrieved_documents(args, input_examples)


    write_json_lines(args.preds_file, input_examples, args.output_folder)

#python dpr_docretrieval.py -i $INPTU_FILE --topk 100 --model_type ftwctx --use_context
