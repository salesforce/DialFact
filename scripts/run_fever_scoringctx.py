import argparse
import sys
import jsonlines
from tqdm import tqdm
import logging
import json
import torch
import torch.nn.functional as F
import jsonlines
import random
import os
import numpy as np
from scipy.special import softmax
# os.environ["NCCL_SHM_DISABLE"] = "1"
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, default_data_collator, set_seed
from transformers import InputExample, PreTrainedTokenizer, InputFeatures
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

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


class ClassificationModel():
    def __init__(self, num_labels=2, max_length=256, model_name_or_path='albert-large-v2', config_name=None, tokenizer_name=None):
        NUM_LABELS = num_labels
        self.max_seq_length = 256
        self.model_name_or_path = model_name_or_path
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        config = AutoConfig.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            num_labels=NUM_LABELS,
            # cache_dir='.cache/',
        )
        add_prefix_space = False
        if 'roberta' in self.model_name_or_path:
            add_prefix_space = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            # cache_dir=model_args.cache_dir,
            add_prefix_space=True,
            # use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
            # cache_dir=args.cache_dir,
        )

    def get_string_text(self, tokens_a, tokens_b):
        max_num_tokens = self.max_seq_length - 3
        total_length = len(tokens_a) + len(tokens_b)
        if total_length > max_num_tokens:
            len_b = len(tokens_b)
            a_begin = max_num_tokens - len_b
            tokens_a = tokens_a[-a_begin:]
        try:
            assert len(tokens_a) + len(tokens_b) <= max_num_tokens
            assert len(tokens_a) >= 1
        except:
            import pdb;
            pdb.set_trace()
            print('some problem with preproc')
        # assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append(self.tokenizer.cls_token)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append(self.tokenizer.sep_token)
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(self.tokenizer.sep_token)
        segment_ids.append(1)

        return tokens, segment_ids

    def tokenize_function_test(self, examples):
        # Remove empty lines
        #         examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        #         examples = [line for line in examples if len(line) > 0 and not line.isspace()]
        all_texts = []
        all_segment_ids = []
        all_labels = []
        #     import pdb;pdb.set_trace()
        processed = []
        items = []
        # keys = list(examples.keys())
        # for i in range(len(examples[keys[0]])):
        #     ex = {}
        #     for k in keys:
        #         ex[k] = examples[k][i]
        #     items.append(ex)
        #     import pdb;pdb.set_trace()
        items = examples
        max_seq_length = 216
        for example in items:
            first_tokens = self.tokenizer.tokenize(example['actual'])
            for sent2 in example['prediction']:
                sec_tokens = self.tokenizer.tokenize(sent2)
                tokens = ["[CLS]"] + first_tokens + ["[SEP]"] + sec_tokens
                if len(sec_tokens) + len(first_tokens) > max_seq_length - 1:
                    tokens = tokens[:(max_seq_length - 1)]
                tokens = tokens + ["[SEP]"]

                segment_ids = [0] * (len(first_tokens) + 2)
                segment_ids += [1] * (len(sec_tokens) + 1)
                all_texts.append(tokens)
                all_segment_ids.append(segment_ids)

        tokenized = self.tokenizer.batch_encode_plus(
            all_texts,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
            return_special_tokens_mask=True,
            add_special_tokens=False,
        )

        # print(len(tokenized['input_ids']))
        padded_length = len(tokenized['input_ids'][0])
        all_segment_ids = [x + [0] * (padded_length - len(x)) for x in all_segment_ids]
        tokenized['token_type_ids'] = all_segment_ids
        # tokenized['label'] = all_labels

        return tokenized

    def tokenize_function(self, examples, sent2_type='evidence_touse', sent1_type='prediction'):
        all_texts = []
        all_segment_ids = []
        all_labels = []
        processed = []
        items = []
        max_seq_length = 216
        for example in examples:
            evidence_data = example[sent2_type]
            sent2 = evidence_data
            for p, sent1 in enumerate(example[sent1_type]):
                if type(evidence_data) is list:
                    sent2 = example[sent2_type][p]
                items.append([sent2, sent1])
        # import pdb;pdb.set_trace()
        try:
            batch_encoding = self.tokenizer(
                [(example[0], example[1])
                 for example in items],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
        except:
            import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()

        features = []
        input1 = list(batch_encoding.keys())[0]
        num_inputs = len(batch_encoding[input1])
        for i in range(num_inputs):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs)
            features.append(feature)

        return features

    def tokenize_function_data(self, examples, sent2_type='evidence_touse', sent1_type='response'):
        all_texts = []
        all_segment_ids = []
        all_labels = []
        processed = []
        items = []
        max_seq_length = 256
        for example in examples:
            evidence_data = example[sent2_type]
            sent2 = evidence_data
            sent1 = example[sent1_type]
            sent1 = '[CONTEXT]: ' + ' [EOT] '.join(example['context'][-2:]) + ' [RESPONSE]: ' + sent1
            items.append([sent2, sent1])
        # import pdb;pdb.set_trace()
        try:
            batch_encoding = self.tokenizer(
                [(ex[0], ex[1])
                 for ex in items],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
        except:
            import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()

        features = []
        input1 = list(batch_encoding.keys())[0]
        num_inputs = len(batch_encoding[input1])
        for i in range(num_inputs):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs)
            features.append(feature)

        return features

def create_data_loader(tokenized_eval_dataset, batch_size):

    return DataLoader(
        tokenized_eval_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=default_data_collator
        
    )

def score_testdata(args, classification_model_dnli, testdata):
    tokenized_eval_dataset = classification_model_dnli.tokenize_function_data(testdata, sent1_type=args.response_tag)
#     import pdb;pdb.set_trace()
#     tdataset = Dataset.from_dict(tokenized_eval_dataset)
#     test_data_loader = create_data_loader(tdataset, args.batch_size)
    test_data_loader = create_data_loader(tokenized_eval_dataset, args.batch_size)
    all_scores = []
    parsed = 0
    for idx, d in enumerate(tqdm(test_data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        outputs = classification_model_dnli.model(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        outputs = softmax(outputs['logits'].tolist(),axis=1)
        for oidx, out in enumerate(outputs):
            softmax_l1 = out.tolist()
            # dnli_score = [x[0] for x in softmax_l1]
    #         print(softmax_l1)s
            # all_scores+=softmax_l1
            testdata[parsed][args.typeprefix+'fever_score'] = softmax_l1
            parsed+=1
    
def score_data(args, classification_model_dnli, max_evidences=5):
    testdata = get_json_lines(args.input_file)
    for i, datapoint in enumerate(tqdm(testdata)):
#         lines = datapoint[args.response_tag]
        if 'evidence_list' in datapoint:
            all_evidences = datapoint['evidence_list'][:max_evidences]
            # for e, evilist in enumerate(datapoint['evidence_list'][:max_evidences]):
            #     all_evidences = evilist#datapoint['evidence_list']
            #     print(all_evidences)
            #     print(['title: ' + x[0] + ' content: ' + x[2] for x in all_evidences])
            all_evidence_texts = ['title: ' + x[0] + ' content: ' + x[2] for x in all_evidences]
            # evidence_text = ' ### '.join(all_evidence_texts)
            evidence_text = ' '.join(all_evidence_texts)
            datapoint['evidence_touse'] = evidence_text

        if args.claim_only:
            datapoint['evidence_touse'] = ''
        # import pdb;pdb.set_trace()
        if len(datapoint[args.response_tag])==0:
            continue


    score_testdata(args, classification_model_dnli, testdata)
        # scores = lm_scores(lines, model, tokenizer, device)
#         datapoint['dnli_score'] = scores
        
    write_json_lines(args.preds_file, testdata, args.output_folder)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_device', type=int, help='id of GPU to use', default=0)
    parser.add_argument('-m', '--model', type=str, help='model name to use', default='colloquial_bert_large/')
    parser.add_argument('-i', '--input_file', type=str, help='path to the file containing the evaluation data', required=True)
    parser.add_argument('-o', '--preds_file', type=str, help='output file to save the results')
    parser.add_argument('--output_folder', type=str, help='output file to save the results', default='colloquialfeverscores/')
    parser.add_argument('--response_tag', type=str, help='tag', default='response')
    parser.add_argument('--batch_size', type=int, help='batch size', default=20)
    parser.add_argument('--claim_only', action='store_true', default=False, help='Disables evidence')
    parser.add_argument('--max_seq_length', type=int, help='batch size', default=256)
    parser.add_argument('--knowledgeformat', type=str, help='tag', default='') # wikijoin
    parser.add_argument('--typeprefix', type=str, help='tag', default='')
    parser.add_argument('--outputprefix', type=str, help='tag', default='')
    
#     parser.add_argument('-append', action='store_true', help='allow append to previous run', default=False)

    args = parser.parse_args()
    if args.preds_file is None:
        args.preds_file = args.input_file.split('/')[-1]
    
    args.preds_file = args.outputprefix + args.preds_file
#     assert(not os.path.exists(args.preds_file))
    if args.cuda_device>=0:
        device = 'cuda:'+str(args.cuda_device)
    else:
        device = 'cpu'
        
    args.device = device
    classification_model_dnli = ClassificationModel(num_labels=3,model_name_or_path=args.model)
    classification_model_dnli.model = classification_model_dnli.model.to(device)
    print('model loaded')
    classification_model_dnli.model.eval()
    score_data(args, classification_model_dnli)
        
# python fever_scoring.py -i ../post_generation/contextagg_maskfill_mix1_wow_test_tsc_200_t1.5.jsonl --output_folder vitamincscores/ -m tals/albert-xlarge-vitaminc
# python fever_scoring.py -i ../post_generation/contextagg_maskfill_mix1_wow_test_tsc_200_t1.5.jsonl --knowledgeformat wikijoin --typeprefix colloq_ --output_folder colloquialfeverscores/ -m colloquial_bert_large
