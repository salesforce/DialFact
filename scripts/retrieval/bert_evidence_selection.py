import random, os
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from torch.autograd import Variable
# from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam
import torch
import torch.optim as optim
import torch.nn.functional as F
# from data_loader import DataLoader, DataLoaderTest
from torch.nn import NLLLoss
import logging
import json
from tqdm import tqdm
logger = logging.getLogger(__name__)
from bert_model import BertForSequenceEncoder
from transformers import BertForSequenceClassification, BertTokenizer
from eviselect_models import inference_model
from data_loader import DataLoaderLines



def get_model_predictions(validset_reader, model, topk=5):
    all_results = []
    all_predict = dict()
    for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in (validset_reader):
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(evi_list)
        for i in range(len(probs)):
            if ids[i] not in all_predict:
                all_predict[ids[i]] = []
            #if probs[i][1] >= probs[i][0]:
            all_predict[ids[i]].append(evi_list[i] + [probs[i]])
        
    for key, values in all_predict.items():
        sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
        data = {"id": key, "evidence": sorted_values[:topk]}
        all_results.append(data)
    
    return all_results

def test_samples(args, tokenizer, model):
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading data set")
    # validset_reader = DataLoaderTest(args.test_path, tokenizer, args, batch_size=args.batch_size)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    # bert_model = BertForSequenceClassification.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    model.eval()
    logger.info('Start eval!')

    for inp in inputs:
        if 'context' in inp:
            inp['original_claim'] = inp['claim']
            inp['claim'] = ' [EOT] '.join(inp['context']) + ' [EOT] ' + inp['claim']

    validset_reader = DataLoaderLines(inputs, tokenizer)

    all_results = get_model_predictions(validset_reader, model)
    
    pprint(all_results)
    


        
class Sentence_Retrieval_Bert:

    def __init__(self, bert_pretrain='bert-base-cased', no_cuda=False, deviceid=0, outdir='./output', checkpoint=None, max_len=256):
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.deviceid = deviceid
        self.device = torch.device('cuda:' + str(self.deviceid))
        self.bert_pretrain = bert_pretrain
        self.outdir = outdir
        self.max_len = max_len
        self.checkpoint = checkpoint
        if self.checkpoint is None:
            self.checkpoint = '/export/home/code/common/gitrepos/KernelGAT/checkpoint/retrieval_modelft_wowctxutt/model.best.pt'
            
        handlers = [logging.FileHandler(os.path.abspath(outdir) + '/test_log.txt'), logging.StreamHandler()]
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                            datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrain, do_lower_case=False)
        logger.info("loading data set")
        # validset_reader = DataLoaderTest(args.test_path, tokenizer, args, batch_size=args.batch_size)

        logger.info('initializing estimator model')
        self.bert_model = BertForSequenceEncoder.from_pretrained(self.bert_pretrain)
        # bert_model = BertForSequenceClassification.from_pretrained(args.bert_pretrain)
#         if self.cuda:
#             self.bert_model = self.bert_model.to(self.device)
        self.model = inference_model(self.bert_model, max_len=self.max_len)
        self.model.load_state_dict(torch.load(self.checkpoint,map_location='cpu')['model'])
        if self.cuda:
            self.model = self.model.to(self.device)

        self.model.eval()

    def get_evidence_scores(self, inputs, use_context=True, topk=5):
        mod_inputs = []
        for inp in inputs:
            inp['original_claim'] = inp['claim']
            if use_context:
                inp['claim'] = ' [EOT] '.join(inp.get('context', [])) + ' [EOT] ' + inp['claim']
            mod_inputs.append(inp)
#         pprint(mod_inputs)
        validset_reader = DataLoaderLines(mod_inputs, self.tokenizer, max_len=self.max_len, deviceid=self.deviceid)

        all_results = get_model_predictions(validset_reader, self.model, topk=topk)

        return all_results



inputs = [{"id":123,"claim":"Really ? But despite it he has been the band 's sole constant member since 1985 .","context":["That is a pity, his heroin addiction,  He was very talented, he achieved worldwide success in the late 1980s.","rumor had it that the sound engineer Bob Clearmountain said Axl Rose would threaten to quit the band three times a week."],"evidence":[["Axl Rose",0,"He is the lead vocalist of the hard rock band Guns N' Roses, and has also been the band's sole constant member since its inception in 1985.",1],["Axl Rose",1,"Formed in 1968 as the Polka Tulk Blues Band, a blues rock band, the group went through line up changes, renamed themselves as Earth, broke up and reformed.",1],["Axl Rose",1,"Motörhead released 22 studio albums, 10 live recordings, 12 compilation albums, and five EPs over a career spanning 40 years.",1],["Axl Rose",1,"Founding bassist Steve Soto has been the sole constant member of the band since its inception, with singer Tony Reflex being in the group for all but one album.",1],["Axl Rose",1,"Nothing Records became largely defunct in 2004 due to a lawsuit by Reznor against John Malm.",1],["Axl Rose",1,"The band are often considered a precursor to the new wave of British heavy metal, which re-energised heavy metal in the late 1970s and early 1980s.",1]]}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--name', help='train path')
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', default='./output', help='path to output directory')
    parser.add_argument('--bert_pretrain', default='bert-base-cased')
    parser.add_argument('--checkpoint', default='retrieval_modelft_wowctxutt/model.best.pt')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
#     args_list = ['--checkpoint', 'retrieval_modelft_wowctxutt/model.best.pt']
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/test_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)

#     test_samples(args, tokenizer, model)
    sent_retrieval = Sentence_Retrieval_Bert(deviceid=1)
    all_results = sent_retrieval.get_evidence_scores(inputs)
    pprint(all_results)
