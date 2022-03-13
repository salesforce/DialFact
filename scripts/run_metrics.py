import argparse
import os
import sys
import jsonlines
from tqdm import tqdm
import argparse
import json
import random
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix


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

scores_list = [ 'fever_score']

def read_json_scores(args, input_examples):
    # scores_list = ['dnli_score', 'contradict_score', 'fever_score', 'colloq_fever_score', 'corefbertfever_score']
    # scores_list = ['fever_score', 'colloq_fever_score', 'corefbertfever_score', 'augwow_fever_score']
    # scores_list = ['augwow_fever_score']
    score_map = {
        'dnli_score':{
            '0': 'REFUTES',
            '1': 'NOT ENOUGH INFO',
            '2': 'SUPPORTS'
        },
        'contradict_score':{
            '1': 'REFUTES',
            '0': 'SUPPORTS'
        },
        'fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
        'colloq_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
        'augwow_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
        'corefbertfever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
        'colloqclaimonly_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
    }

    models_dict = dict()
    all_labels = []
    all_srlabels = []
    all_nonp_labels = []
    label_counter = defaultdict(int)
    typelabel_counter = defaultdict(int)
    print('read_json_scores', len(input_examples))
    for k in input_examples[0]:
        if '_score' in k:
            scores_list = [k]
    for s in scores_list:
        models_dict[s] = dict()
        models_dict[s]['all_preds'] = []
        models_dict[s]['all_sr_preds'] = []
        models_dict[s]['all_nonp_preds'] = []
    for i, example in enumerate(input_examples):
        response = example['response']
            # if len(response)<10:
            #     continue
            # r_category_agreement = example['category_agreement']
        r_response_label = example['response_label']
        r_type_label = example['type_label']

        # if 'written' not in example['data_type']:
        #     continue
        all_labels.append(r_response_label)
        typelabel_counter[r_type_label] += 1
        if r_response_label in ['SUPPORTS', 'REFUTES']:  # , 'NOT ENOUGH INFO', 'conflict']:
            all_srlabels.append(r_response_label)
        if r_type_label in ['factual']:
            all_nonp_labels.append(r_response_label)
            label_counter[r_response_label] += 1
        for type_score in scores_list:
            r_scores = example[type_score]
            max_index = r_scores.index(max(r_scores))
            label_pred = score_map[type_score][str(max_index)]
            models_dict[type_score]['all_preds'].append(label_pred)

            if r_response_label in ['SUPPORTS', 'REFUTES']:
                models_dict[type_score]['all_sr_preds'].append(label_pred)
            if r_type_label in ['factual']:
                models_dict[type_score]['all_nonp_preds'].append(label_pred)


    print(len(all_labels), 'len all_labels')
    print(len(all_srlabels), 'len all_srlabels')
    print(len(all_nonp_labels), 'len all_nonp_labels')
    print(label_counter)
    print(typelabel_counter)
    # print(len(models_dict['dnli_score']['all_preds']))

    model_f1_dict_all = dict()
    model_f1_dict_nonp = dict()
    for model in scores_list:
        # print(model)
        cr_all = classification_report(all_labels, models_dict[model]['all_preds'], digits=4, output_dict=True)
        model_f1_dict_all[model] = cr_all['macro avg']['f1-score']

    for model in scores_list:
        # print(model)
        # print(classification_report(all_nonp_labels, models_dict[model]['all_nonp_preds'], digits=4))
        cr_nonp = classification_report(all_nonp_labels, models_dict[model]['all_nonp_preds'], digits=4, output_dict=True)
        model_f1_dict_nonp[model] = cr_nonp['macro avg']['f1-score']



    print('\nResults including all:')
    for model in scores_list:
        print(model)
        print(classification_report(all_labels, models_dict[model]['all_preds'], digits=4))
        print(confusion_matrix(all_labels, models_dict[model]['all_preds'], labels=[ "REFUTES", "SUPPORTS"]))

    print('\nResults only factual:')
    for model in scores_list:
        print(model)
        print(classification_report(all_nonp_labels, models_dict[model]['all_nonp_preds'], digits=4))
        print(confusion_matrix(all_nonp_labels, models_dict[model]['all_nonp_preds'], labels=[ "REFUTES", "SUPPORTS"]))


    print('\nACCURACY including all:')
    for model in scores_list:
        accuracy = get_accuracy(all_labels, models_dict[model]['all_preds'])
        f1 = model_f1_dict_all[model]*100
        print(f'{model:<15}', '\t', '{0:.3f}'.format(accuracy), '\t', '{0:.3f}'.format(f1))


    print('\nACCURACY only factual:')
    for model in scores_list:
        accuracy_nonp = get_accuracy(all_nonp_labels, models_dict[model]['all_nonp_preds'])
        f1 = model_f1_dict_nonp[model]*100
        # print(f'{model:<15}', '\t', '{0:.3f}'.format(accuracy_nonp))
        print(f'{model:<15}', '\t', '{0:.3f}'.format(accuracy_nonp), '\t', '{0:.3f}'.format(f1))


def read_json_notsupport(args, input_examples):
    # scores_list = ['dnli_score', 'contradict_score', 'fever_score', 'colloq_fever_score', 'corefbertfever_score']
    # scores_list = ['fever_score', 'colloq_fever_score', 'corefbertfever_score', 'augwow_fever_score']
    # scores_list = ['augwow_fever_score']

    score_map = {
        'dnli_score':{
            '0': 'REFUTES',
            '1': 'REFUTES',
            '2': 'SUPPORTS'
        },
        'contradict_score':{
            '0': 'SUPPORTS',
            '1': 'REFUTES'
        },
        'fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'REFUTES'
        },
        'colloq_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'REFUTES'
        },
        'augwow_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'NOT ENOUGH INFO'
        },
        'corefbertfever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'REFUTES'
        },
        'colloqclaimonly_fever_score': {
            '0': 'SUPPORTS',
            '1': 'REFUTES',
            '2': 'REFUTES'
        },
    }

    models_dict = dict()
    all_labels = []
    all_srlabels = []
    all_nonp_labels = []
    label_counter = defaultdict(int)
    typelabel_counter = defaultdict(int)
    print('\n\nread_json_notsupport', len(input_examples))
    for k in input_examples[0]:
        if '_score' in k:
            scores_list = [k]

    for s in scores_list:
        models_dict[s] = dict()
        models_dict[s]['all_preds'] = []
        models_dict[s]['all_sr_preds'] = []
        models_dict[s]['all_nonp_preds'] = []
    for i, example in enumerate(input_examples):
        response = example['response']
            # if len(response)<10:
            #     continue
        r_response_label = example['response_label']

        if r_response_label=='NOT ENOUGH INFO':
            r_response_label = 'REFUTES'

        r_type_label = example['type_label']
        all_labels.append(r_response_label)
        typelabel_counter[r_type_label] += 1
        if r_response_label in ['SUPPORTS', 'REFUTES']:  # , 'NOT ENOUGH INFO', 'conflict']:
            all_srlabels.append(r_response_label)
        if r_type_label in ['factual']:
            all_nonp_labels.append(r_response_label)
            label_counter[r_response_label] += 1
        for type_score in scores_list:
            r_scores = example[type_score]
            max_index = r_scores.index(max(r_scores))
            label_pred = score_map[type_score][str(max_index)]
            if label_pred=='NOT ENOUGH INFO':
                label_pred = 'REFUTES'
            models_dict[type_score]['all_preds'].append(label_pred)

            if r_response_label in ['SUPPORTS', 'REFUTES']:
                models_dict[type_score]['all_sr_preds'].append(label_pred)
            if r_type_label in ['factual']:
                models_dict[type_score]['all_nonp_preds'].append(label_pred)


    print(len(all_labels), 'len all_labels')
    print(len(all_srlabels), 'len all_srlabels')
    print(len(all_nonp_labels), 'len all_nonp_labels')
    print(label_counter)
    print(typelabel_counter)
    # print(len(models_dict['dnli_score']['all_preds']))


    print('\nResults including all:')
    for model in scores_list:
        print(model)
        print(classification_report(all_labels, models_dict[model]['all_preds'], digits=4))
        print(confusion_matrix(all_labels, models_dict[model]['all_preds'], labels=[ "REFUTES", "SUPPORTS"]))

    print('\nResults only factual:')
    for model in scores_list:
        print(model)
        print(classification_report(all_nonp_labels, models_dict[model]['all_nonp_preds'], digits=4))
        print(confusion_matrix(all_nonp_labels, models_dict[model]['all_nonp_preds'], labels=[ "REFUTES", "SUPPORTS"]))

    print('\nACCURACY including all:')
    for model in scores_list:
        accuracy_p = get_accuracy(all_labels, models_dict[model]['all_preds'])
        print(f'{model:<15}', '\t', '{0:.3f}'.format(accuracy_p))

    print('\nACCURACY only factual:')
    for model in scores_list:
        accuracy_nonp = get_accuracy(all_nonp_labels, models_dict[model]['all_nonp_preds'])
        print(f'{model:<15}', '\t', '{0:.3f}'.format(accuracy_nonp))


def get_accuracy(lables, preds):
    assert len(preds) == len(lables)
    count_correct = 0
    for i, l in enumerate(lables):
        if preds[i]==l:
            count_correct+=1
    acc = count_correct/len(preds) * 100

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='path to the file containing the evaluation data', required=True)
    parser.add_argument('-o', '--preds_file', type=str, help='output file to save the results')
    parser.add_argument('--writing', action='store_true', default=False, help='')
    parser.add_argument('--output_folder', type=str, help='output file to save the results', default='out_merged_scores/')
#     parser.add_argument('-append', action='store_true', help='allow append to previous run', default=False)

    args = parser.parse_args()
    if args.preds_file is None:
        args.preds_file = args.input_file


    INPUT_FILE_NAME = args.input_file

    input_examples = get_json_lines(INPUT_FILE_NAME)  # [:5]
    print(INPUT_FILE_NAME)
    print(len(input_examples))

    read_json_scores(args, input_examples)
    read_json_notsupport(args, input_examples)

# python get_scores_metrics.py -i $FILE
