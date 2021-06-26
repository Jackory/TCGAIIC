# coding:utf-8
import time
import numpy as np

def process_pretrain_dataset(path, is_test=False):
    S = time.time()
    sentence = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if is_test:
                text_a, text_b = line.strip().split('\t')
            else:
                text_a, text_b, tgt = line.strip().split('\t')
            src = text_a + ' ' + text_b
            src2 = text_b + ' ' + text_a
            sentence.append(src)
            sentence.append(src2)
    cost = (time.time() - S) / 60.00
    print("Loading sentences from {}, Time cost:{:.2f}".format(path, cost))
    return sentence

def process_finetune_dataset(path):
    S = time.time()
    sentence = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            text_a, text_b, tgt = line.strip().split('\t')
            src = text_a + '\t' + text_b + '\t' + tgt
            src2 = text_b + '\t' + text_a + '\t' + tgt 
            sentence.append(src)
            sentence.append(src2)
    cost = (time.time() - S) / 60.00
    print("Loading sentences from {}, Time cost:{:.2f}".format(path, cost))
    return sentence

def write_result(origin_path, output_path, sentences):
    with open(origin_path, mode='w', encoding='utf-8') as f:
        for i in sentences:
            f.writelines(i+'\n')

    with open(origin_path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fd:
        for text in fr.readlines():
            if text.split():
                fd.write(text)
        print('输出成功....')

def presudo_data(test='test.tsv', prob = '../result.tsv'):
    test_lines = open(test,mode='r', encoding='utf-8').readlines()
    prob_lines = open(prob,mode='r', encoding='utf-8').readlines()
    assert(len(test_lines)==len(prob_lines))
    presudo_lines = []
    for line, p in zip(test_lines, prob_lines):
        line = line[:-1]
        p = p[:-1]
        p = float(p)
        if(p >= 0.50):
            presudo_lines.append(line + '\t' + '1')
        else:
            presudo_lines.append(line + '\t' + '0')
    with open('presodu_data.tsv', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(presudo_lines))    

def kfold_odd_data(datapath = 'train.tsv'):
    lines = open(datapath, 'r', encoding='utf-8').readlines()
    np.random.shuffle(lines)
    num_lines = len(lines)
    for fold in range(5):
        dev_start = int(fold * 0.2 * num_lines)
        dev_end = int((fold+1) * 0.2 * num_lines)
        dev_lines = lines[dev_start:dev_end]
        train_lines = lines[:dev_start] + lines[dev_end:]
        train_lines_odd = []
        for line in train_lines:
            text_a, text_b, tgt = line.strip().split('\t')
            src = text_a + '\t' + text_b + '\t' + tgt
            src2 = text_b + '\t' + text_a + '\t' + tgt 
            train_lines_odd.append(src)
            train_lines_odd.append(src2)
        with open(f'traindata_odd_fold{fold}.tsv', 'w', encoding='utf-8') as fr:
            for i in train_lines_odd:
                fr.writelines(i+'\n')
        with open(f'devdata_odd_fold{fold}.tsv', 'w', encoding='utf-8') as fd:
            for i in dev_lines:
                fd.writelines(i)

            

if __name__ == '__main__':
    # sent = []
    # #path = "pretrain_data.tsv"
    # path = "aug_train_data.tsv"
    # sent = process_finetune_dataset('train.tsv')
    # with open(path, 'w', encoding='utf-8') as fp:
    #     fp.write('\n'.join(sent)) 
    # kfold_odd_data()
    presudo_data()
