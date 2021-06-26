import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer
from NeZha_Chinese_PyTorch.model.modeling_nezha import NeZhaModel 
import os
import numpy as np
import config
from util import getLogger

class Model(nn.Module):
    def __init__(self, pretrain_model_path, hidden_size=768):
        super(Model, self).__init__()
        self.pretrain_model_path = pretrain_model_path
        self.bert = NeZhaModel.from_pretrained(self.pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.embed_size = hidden_size
        self.fc1 = nn.Linear(self.embed_size, 256)
        self.fc2 = nn.Linear(256,2)

    def forward(self, ids, segment):
        context = ids
        types = segment
        mask = torch.ne(context, 0)
        sequence_out, cls_out = self.bert(context, token_type_ids=types, attention_mask=mask)
        s = self.dropout1(cls_out)
        s = torch.tanh(s)
        s = self.fc1(s)
        x = self.dropout2(s)
        x = torch.tanh(x)
        logits = self.fc2(x)
        return logits

def read_dataset(path, pretrain_model_path, is_test=False):
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if is_test:
                text_a, text_b = line.split('\t')
            else:
                text_a, text_b, tgt = line.split('\t')
                tgt = int(tgt)
            src_a = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(text_a) + [SEP_TOKEN])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b) + [SEP_TOKEN])
            src = src_a + src_b
            seg = [0] * len(src_a) + [1] * len(src_b)
            if len(src) > seq_length:   
                src = src[: seq_length]
                seg = seg[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
            if is_test:
                dataset.append((src, seg))
            else:
                dataset.append((src, tgt, seg))
    return dataset

class TextDataset(Dataset):
    def __init__(self,data) -> None:
        self.data = data
     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), \
               torch.tensor(self.data[idx][1])


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
seq_length = 32
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
test_data = read_dataset(config.test_path, config.vocab_path, is_test=True)
test_loader = DataLoader(TextDataset(test_data), batch_size=256,shuffle=False)
pbar = tqdm(test_loader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def one_test():
    model = Model(config.pretrain_model_path)
    model.load_state_dict(torch.load(config.test_model_path))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
    preds_list = []
    with torch.no_grad():
        for text,token_type in pbar:
            model.eval()
            text = text.to(device)
            token_type = token_type.to(device)
            logits = model(text,token_type)
            preds = nn.Softmax(dim=1)(logits)[:,1]
            preds_list.append(preds.cpu().detach().numpy())
        preds_list =np.concatenate(preds_list)
        with open('result.tsv','w') as fp:
                fp.write('\n'.join(str(i) for i in preds_list))


def get_best_model_path(model_dir = './model/', prefix='nezhaodd'):
    from collections import defaultdict
    base_dir = model_dir
    files = os.listdir(base_dir)
    x = defaultdict(lambda :[])
    for file in files:
        if '_' not in file or prefix not in file:
            continue
        _, fold, score = file.split('_')
        score = score[:-4] # 去除末尾的'.pth'
        score = float(score)
        x[fold].append(score)
    paths = []
    for key,value in x.items():
        path = base_dir+prefix + '_' + key + '_' + '%.5f'%max(value) + '.pth'
        paths.append(path)
    return paths


def kfold_test():
    log = getLogger('test')
    model_paths  = get_best_model_path()
    log.info(model_paths)
    predict_final = []
    for fold in range(5):
        model_path = model_paths[fold]
        log.info(f'======fold:{fold} load model path from: {model_path}=====')
        test_loader = DataLoader(TextDataset(test_data), batch_size=256,shuffle=False)
        pbar = tqdm(test_loader)
        model = Model(config.pretrain_model_path)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) 
        preds_list = []
        with torch.no_grad():
            for text,token_type in pbar:
                model.eval()
                text = text.to(device)
                token_type = token_type.to(device)
                logits = model(text,token_type)
                preds = nn.Softmax(dim=1)(logits)[:,1]
                preds_list.extend(i.item() for i in preds.data)
        predict_final.append(np.array(preds_list))
    assert len(predict_final)==5, "kfold predict list error"
    predict_final = np.mean(predict_final, axis=0)

    with open('result.tsv','w') as fp:
            fp.write('\n'.join(str(i) for i in predict_final))

def softmax(x):
    output = np.zeros_like(x)
    if(x.ndim == 1):
        sumx = np.sum(np.exp(x))
        for i,x_i in enumerate(x):
            output[i] = np.exp(x_i) / sumx
    if(x.ndim == 2):
        sumx = np.sum(np.exp(x),axis=1)
        for i,x_i in enumerate(x):
            output[i] = np.exp(x_i) / sumx[i]
    
    return output

def kfoldtest_before():
    log = getLogger('test')
    model_paths  = get_best_model_path()
    print(model_paths)
    predict_final = []
    for fold in range(5):
        model_path = model_paths[fold]
        log.info(f'======fold:{fold} load model path from: {model_path}=====')

        model = Model(config.pretrain_model_path)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) 
        preds_list = []
        with torch.no_grad():
            for text,token_type in pbar:
                model.eval()
                text = text.to(device)
                token_type = token_type.to(device)
                logits = model(text,token_type).cpu().numpy()
                preds_list.append(logits)
        preds_list = np.concatenate(preds_list)
        predict_final.append(preds_list)
    assert len(predict_final)==5, "kfold predict list error"
    predict_final = np.mean(predict_final, axis=0)
    predict_final = softmax(predict_final)[:,1]

    with open('result.tsv','w') as fp:
            fp.write('\n'.join(str(i) for i in predict_final))

if __name__ == '__main__':
    one_test()
    #kfoldtest_before()
    #kfold_test()


