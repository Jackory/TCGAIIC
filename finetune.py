# coding:utf-8
import random
from random import shuffle
import torch
import torch.nn as nn
from transformers import BertTokenizer
from NeZha_Chinese_PyTorch.model.modeling_nezha import NeZhaModel 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import os
import config
from util import getLogger


seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
#torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
log = getLogger('finetune')

'''
可以自行修改params
'''
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
seq_length = 32
batch_size= 256
os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'

tokenizer = BertTokenizer.from_pretrained(config.vocab_path)

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

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='word_embeddings'):
    #def attack(self, epsilon=1., emb_name='embeds'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name in name): 
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name in name): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
               torch.tensor(self.data[idx][1]), \
                torch.tensor(self.data[idx][2])

train = read_dataset(config.train_path, config.vocab_path) #TODO: 注意修改路径
presudo = read_dataset('./data/presodu_data.tsv', config.vocab_path)
shuffle(train)
num_lines = int(0.9*len(train))
train_loader = DataLoader(TextDataset(train[:num_lines] + presudo), batch_size=batch_size,shuffle=True)
dev_loader = DataLoader(TextDataset(train[num_lines:]), batch_size=batch_size,shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(config.pretrain_model_path, hidden_size=768) # TODO: 注意修改路径
bert_params = list(map(id, model.bert.parameters()))
base_params = filter(lambda p: id(p) not in bert_params,model.parameters())

    
optim = torch.optim.AdamW([{'params': base_params,'lr':6e-5},{'params': model.bert.parameters()}], lr=2e-5)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.8, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max',verbose=True)
#optim = torch.optim.Adam([{'params': base_params,'lr':6e-5},{'params': model.bert.parameters()}], lr=2e-5)
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) 
if(config.use_attack):
    attacker = FGM(model)

num_epochs = 4
log.info("num_epochs:{}".format(num_epochs))

def evaluate():
    model.eval()
    labels_list = []
    preds_list = []
    pbar = tqdm(dev_loader)
    with torch.no_grad():
        for text,label,token_type in pbar:
            text = text.to(device)
            label = label.to(device)
            token_type = token_type.to(device)
            logits = model(text,token_type)
            #preds = logits[:,1] / (logits.sum(axis=1) + 1e-8)
            preds = nn.Softmax(dim=1)(logits)[:,1]
            preds_list.append(preds.cpu().detach().numpy())
            labels_list.append(label.cpu().detach().numpy())
        preds_list = np.concatenate(preds_list)
        labels_list = np.concatenate(labels_list)
        auc_score = roc_auc_score(labels_list,preds_list)
        log.info(f'auc_pred:{auc_score}')
        return auc_score 

def train():
    
    #optim = torch.optim.AdamW(model.parameters(),lr=2e-5)
    losses_list = []
    best_auc = 0.5
    for epoch in range(num_epochs):
        log.info(f"==============epoch: {epoch}===============")
        pbar = tqdm(train_loader)
        model.train()
        step = 0
        for text,label,token_type in pbar:
            text = text.to(device)
            label = label.to(device)
            token_type = token_type.to(device)
            logits = model(text,token_type)
            loss = nn.CrossEntropyLoss()(logits, label.view(-1))
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            losses_list.append(loss.cpu().detach().numpy())
            optim.zero_grad()
            loss.backward()

            if(config.use_attack):
                attacker.attack()
                logits = model(text,token_type)
                loss_adv = nn.CrossEntropyLoss()(logits,label.view(-1))
                loss_adv.backward()
                attacker.restore()
            optim.step()
            pbar.set_description(f'epoch:{epoch}, loss:{np.mean(losses_list):.4f}')   
            if step%100 == 0:
                log.info('----start eval ----')
                cur_auc = evaluate()
                if(cur_auc > best_auc and cur_auc > 0.97):
                    best_auc = cur_auc
                    torch.save(model.module.state_dict(),f'model/nezha{best_auc:.5f}.pth', _use_new_zipfile_serialization=False)
                model.train()
                scheduler.step(cur_auc)
            step+=1
        log.info('----start eval ----')
        cur_auc = evaluate()
        if(cur_auc > best_auc and cur_auc > 0.97):
            best_auc = cur_auc
            torch.save(model.module.state_dict(),f'model/nezha{best_auc:.5f}.pth', _use_new_zipfile_serialization=False)
        #torch.save(model.state_dict(),f'model/nezha{best_auc:.3f}.pth', _use_new_zipfile_serialization=False)
    log.info("========================== {} ========================".format(best_auc))

if __name__ == '__main__':
    train()
