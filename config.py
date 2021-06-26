import transformers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
seed = 2021
transformers.set_seed(seed)


max_seq_length = 32
min_freq = 3


#train_path = 'data/aug_train_data.tsv'
train_path = 'data/train.tsv'
test_path = 'data/gaiic_track3_round1_testB_20210317.tsv'
vocab_path = './data/vocab.txt'
aug_data_path = './data/pretrain_data.tsv'

#pretrain
# load_pretrain_json_path = "./nezha-cn-base/nezha-cn-base/config.json"
# load_pretrain_model_path = "./nezha-cn-base/nezha-cn-base/"
# pretrain
load_pretrain_json_path = "./pretrain/model_nezhabase_ngram/checkpoint-23250/config.json"
load_pretrain_model_path = "./pretrain/model_nezhabase_ngram/checkpoint-23250/"
pretrain_output_path = './pretrain/model_nezhabase_ngram_wei/'


# fine_tune
num_epochs = 10
batch_size = 256
learning_rate = 2e-5
train_size = 0.9
#pretrain_model_path = './pretrain/model_nezha2/nezhaloss_0.2991/'
pretrain_model_path = './pretrain/model_nezhabase_ngram_wei/checkpoint-47000'
finetune_output_path = 'finetune/nezha_model/'

# test
test_model_path = 'model/nezha0.97811.pth'
use_attack = True

