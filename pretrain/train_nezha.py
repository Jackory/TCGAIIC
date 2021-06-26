from transformers import BertTokenizer,\
    DataCollatorForLanguageModeling,\
    Trainer, TrainingArguments
from NeZha_Chinese_PyTorch.model.modeling_nezha import NeZhaForMaskedLM 
from NeZha_Chinese_PyTorch.model.configuration_nezha import NeZhaConfig
import os 
from util import LineByLineTextDataset, DataCollatorForLMngram
import config 

nezha_config = NeZhaConfig.from_json_file(config.load_pretrain_json_path)
model = NeZhaForMaskedLM.from_pretrained(config.load_pretrain_model_path, config=nezha_config)


tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
lines = open(config.aug_data_path, encoding='utf-8').readlines()
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    lines=lines,
    mode=0 
)

data_collator = DataCollatorForLMngram(
    tokenizer=tokenizer, mlm=True, plm_probability=0.15, max_span_length=2
)

if not os.path.exists(config.pretrain_output_path):
    os.makedirs(config.pretrain_output_path)

training_arg = TrainingArguments(
    output_dir = config.pretrain_output_path,
    overwrite_output_dir= True,
    num_train_epochs=50,
    per_device_train_batch_size= 64,
    save_total_limit= 10,
    logging_steps= 200,
    save_steps= 1000,
    prediction_loss_only = True,
    learning_rate= 1e-5,
    warmup_ratio= 0.1, 
    lr_scheduler_type = 'constant',
)

trainer = Trainer(
    model = model,
    args=training_arg,
    data_collator=data_collator,
    train_dataset=dataset,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model(config.pretrain_output_path)

