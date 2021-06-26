# 小布助手对话短文本语义匹配
## 依赖
transformers安装4.x版
python >= 3.7

## 目录
- data/：包含生成vocab.txt, 预训练数据集的.py和已生成的数据集
- model/: pretrain+finetune生成的模型权重的目录
- NeZha_Chinese_PyTorch/: torch版本NezhaModel实现，github clone 而来
- nezha-xxx/: Nezha不同版本的官方预训练权重
- pretrain/: 包含pretrain.py和pretrain生成的模型权重，其中trainer_state.json可以看到训练日志

## 运行
- 先确认工作目录在主目录下
- 加载NeZha预训练：`python ./pretrain/train_nezha.py`
- finetune微调：`python finetune.py`；若K折交叉验证，则`python koldtrain.py`
- 生成.tsv：`python test.py`

## 方法
- 1-3 gram混合MLM预训练
- pretrain数据集：两句话中间用空格分开，使用对偶(两句话对调)增强数据集，size=(125000*2)
- 开始固定学习率6e-5，当验证集效果降低时，学习率开始递减(比transformers默认学习率从初值线性递减效果更好)
- finetune：两层神经网络+对抗训练(FGM)

