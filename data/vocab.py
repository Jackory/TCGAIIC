import config

with open(config.train_path) as fp:
    train_data = fp.readlines()
train_texts = [line.strip()[:-1].split('\t')[:-1] for line in train_data]
train_texts = [[sa.split(), sb.split()] for sa, sb in train_texts]
with open(config.test_path) as fp:
    test_data = fp.readlines()
test_texts = [line.strip()[:-1].split('\t') for line in test_data]
test_texts = [[sa.split(), sb.split()] for sa, sb in test_texts]
texts = train_texts + test_texts
counter = {}
for sa, sb in texts:
    for w in sa:
        counter[w] = counter.get(w, 0) + 1
    for w in sb:
        counter[w] = counter.get(w, 0) + 1
counter = {int(k): v for k, v in counter.items() if v >= config.min_freq}
with open(config.vocab_path, 'w') as fp:
    fp.writelines(["[PAD]\n", "[UNK]\n", "[CLS]\n", "[SEP]\n", "[MASK]\n"])
    for k in sorted(counter.items(),key=lambda v:(v[1]),reverse=True):
        fp.write(str(k[0]) + '\n')