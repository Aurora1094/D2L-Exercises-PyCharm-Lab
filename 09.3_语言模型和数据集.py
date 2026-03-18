import random
import torch
from d2l import torch as d2l
import re

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

tokens=d2l.tokenize(read_time_machine())
corpus=[token for line in tokens for token in line]
# 双向映射表
vocab=d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for _, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
d2l.plt.tight_layout()
d2l.plt.show()

# 词元组合

# 二元词组
# 类似滑动窗口，得到所有zip组合
bigram_tokens=[pair for pair in zip(corpus[:-1], corpus[1:])]
# 将元组转换为字符串列表
# 这样 ('the', 'time') 就会变成 'the time'
# 原因：d2l.Vocab为了处理没在词表中出现过的词会默认带上 ['<unk>']
# combined = ['<unk>', ('the', 'time'), ('time', 'machine')]
# bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_str_tokens = [' '.join(pair) for pair in bigram_tokens]
bigram_vocab=d2l.Vocab(bigram_str_tokens)
print('-----二元-----')
print(bigram_vocab.token_freqs[:10])

# 三元词组
trigram_tokens=[
    triple for triple in zip(corpus[:-2],corpus[1:-1], corpus[2:])
]
# trigram_vocab=d2l.Vocab(trigram_tokens)
trigram_str_tokens = [' '.join(triple) for triple in trigram_tokens]
trigram_vocab = d2l.Vocab(trigram_str_tokens)
print('-----三元-----')
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.tight_layout()
d2l.plt.show()

# 数据采样(下一个数据是上一个数据的标签)【随机采样】
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    # 随即起始点
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 子序列数量
    num_subseqs = (len(corpus) - 1) // num_steps
    # 子序列起始下标的list
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    # 返回子序列
    def data(pos):
        return corpus[pos: pos + num_steps]

    # batch数量
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        # 标签是特征往后挪一位
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 构造人造数据，生成一个从0~34的序列
my_seq=list(range(35))
for X, Y in seq_data_iter_random(my_seq, 2, 5):
    print('X:',X,'\nY:', Y)

# 数据采样(下一个数据是上一个数据的标签)【顺序分区】
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # 把一长条 num_tokens 长的序列，横向切成了 batch_size 份，然后上下堆叠
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # 第 1 个 Batch 第一行的最后一个元素是 3，第 2 个 Batch 第一行的第一个元素正好是 4
        # 它们在物理内存上虽然被分到了两个 yield 出来的张量里，但在逻辑语意上是严丝合缝接上的
        # 后续RNN才可以体现
        yield X, Y

# 将上面的采样函数包装到迭代器
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random  # 随机采样函数
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential  # 顺序分区函数
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)