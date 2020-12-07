# -*- coding:utf-8 -*-
""" -------------import------------- """
import argparse
import math
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from models import *
from utils import *


""" -------------build dataloader------------- """


def build_trigram(doc, word2idx):
    """
    三元语法，用前两个词预测下一词语
    :param doc:
    :param word2idx:
    :return:
    """
    trigrams = [([doc[i], doc[i + 1]], doc[i + 2]) for i in range(len(doc) - 2)]
    input_ids = []
    target_ids = []
    for context, target in trigrams:
        ids = torch.tensor([word2idx[w] for w in context], dtype=torch.long)
        input_ids.append(ids)
        ids = torch.tensor([word2idx[target]], dtype=torch.long)
        target_ids.append(ids)
    return input_ids, target_ids


class MyDataset(Dataset):
    def __init__(self, input_ids, target_ids):
        """

        :param input_ids: [tensor]
        :param target_ids: [tensor]
        """
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.len = len(input_ids)

    def __getitem__(self, index):
        """
        重载
        :param index:
        :return: tensor，tensor
        """
        return self.input_ids[index], self.target_ids[index]

    def __len__(self):
        return self.len


""" -------------train------------- """


def train(model, dataloader, optimizer, criterion, scheduler=None, updates=0):
    """

    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :param scheduler:
    :param updates: 训练的batch的数量
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('use', device)
    model.train()
    model.to(device)
    all_losses = []
    for _ in tqdm(range(1, config.n_epochs + 1)):
        hidden = None
        for batch in dataloader:
            # batch list of 2 tensor [tensor[batch_size, single_input_size], tensor[batch_size, single_target_size]]
            output = model(batch[0].to(device), hidden)
            loss = criterion(output, batch[1].view(-1))  # 第二个参数需要是1D
            model.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
            updates += 1

            # 每隔interval次在验证集上进行一次评估，保留更好的模型
            # if updates % config.eval_interval == 0:
            #     print(evaluating after {updates} updates...\r')
            #     score = evaluate(model, dataloader, epoch, updates)
            #     if score >= max_bleu:
            #         save_model(config.save_path + str(score) + '_checkpoint.pt', model, optimizer, epoch, loss)
            #         max_bleu = score
            # if updates % config.print_every == 0:
            #     print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / config.n_epochs * 50, loss))
    save_model('checkpoints/final_checkpoint.pt', model, optimizer, config.n_epochs + 1, all_losses)


""" -------------eval------------- """


def evaluate(model, word2idx, prime_str='萧炎', predict_len=100, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_ids = torch.tensor([word2idx[w] for w in prime_str[-2:]], dtype=torch.long)
    input_ids = input_ids.view(1, 2).to(device)
    hidden = None
    for p in range(predict_len):
        output = model(input_ids, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted word to string and use as next input
        predicted_word = list(word2idx.keys())[list(word2idx.values()).index(top_i)]
        prime_str += "" + predicted_word
        input_ids = torch.tensor([word2idx[w] for w in prime_str[-2:]], dtype=torch.long)
        input_ids = input_ids.view(1, 2).to(device)
    return prime_str
    # model = RNN(*args, **kwargs)
    # optimizer = TheOptimizerClass(*args, **kwargs)
    #
    # device = torch.device("cuda")
    # checkpoints = torch.load(path, map_location="cuda:0")
    # model.load_state_dict(checkpoints['model_state_dict'])
    # optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    # epoch = checkpoints['epoch']
    # loss = checkpoints['loss']
    # model.to(device)
    # model.eval()
    # # - or -
    # model.train()


""" -------------save model------------- """


def save_model(path, model, optimizer, epoch, loss):
    """
    Saving & Loading a General Checkpoint for Inference and/or Resuming Training
    :param path:
    :param model:
    :param optimizer:
    :param epoch:
    :param loss:
    :return:
    """
    # 如果使用并行的话使用的是model.module.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


""" -------------args------------- """


class AttrDict(dict):
    """
    https://stackoverflow.com/questions/2641484/class-dict-self-init-args?answertab=votes#tab-top
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_args():
    """

    :return: args 全局参数，config 模型参数
    """
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('-c', '--config_file', required=True, help='模型配置文件位置')
    # parser.add_argument('-print_every', default=50,
    #                     help="print_every")
    # parser.add_argument('-plot_every', default='20', type=str,
    #                     help="plot_every ")
    # parser.add_argument('-n_epochs', default='graph2seq', type=str,
    #                     choices=['seq2seq', 'graph2seq', 'bow2seq', 'h_attention'])
    # parser.add_argument('-gpus', default=[1], type=int,
    #                     help="Use CUDA on the listed devices.")

    global_args = parser.parse_args()
    model_config = AttrDict(yaml.safe_load(open(global_args.config_file, 'r')))
    return global_args, model_config


args, config = parse_args()
print(config)

""" -------------main------------- """


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    text_list = read_data('datasets/doupo/train.txt')
    doc = clean(text_list, 'datasets/stopwords.txt')
    vocab = Vocab(doc)
    input_ids, target_ids = build_trigram(doc, vocab.word2idx)
    print(f'total {len(input_ids)} train data')

    my_dataset = MyDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset=my_dataset, batch_size=32, shuffle=True)
    model = RNN(len(vocab.word2idx), config.hidden_size, len(vocab.word2idx), config.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler =
    criterion = nn.CrossEntropyLoss()
    train(model, dataloader, optimizer, criterion)
    output = evaluate(model, vocab.word2idx, predict_len=1000)
    with open('./output.txt', 'w') as f:
        f.write(output)
    print(output[:100])


""" -------------start------------- """
if __name__ == '__main__':
    main()
