import argparse
import sys
import os
import numpy as np
import torch
import utils
import model
import logging
import matplotlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)


def process(args):
    #utils.gen_json(args.json_path)
    train_dataset = utils.ImgTextDataset(args, type='train')
    dev_dataset = utils.ImgTextDataset(args, type='dev')
    test_dataset = utils.ImgTextDataset(args, type='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    Net = model.JointEmbeddingNetwork(args).to(device)
    optimizer = torch.optim.SGD(Net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train(Net, args, optimizer, train_loader)

    max_accuracy = 0
    best_threshold = 0
    best_rate=0
    for i in range(1,10):
        threshold, accuracy = dev(Net, args, dev_loader, i / 10)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_threshold = threshold
            best_rate=i/10
    logging.info(f'(dev) Using rate: {best_rate} Using threshold: {best_threshold}')

    test(Net, test_dataset, test_loader, best_threshold)


def test(Net, test_dataset, test_loader, threshold):
    Net.eval()
    total, acc = 0, 0
    for i, (texts, images, matches) in enumerate(test_loader):
        texts = texts.to(device)
        images = images.to(device)
        matches = matches.to(device)

        texts, images = Net(texts, images)
        dis = model.JointEmbeddingNetwork.calc_dis(images, texts)
        for j, match in enumerate(matches):
            total += 1
            if match and dis[j][j].item() < threshold:
                acc += 1
            elif not match and dis[j][j].item() > threshold:
                acc += 1

        del texts, images, matches, dis
        torch.cuda.empty_cache()
    logging.info(f'(test) Single sentence matching accuracy: {acc / total}')
    TP, FP, TN, FN, = 0, 0, 0, 0
    for i, (text, image, match) in enumerate(test_dataset.test_data):
        text = text.to(device)
        image = image.unsqueeze(0).to(device)
        text, image = Net(text, image)

        dis = model.JointEmbeddingNetwork.calc_dis(image, text)
        pos = (dis < threshold).count_nonzero()
        neg = (dis > threshold).count_nonzero()
        predict = pos >= neg
        # predict = dis.mean() <= threshold

        if match:
            if predict:
                TP += 1
            else:
                FN += 1
        elif not match:
            if not predict:
                TN += 1
            else:
                FP += 1

        # logging.info(f'(test) {i + 1}:  actual: {match} predict: {predict}')
        logging.info(f'(test) {i + 1}: pos: {pos} neg: {neg} actual: {match} predict: {predict}')

        del text, image
        torch.cuda.empty_cache()
    logging.info(f'(test) Text-Image matching accuracy: {(TP + TN) / (TP + TN + FP + FN)}')
    logging.info(f'(test) TP: {TP} FN: {FN}')
    logging.info(f'(test) FP: {FP} TN: {TN}')

    return (TP + TN) / (TP + TN + FP + FN)


def dev(Net, args, dev_loader, threshold_rate):
    Net.eval()
    pos_dis = []
    neg_dis = []
    for i, (texts, images, matches) in enumerate(dev_loader):
        texts = texts.to(device)
        images = images.to(device)
        matches = matches.to(device)

        texts, images = Net(texts, images)
        dis = model.JointEmbeddingNetwork.calc_dis(images, texts)

        for j, match in enumerate(matches):
            if match:
                pos_dis.append(dis[j][j].item())
            else:
                neg_dis.append(dis[j][j].item())

        del texts, images, matches, dis
        torch.cuda.empty_cache()

    # plt.scatter(range(len(pos_dis)),pos_dis,marker='o')
    # plt.scatter(range(len(pos_dis),len(pos_dis)+len(neg_dis)),neg_dis,marker='^')
    # plt.show()

    pos_d = torch.Tensor(pos_dis).mean()
    neg_d = torch.Tensor(neg_dis).mean()
    logging.info(f'(dev) Positive pair mean distance: {pos_d}')
    logging.info(f'(dev) Negative pair mean distance: {neg_d}')

    # assert pos_dis < neg_dis
    threshold = pos_d * (1 - threshold_rate) + neg_d * threshold_rate
    logging.info(f'(dev) Select threshold: {threshold}')

    total = torch.Tensor(pos_dis).shape[0] + torch.Tensor(neg_dis).shape[0]
    acc = (torch.Tensor(pos_dis) <= threshold).count_nonzero() + (torch.Tensor(neg_dis) > threshold).count_nonzero()
    logging.info(f'(dev) Use rate: {threshold_rate} Single sentence matching accuracy: {acc / total}')

    return threshold, acc / total


def train(Net, args, optimizer, train_loader):
    Net.train()
    pos_dis = []
    neg_dis = []
    for epoch in range(args.num_epochs):
        for i, (texts, images, matches) in enumerate(train_loader):
            texts = texts.to(device)
            images = images.to(device)
            matches = matches.to(device)

            texts, images = Net(texts, images)
            loss = model.JointEmbeddingNetwork.loss(images, texts, matches, args)

            dis = model.JointEmbeddingNetwork.calc_dis(images, texts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, match in enumerate(matches):
                if match:
                    pos_dis.append(dis[j][j].item())
                else:
                    neg_dis.append(dis[j][j].item())

            if i % 10 == 0 or i + 1 == len(train_loader):
                logging.info(
                    f'Epoch: [{epoch + 1}/{args.num_epochs}] Step: [{i + 1}/{len(train_loader)}] Loss: {loss.data.item()}')

            del texts, images, matches, loss, dis
            torch.cuda.empty_cache()
    # plt.scatter(range(len(pos_dis)),pos_dis,marker='o')
    # plt.scatter(range(len(pos_dis),len(pos_dis)+len(neg_dis)),neg_dis,marker='^')
    # plt.show()
    pos_dis = torch.Tensor(pos_dis).mean()
    neg_dis = torch.Tensor(neg_dis).mean()
    logging.info(f'(train) Positive pair mean distance: {pos_dis}')
    logging.info(f'(train) Negative pair mean distance: {neg_dis}')


if __name__ == '__main__':
    argser = argparse.ArgumentParser()

    argser.add_argument('--img_size', type=int, default=512, help='Height and width of images.')
    argser.add_argument('--img_dim', type=int, default=4096, help='The dimension of the image vector.')
    argser.add_argument('--word_dim', type=int, default=50, help='The dimension of the word vector.')
    argser.add_argument('--num_words', type=int, default=120, help='Number of words spliced in each sentence.')
    argser.add_argument('--threshold_rate', type=float, default=0.7, help='Threshold_rate.')
    argser.add_argument('--wordvec_file', type=str, default='En_vectors.txt', help='Path of the word vectors file.')
    argser.add_argument('--root', type=str, default='data', help='Root path.')
    argser.add_argument('--json_path', type=str, default=os.path.join('data', 'data.json'),
                        help='The path of json file.')
    argser.add_argument('--train_path', type=str, default=os.path.join('data', 'train.json'),
                        help='The path of train json file.')
    argser.add_argument('--dev_path', type=str, default=os.path.join('data', 'dev.json'),
                        help='The path of dev json file.')
    argser.add_argument('--test_path', type=str, default=os.path.join('data', 'test.json'),
                        help='The path of test json file.')
    argser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension of every branches.')
    argser.add_argument('--output_dim', type=int, default=512, help='Output dimension of every branches.')
    argser.add_argument('--batch_size', type=int, default=400, help='Batch size of the train process.')
    argser.add_argument('--margin', type=float, default=0.1, help='Margin of the distance.')
    argser.add_argument('--lambda1', type=float, default=2, help='Argument lambda1 of the paper.')
    argser.add_argument('--k', type=int, default=50,
                        help='\'Most top K violations of each relevant constraint\' in paper')
    argser.add_argument('--num_epochs', type=int, default=30, help='Number of training rounds.')
    argser.add_argument('--lr', type=float, default=1, help='Learning rate of SGD.')
    argser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight_decay of SGD.')
    argser.add_argument('--momentum', type=float, default=0.9, help='Momentum of SGD.')

    args, unparsed = argser.parse_known_args()
    process(args)
