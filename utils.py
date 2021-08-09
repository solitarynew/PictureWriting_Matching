from functools import reduce

import nltk
import os
import json
import numpy as np
from nltk.corpus import stopwords
import torch
from PIL import Image
import torchvision.models as models
from torchvision.transforms import transforms
import logging
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)


def gen_json(file_name):
    dir_path = os.path.dirname(file_name)
    train_path = os.path.join(dir_path, 'train.json')
    dev_path = os.path.join(dir_path, 'dev.json')
    test_path = os.path.join(dir_path, 'test.json')
    train_data = dict()
    dev_data = dict()
    test_data = dict()
    with open(file_name, 'r') as f:
        json_data = json.load(f)
        for key, value in json_data.items():
            r = random.randint(1, 10)
            if r <= 7:
                train_data[key] = value
            elif r == 8:
                dev_data[key] = value
            else:
                test_data[key] = value

    with open(train_path, 'w') as trainf, open(dev_path, 'w') as devf, open(test_path, 'w') as testf:
        json.dump(train_data, trainf, indent=1)
        json.dump(dev_data, devf, indent=1)
        json.dump(test_data, testf, indent=1)


class ImgTextDataset(torch.utils.data.Dataset):
    def __init__(self, args, type='train'):
        ImgTextDataset.word_embedding_cache = None
        self.args = args
        self.type = type
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
        self.vgg19 = models.vgg19_bn(pretrained=True).to(device)
        self.vgg19.classifier = torch.nn.Sequential(
            *list(self.vgg19.children())[-1][:5])
        self.word2idx, self.word2embedding = self.load_wordvector(args.wordvec_file)
        self.stopwords = set(stopwords.words('english'))

        self.data = []
        self.test_data = []
        select = {'train': args.train_path, 'dev': args.dev_path, 'test': args.test_path}
        with open(select[type], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for i, (key, value) in enumerate(json_data.items()):
                text = self.load_text(os.path.join(self.args.root, key))
                match = bool(value['match'])
                imgs = [self.load_image(os.path.join(self.args.root, img)) for img in value['images']]
                # How to build
                # for sent in text:
                #     for img in imgs:
                #         self.data.append([sent, img, match])
                img = reduce(lambda x, y: x + y, imgs) / len(imgs)
                # s=reduce(lambda x, y: x + y, text) / len(text)
                # self.data.append([s, img, match])
                for sent in text:
                    self.data.append([sent, img, match])
                self.test_data.append([text, img, match])
                # self.data.append([text[0], imgs[0], match])
                if (1 + i) % 100 == 0 or i == len(json_data) - 1:
                    logging.info(f"({type}) image and text {i + 1}/{len(json_data)} have been loaded.")
        logging.info(f"({type}) All images and texts have been loaded.")

    def load_wordvector(self, file_name):
        if ImgTextDataset.word_embedding_cache is not None:
            return ImgTextDataset.word_embedding_cache
        word2idx = {}
        word2embedding = {}
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                word, vec = line.split()
                word2idx[word] = i + 1
                vec = [float(val) for val in vec.split(',')]
                word2embedding[word] = vec
        logging.info(f"({self.type}) All embedding vectors have been loaded.")
        ImgTextDataset.word_embedding_cache = [word2idx, word2embedding]
        return ImgTextDataset.word_embedding_cache

    def load_image(self, file_name):
        img = Image.open(file_name).convert('RGB')
        img = self.trans(img)
        img = img.unsqueeze(dim=0).to(device)
        return self.vgg19(img).data[0]

    def load_text(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            str = f.read().lower()
            sentence = nltk.tokenize.sent_tokenize(str)
            sentence = [nltk.tokenize.word_tokenize(sent) for sent in sentence]
            sentence = [[w for w in sent if ((not w in self.stopwords) and (
                    w in self.word2idx.keys()))] for sent in sentence if sent]
            # sentence = [np.array([self.word2embedding[w]
            #                       for w in sent]).mean(axis=0) for sent in sentence]
            # sentence = np.array(sentence)
            # assert sentence.shape[1] == self.args.word_dim
            # return torch.Tensor(sentence).to(device)
            sentence = [torch.Tensor([self.word2embedding[w]
                                      for w in sent]) for sent in sentence]
            sentence = [torch.cat([sent[i % sent.shape[0]] for i in range(self.args.num_words)], 0).numpy() for sent in
                        sentence]
            return torch.Tensor(sentence).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
