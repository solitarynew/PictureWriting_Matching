import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Branch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.layer2=nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.normalize(x)
        return x


class JointEmbeddingNetwork(nn.Module):
    def __init__(self, args):
        super(JointEmbeddingNetwork, self).__init__()
        self.args = args
        self.text_branch = Branch(args.word_dim*args.num_words, args.hidden_dim, args.output_dim)
        self.img_branch = Branch(args.img_dim, args.hidden_dim, args.output_dim)

    def forward(self, texts, images):
        return self.text_branch(texts), self.img_branch(images)

    @staticmethod
    def calc_dis(a, b):
        """
            shape of a : (batch,output_dim)
            shape of b : (batch,output_dim)
            Return a Tensor which shape is (batch,batch) and [i,j] means the distance between a[i] and b[j]
        """
        a_square = torch.sum(a * a, 1).view(-1, 1)
        b_square = torch.sum(b * b, 1).view(1, -1)
        return torch.sqrt(a_square - 2 * torch.mm(a, b.transpose(0, 1)) + b_square + 1e-4).to(device)

    @staticmethod
    def loss(images, texts, matches, args):
        k=min(args.k,matches.shape[0])
        dis = JointEmbeddingNetwork.calc_dis(images, texts)

        pos_image = torch.zeros((matches.shape[0], 1)).to(device)
        for i, match in enumerate(matches):
            if match:
                pos_image[i] = dis[i, i]
        image_loss = torch.clamp(args.margin + pos_image - dis, 0, 1e6).topk(k)[0]
        image_loss = image_loss.mean(1)
        image_loss = torch.masked_select(image_loss, matches).mean()

        pos_text = torch.zeros((matches.shape[0])).to(device)
        for i, match in enumerate(matches):
            if match:
                pos_text[i] = dis[i, i]
        text_loss = torch.clamp(args.margin + pos_text - dis, 0, 1e6).topk(k, dim=0)[0]
        text_loss = text_loss.mean(0)
        text_loss = torch.masked_select(text_loss, matches).mean()

        return image_loss + args.lambda1 * text_loss
