import torch
import torch.nn as nn


class GMMNnetwork(nn.Module):
    def __init__(
        self,
        noise_dim,
        embed_dim,
        hidden_size,
        feature_dim,
        semantic_reconstruction=False,
    ):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.4))
            return layers

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(noise_dim + embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(noise_dim + embed_dim, feature_dim)

        self.model.apply(init_weights)
        self.semantic_reconstruction = semantic_reconstruction
        if self.semantic_reconstruction:
            self.semantic_reconstruction_layer = nn.Linear(
                feature_dim, noise_dim + embed_dim
            )

    def forward(self, embd, noise):
        features = self.model(torch.cat((embd, noise), 1))
        if self.semantic_reconstruction:
            semantic = self.semantic_reconstruction_layer(features)
            return features, semantic
        else:
            return features


class GMMNLoss:
    def __init__(self, sigma=[2, 5, 10, 20, 40, 80], cuda=True):
        self.sigma = sigma
        self.cuda = cuda

    def build_loss(self):
        return self.moment_loss

    def get_scale_matrix(self, M, N):
        s1 = torch.ones((N, 1)) * 1.0 / N
        s2 = torch.ones((M, 1)) * -1.0 / M
        if self.cuda:
            s1, s2 = s1.cuda(), s2.cuda()
        return torch.cat((s1, s2), 0)

    def moment_loss(self, gen_samples, x):
        X = torch.cat((gen_samples, x), 0)
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X * X, 1, keepdim=True)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()
        M = gen_samples.size()[0]
        N = x.size()[0]
        s = self.get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in self.sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)
        return loss


class RelationNet(nn.Module):
    def __init__(self, feature_size, embed_size):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(feature_size + embed_size, feature_size)
        self.fc2 = nn.Linear(feature_size, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation

class Decoder(nn.Module):
    def __init__(self, drop_out_rate, embedding_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_out_rate),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_out_rate),
        )
    def forward(self, z):
        x = self.decoder(z)
        return x


class Encoder(nn.Module):
    def __init__(self, drop_out_rate, embedding_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_out_rate),
        )
    def forward(self, z):
        x = self.encoder(z)
        return x