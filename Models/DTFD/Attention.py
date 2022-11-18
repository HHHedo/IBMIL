import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            # A = F.softmax(A, dim=1)  # softmax over N
            A = A.sigmoid()
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
            # A = A.sigmoid()

        return A  ### K x N


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, confounder_path=False):
        super(Classifier_1fc, self).__init__()
        self.confounder_path = confounder_path
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

        if confounder_path:
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, n_channels).float())
            conf_tensor = torch.cat(conf_list, 0) 
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            self.W_q = nn.Linear(n_channels, joint_space_dim)
            self.W_k = nn.Linear(n_channels, joint_space_dim)
            self.fc =  nn.Linear(n_channels*2, n_classes)
        else:
            self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)

        if self.confounder_path:
            M = x
            device = M.device
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            A = torch.mm(conf_k, bag_q.transpose(0, 1))
            A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            M = torch.cat((M, conf_feats),dim=1)
            pred = self.fc(M)
            Y_hat = torch.ge(pred, 0.5).float()
            return pred, M, A
        else:
            pred = self.fc(x)

            return pred, x, None

class Attention_with_Classifier(nn.Module):
    def __init__(self, args, L=512, D=128, K=1, num_cls=2, droprate=0, confounder_path=False):
        super(Attention_with_Classifier, self).__init__()
        
        
        if confounder_path:
            self.attention = Attention_Gated(L, D, K)
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, L).float())
            conf_tensor = torch.cat(conf_list, 0) 
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.5
            self.W_q = nn.Linear(L, joint_space_dim)
            self.W_k = nn.Linear(L, joint_space_dim)
            self.classifier =  nn.Linear(L*2, num_cls)
            self.dropout = nn.Dropout(dropout_v)
        else:
            self.confounder_path = False
            self.attention = Attention_Gated(L, D, K)
            self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        M = torch.mm(AA, x) ## K x L
        

        if self.confounder_path:
            device = M.device
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            A = torch.mm(conf_k, bag_q.transpose(0, 1))
            A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            M = torch.cat((M, conf_feats),dim=1)
            pred = self.classifier(M)
            Y_hat = torch.ge(pred, 0.5).float()
            return pred, M, A
        else:
            pred, _, _ = self.classifier(M) ## K x num_cls
            return pred, M, AA


        # return Y_prob, M, A

            