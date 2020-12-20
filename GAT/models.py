import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

# 图注意模型的核心代码
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # 构建多头注意力机制
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
         
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # 进行池化，防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        #  单个注意力机制的输出向量进行叠加
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # 进行池化，防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        # 使用elu激活函数，进行非线性表达
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
