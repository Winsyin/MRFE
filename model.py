

import torch
import torch.nn as nn
from residual_dual_interaction_SCINet import SCINet

class convBasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=None, stride=1, padding='same'):
        super(convBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)     
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class CPC(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, input_size4, num_levels):
        super(CPC, self).__init__()
        self.kmernd_conv = nn.ModuleList()
        self.kmernd_conv.append(convBasicBlock(input_size2, 64, 1, padding='valid'))
        self.kmernd_conv.append(convBasicBlock(64, 64, 3, padding='same'))

        self.kmernd_conv1 = nn.ModuleList()
        self.kmernd_conv1.append(convBasicBlock(input_size2, 64, 1, padding='valid'))
        self.kmernd_conv1.append(convBasicBlock(64, 64, 5, padding='same'))

        self.rna2vec_conv = nn.ModuleList()
        self.rna2vec_conv.append(convBasicBlock(input_size1, 64, 1, padding='valid'))
        self.rna2vec_conv.append(convBasicBlock(64, 64, 3, padding='same'))

        self.rna2vec_conv1 = nn.ModuleList()
        self.rna2vec_conv1.append(convBasicBlock(input_size1, 64, 1, padding='valid'))
        self.rna2vec_conv1.append(convBasicBlock(64, 64, 5, padding='same'))

        self.seq_conv = nn.ModuleList()
        self.seq_conv.append(convBasicBlock(64 * 4, 64 * 4, 1, padding='same'))

      
        self.seq_avgpool = nn.AvgPool1d(7, stride=2)
       
        self.seq_maxpool = nn.MaxPool1d(7, stride=2)

    
        self.st_avgpool = nn.AvgPool1d(7, stride=2)
     
        self.st_maxpool = nn.MaxPool1d(7, stride=2)

        self.dot_conv = nn.ModuleList()
        self.dot_conv.append(convBasicBlock(input_size4, 64, 1, padding='valid'))
        self.dot_conv.append(convBasicBlock(64, 64, 3, padding='same'))

        self.dot_conv1 = nn.ModuleList()
        self.dot_conv1.append(convBasicBlock(input_size4, 64, 1, padding='valid'))
        self.dot_conv1.append(convBasicBlock(64, 64, 5, padding='same'))

        self.st_conv = nn.ModuleList()
        self.st_conv.append(convBasicBlock(64 * 2, 64 * 2, 1, padding='same'))

        self.scinet1 = SCINet(input_len=48, input_dim=64 * 4, hid_size=0.5, num_stacks=1, num_levels=num_levels,
                              groups=1,
                              kernel=5, dropout=0.5, modified=True)


       
        self.lstm1 = nn.LSTM(input_size=64 * 4, hidden_size=64 * 4, batch_first=True, bidirectional=True)

      
        self.lstm2 = nn.LSTM(input_size=64 * 2, hidden_size=64, batch_first=True, bidirectional=True)

        self.linear = nn.Sequential(
            nn.Linear(48 * 64 * 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, data):
        x1_1 = data['rna2vec_feature'].cuda()
        x2_1 = data['kmernd_feature'].cuda()
        x3_1 = data['dot_feature'].cuda()
       
      
        x1_2 = x1_1

        x1_1 = x1_1.transpose(1, 2).contiguous()
        x1_1 = self.rna2vec_conv[0](x1_1)

        x1_1 = self.rna2vec_conv[1](x1_1)


        x1_2 = x1_2.transpose(1, 2).contiguous()
        x1_2 = self.rna2vec_conv1[0](x1_2)


        x1_2 = self.rna2vec_conv1[1](x1_2)

        x1 = torch.cat((x1_1, x1_2), dim=1)
      
        x2_2 = x2_1
        x2_1 = x2_1.transpose(1, 2).contiguous()
        x2_1 = self.kmernd_conv[0](x2_1)

        x2_1 = self.kmernd_conv[1](x2_1)

        x2_2 = x2_2.transpose(1, 2).contiguous()
        x2_2 = self.kmernd_conv1[0](x2_2)

        x2_2 = self.kmernd_conv1[1](x2_2)

        x2 = torch.cat((x2_1, x2_2), dim=1)

        x1 = torch.cat((x1, x2), dim=1)

        x1 = self.seq_conv[0](x1)
       
        x1_1 = self.seq_avgpool(x1)
        x1_2 = self.seq_maxpool(x1)
        x1 = x1_1 + x1_2
        x1 = x1.transpose(1, 2).contiguous()

      
        out1 = self.scinet1(x1)

       
        out1, (hn, cn) = self.lstm1(out1)

      
        x3_2 = x3_1
        x3_1 = x3_1.transpose(1, 2).contiguous()
        x3_1 = self.dot_conv[0](x3_1)

        x3_1 = self.dot_conv[1](x3_1)

        x3_2 = x3_2.transpose(1, 2).contiguous()
        x3_2 = self.dot_conv1[0](x3_2)
        x3_2 = self.dot_conv1[1](x3_2)

        x3 = torch.cat((x3_1, x3_2), dim=1)

        x2 = self.st_conv[0](x3)

       
        x2_1 = self.st_avgpool(x2)
        x2_2 = self.st_maxpool(x2)
        x2 = x2_1 + x2_2
        x2 = x2.transpose(1, 2).contiguous()

        out2, (hn, cn) = self.lstm2(x2)

       
        out = torch.cat((out1, out2), dim=2)
        out = out.contiguous().view(out.size(0), -1)

        return self.linear(out)
