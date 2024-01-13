from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np

class CombData_seq_st(Dataset):
    def __init__(self, rna_embed_fn, rna2vec_feature, label, st_feature, dot_feature, kmernd_feature):
        self.rna_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(rna_embed_fn)
        )
        self.rna2vec_feature = torch.from_numpy(rna2vec_feature).long()
        self.label = torch.from_numpy(label[:, 1]).long()
        self.st_feature = torch.from_numpy(st_feature.astype(np.float32))
        self.dot_feature = torch.from_numpy(dot_feature.astype(np.float32))
        self.kmernd_feature = torch.from_numpy(kmernd_feature.astype(np.float32))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        rna2vec_feature = self.rna2vec_feature[item]
        # Single Embedding-Rna
        new_rna2vec_feature = self.rna_embedding(rna2vec_feature)
        kmernd_feature = self.kmernd_feature[item]
        st_feature = self.st_feature[item]
        dot_feature = self.dot_feature[item]
        label = self.label[item]

        return {
            'rna2vec_feature': new_rna2vec_feature,
            'kmernd_feature': kmernd_feature,
            'st_feature': st_feature,
            'dot_feature': dot_feature,
            'label': label
        }