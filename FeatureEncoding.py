import numpy as np
import gensim
from gensim.models import KeyedVectors


def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def nd(seq, kmer, coden_dict):
    seq = seq.strip()
    k = kmer
    coden_dict = coden_dict
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(k) + 1):
        key = seq[i:i+kmer]
        index = coden_dict[key]
        vectors[i][index] = round(seq[0:i + k].count(key) / (i + 1), 3)
    return vectors


def deal_seq_data(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()

    Kmer_dataX = []

    dataY = []
    with open('./dataset/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                kmer1 = nd(line,1,tris1)
                kmer2 = nd(line,2,tris2)
                kmer3 = nd(line,3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))   
                Kmer_dataX.append(Kmer.tolist())

    with open('./dataset/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                kmer1 = nd(line,1,tris1)
                kmer2 = nd(line,2,tris2)
                kmer3 = nd(line,3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                Kmer_dataX.append(Kmer.tolist())

    Kmer_dataX = np.array(Kmer_dataX)
    return Kmer_dataX






def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index + 1)     
    return index_data


def seq2ngram(seqs, k, s, wv, label):
    list22 = []
    print('need to n-gram %d lines' % len(seqs))
    if label == 'pos':
        y = [[0, 1]] * len(seqs)
    elif label == 'neg':
        y = [[1, 0]] * len(seqs)

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line)
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22, y

def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)

    return np.asarray(bag_sen)


def Generate_Embedding(seq_posfile, seq_negfile, model):
    
    seqpos = read_fasta_file(seq_posfile)
    seqneg = read_fasta_file(seq_negfile)

    X, y, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
    return X, y, embedding_matrix



def pad_seqs(data, max_len):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(max_len) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.int32)
    out[mask] = np.concatenate(data)
    return out



def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list, pos_y = seq2ngram(pos_sequences, k, s, model1.wv, 'pos')
    neg_list, neg_y = seq2ngram(neg_sequences, k, s, model1.wv, 'neg')
    seqs = pos_list + neg_list
    y = pos_y + neg_y
    y = np.array(y, dtype=np.int)

   
    X = pad_seqs(seqs, MAX_LEN)

   
    embedding_matrix = np.zeros((len(model1.wv.vocab) + 1, vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i + 1] = embedding_vector

    return X, y, embedding_matrix


def st_embedding(seq, st):
    st_dict = {'AS': 0,'AH':1, 'AM': 2, 'AI': 3, 'AB': 4, 'AX': 5, 'AE':6,
               'GS': 7, 'GH': 8, 'GM': 9, 'GI': 10, 'GB': 11, 'GX': 12, 'GE': 13,
               'CS': 14, 'CH': 15, 'CM': 16, 'CI': 17, 'CB': 18, 'CX': 19, 'CE': 20,
               'US': 21, 'UH': 22, 'UM': 23, 'UI': 24, 'UB': 25, 'UX': 26, 'UE': 27
    }
   
    sequence_vector = np.zeros([101, 28])
    for i in range(0, 101):
        a = seq[i]
        b = st[i]
        c = a + b
        index = st_dict[c]
        sequence_vector[i, index] = 1.0

    return sequence_vector

def deal_st_data(protein):
    st_dataX = []
    co_st_dataX = []
    seq_dict = {}
    st_dict = {}
    name = ''
    with open('./dataset/' + protein + '/positive') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                name = name.replace(' ', '_')
                name = name.replace(':', '_')
                name = name.replace(',', '_')
                seq_dict[name] = ''
            else:
                line = line.replace('T', 'U').strip()
                seq_dict[name] = seq_dict[name] + line

    with open('./dataset/' + protein + '/negative') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                name = name.replace(' ', '_')
                name = name.replace(':', '_')
                name = name.replace(',', '_')
                seq_dict[name] = ''
            else:
                line = line.replace('T', 'U').strip()
                seq_dict[name] = seq_dict[name] + line


    with open('./dataset/' + protein + '/positive_st.txt') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                st_dict[name] = ''
            else:
                line = line.strip()
                st_dict[name] = st_dict[name] + line
    with open('./dataset/' + protein + '/negative_st.txt') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                st_dict[name] = ''
            else:
                line = line.strip()
                st_dict[name] = st_dict[name] + line

    for key, seq in seq_dict.items():
        st = st_dict[key]
        feature_st = st_embedding(seq, st)
        st_dataX.append(feature_st.tolist())

    return np.array(st_dataX)


def st_embedding_onehot(st):
    st_dict = {'S': 0,'H': 1, 'M': 2, 'I': 3, 'B': 4, 'X': 5, 'E': 6}
    # onehot
    sequence_vector = np.zeros([101, 7])
    for i in range(0, 101):
        b = st[i]
        index = st_dict[b]
        sequence_vector[i, index] = 1.0

    return sequence_vector

def deal_st_data_onehot(protein):
    st_dataX = []
    name = ''
    with open('./dataset/' + protein + '/positive_st.txt') as f:
        for line in f:
            if '>' in line:
                continue
            else:
                line = line.strip()
                st_onehot = st_embedding_onehot(line)
                st_dataX.append(st_onehot.tolist())
    with open('./dataset/' + protein + '/negative_st.txt') as f:
        for line in f:
            if '>' in line:
                continue
            else:
                line = line.strip()
                st_onehot = st_embedding_onehot(line)
                st_dataX.append(st_onehot.tolist())

    st_dataX = np.array(st_dataX)
    return st_dataX



def dot_embedding(seq, st):
    st_dict = {'A.': 0,'A(':1, 'A)': 2, 'U.': 3, 'U(': 4, 'U)': 5, 'C.':6,
               'C(': 7, 'C)': 8, 'G.': 9, 'G(': 10, 'G)': 11
    }
  
    sequence_vector = np.zeros([101, 12])
    for i in range(0, 101):
        a = seq[i]
        b = st[i]
        c = a + b
        index = st_dict[c]
        sequence_vector[i, index] = 1.0

    return sequence_vector

def deal_dot_data(protein):
    st_dataX = []
    seq_dict = {}
    st_dict = {}
    name = ''
    with open('./dataset/' + protein + '/positive_rnafold.txt') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                seq_dict[name] = ''
                st_dict[name] = ''
            elif '.' in line or '(' in line or ')' in line:
                line = line.strip().split(' ')
                st_dict[name] = st_dict[name] + line[0]
            else:
                line = line.strip()
                seq_dict[name] = seq_dict[name] + line

    with open('./dataset/' + protein + '/negative_rnafold.txt') as f:
        for line in f:
            if '>' in line:
                name = line[1:].strip()
                seq_dict[name] = ''
                st_dict[name] = ''
            elif '.' in line or '(' in line or ')' in line:
                line = line.strip().split(' ')
                st_dict[name] = st_dict[name] + line[0]
            else:
                line = line.strip()
                seq_dict[name] = seq_dict[name] + line
    for key, seq in seq_dict.items():
        st = st_dict[key]
        feature_st = dot_embedding(seq, st)
        st_dataX.append(feature_st.tolist())

    return np.array(st_dataX)


def dot_embedding_onehot(st):
    st_dict = {'.': 0, '(': 1, ')': 2
    }
    # onehot
    sequence_vector = np.zeros([101, 3])
    for i in range(0, 101):
        b = st[i]
        index = st_dict[b]
        sequence_vector[i, index] = 1.0

    return sequence_vector

def deal_dot_data_onehot(protein):
    st_dataX = []
    with open('./dataset/' + protein + '/positive_rnafold.txt') as f:
        for line in f:
            if '>' in line:
                continue
            elif '.' in line or '(' in line or ')' in line:
                line = line.strip().split(' ')
                line = line[0]
                dot_onehot = dot_embedding_onehot(line)
                st_dataX.append(dot_onehot.tolist())
            else:
                continue

    with open('./dataset/' + protein + '/negative_rnafold.txt') as f:
        for line in f:
            if '>' in line:
                continue
            elif '.' in line or '(' in line or ')' in line:
                line = line.strip().split(' ')
                line = line[0]
                dot_onehot = dot_embedding_onehot(line)
                st_dataX.append(dot_onehot.tolist())
            else:
                continue

    st_dataX = np.array(st_dataX)
    return st_dataX