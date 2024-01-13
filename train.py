
import logging
from logger import setlogger
import os
import argparse
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import DataLoader
import random
from FeatureEncoding import *
from Dataset import CombData_seq_st
import torch
import torch.nn as nn
from model import CPC
import torch.optim as optim


def parse_arguments(parser):
    parser.add_argument('--protein', type=str, default='WTAP', help='the protein for training model')
    parser.add_argument('--modelType', type=str, default='./circRNA2Vec/circRNA2Vec_model',
                        help='generate the embedding_matrix')
    parser.add_argument('--input_size1', type=int, default=30)
    parser.add_argument('--input_size2', type=int, default=84)
    parser.add_argument('--input_size3', type=int, default=28)
    parser.add_argument('--input_size4', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)   
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    return args

def do_compute_metrics(probas_pred, target):

    pred = (probas_pred >= 0.5).astype(np.int32)  

    acc = metrics.accuracy_score(target, pred)  
    auc_roc = metrics.roc_auc_score(target, probas_pred)    
    fscore = metrics.f1_score(target, pred)   

    p, r, t = metrics.precision_recall_curve(target, probas_pred)   
    aupr = metrics.auc(r, p)     

    precision = metrics.precision_score(target, pred, zero_division=1)
    recall = metrics.recall_score(target, pred)


    return acc, precision, recall, auc_roc, aupr, fscore

def calculate_performace(test_num, pred,  labels):
    pred_y = (pred >= 0.5).astype(np.int32)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1

    if tp == 0 and fp == 0:
        MCC = 0
    else:
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

    return float(MCC)




if __name__ == '__main__':
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    protein = args.protein
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    num_levels = args.num_levels
    input_size1 = args.input_size1
    input_size2 = args.input_size2
    input_size3 = args.input_size3
    input_size4 = args.input_size4
    patience = args.patience
    modelType = args.modelType

    #############saving the training logs
    save_dir = './log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, '{}_train.log'.format(protein)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seqpos_path = './dataset/' + protein + '/positive'
    seqneg_path = './dataset/' + protein + '/negative'

    seq_Embedding, dataY, seq_embedding_matrix = Generate_Embedding(seqpos_path, seqneg_path, modelType)

 
    Kmer = deal_seq_data(protein)  #

   
    st_dataX = deal_st_data(protein)

 
    dot_dataX = deal_dot_data(protein)


    indexes = np.random.choice(seq_Embedding.shape[0], seq_Embedding.shape[0], replace=False)
    training_idx, test_idx = indexes[:round(((seq_Embedding.shape[0]) / 10) * 8)], indexes[round(((seq_Embedding.shape[0]) / 10) * 8):]
    train_sequence, test_sequence = seq_Embedding[training_idx, :], seq_Embedding[test_idx, :]
    train_kmer, test_kmer = Kmer[training_idx, :, :], Kmer[test_idx, :, :]
    train_label, test_label = dataY[training_idx, :], dataY[test_idx, :]
    train_st, test_st = st_dataX[training_idx, :, :], st_dataX[test_idx, :, :]
    train_dot, test_dot = dot_dataX[training_idx, :, :], dot_dataX[test_idx, :, :]
    test_dataset = CombData_seq_st(seq_embedding_matrix, test_sequence, test_label, test_st, test_dot, test_kmer)
    test_data_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


    fold = 0
    kf = KFold(n_splits=5, shuffle=True)

    for train_index, val_index in kf.split(train_label):
        fold += 1
        logging.info('Training for Fold {}'.format(fold))
        train_seq_X1 = train_sequence[train_index]
        train_y = train_label[train_index]
        train_st_X = train_st[train_index]
        train_dot_X = train_dot[train_index]
        train_seq_X2 = train_kmer[train_index]

        val_seq_X1 = train_sequence[val_index]
        val_y = train_label[val_index]
        val_st_X = train_st[val_index]
        val_dot_X = train_dot[val_index]
        val_seq_X2 = train_kmer[val_index]

        logging.info('## Training: {}'.format(len(train_index)))
        logging.info('## validation: {}'.format(len(val_index)))

        train_dataset = CombData_seq_st(seq_embedding_matrix, train_seq_X1, train_y, train_st_X, train_dot_X, train_seq_X2)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = CombData_seq_st(seq_embedding_matrix, val_seq_X1, val_y, val_st_X, val_dot_X, val_seq_X2)
        val_data_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)

        model = CPC(
            input_size1=input_size1,
            input_size2=input_size2,
            input_size3=input_size3,
            input_size4=input_size4,
            num_levels=num_levels
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        loss_func = nn.CrossEntropyLoss()  
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = np.inf
        best_model_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            train_probas_pred = []
            train_ground_truth = []
            val_probas_pred = []
            val_ground_truth = []
            model.train()
            for i,  data in enumerate(train_data_loader):
                label = data['label'].cuda()
                prediction_train = model(data)
                loss = loss_func(prediction_train, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()
                probas_pred = torch.softmax(prediction_train, 1).cpu().detach().numpy()  
                train_probas_pred.append(probas_pred[:, 1])  
                train_ground_truth.append(label.cpu().detach().numpy())
            train_loss /= (i + 1)
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)
            train_acc, train_precision, train_recall, train_auc_roc, train_aupr, train_fscore = do_compute_metrics(train_probas_pred,
                                                                         train_ground_truth)  
            train_mcc = calculate_performace(train_ground_truth.shape[0], train_probas_pred, train_ground_truth)

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for i,  data in enumerate(val_data_loader):
                    label = data['label'].cuda()
                    prediction_val = model(data)
                    loss = loss_func(prediction_val, label)
                    val_loss += loss.detach().item()
                    probas_pred = torch.softmax(prediction_val, 1).cpu().detach().numpy()
                    val_probas_pred.append(probas_pred[:, 1])
                    val_ground_truth.append(label.cpu().detach().numpy())
            val_loss /= (i + 1)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_precision, val_recall, val_auc_roc, val_aupr, val_fscore = do_compute_metrics(val_probas_pred, val_ground_truth)
            val_mcc = calculate_performace(val_ground_truth.shape[0], val_probas_pred, val_ground_truth)
            logging.info(
                'Epoch: {}, train_loss: {:.4}, val_loss: {:.4}, train_acc: {:.4}, val_acc:{:.4}, train_roc: {:.4},'
                ' val_roc: {:.4}, train_auprc: {:.4}, val_auprc: {:.4}, train_mcc: {:.4}, val_mcc: {:.4}'.format(
                    epoch + 1,
                    train_loss, val_loss, train_acc, val_acc,
                    train_auc_roc, val_auc_roc,
                    train_aupr, val_aupr, train_mcc, val_mcc))
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience:
                print('Early stopping at epoch {}...'.format(epoch + 1))
                break

        save_parameter = './parameter/{}'.format(protein)
        if not os.path.exists(save_parameter):
            os.makedirs(save_parameter)
        torch.save(model.state_dict(), save_parameter + '/{}_best.pth'.format(fold))
        logging.info("## parameters for {}_best saved ".format(fold))


