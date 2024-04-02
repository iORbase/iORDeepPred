import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import sys
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import torch,gc
import h5py
import time

class Protein_preprocessing():
    def __init__(self, csv_file):
        self.input_path = csv_file + 'seq.csv'

        self.per_residue = True
        self.per_residue_path = "./output/per_residue_embeddings.h5"  # where to store the embeddings

        self.per_protein = True
        self.per_protein_path = "./output/per_protein_embeddings.h5"  # where to store the embeddings

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print("Using {}".format(device))

    def get_T5_model(self):
        print('get model start')
        model = torch.load('./protT5/Pmodel.pth')
        # print('Using', torch.cuda.device_count(), 'GPUs!')
        torch.cuda.set_device(1)
        model = nn.DataParallel(model, device_ids=[1, 2, 3, 4, 5, 6, 7]).cuda()
        # model = model.to(device) # move model to GPU
        model = model.eval()  # set model to evaluation model
        tokenizer = torch.load("./protT5/Ptokenizer.pth")
        print('get model done')

        return model, tokenizer

    def read_seqs(self, seq_path):
        print('read seqs start')
        seq_list = pd.read_csv(seq_path, header=None, usecols=[2])
        print(seq_list)
        change_name_list = pd.read_csv(seq_path, header=None, usecols=[1])
        print(change_name_list)
        seq_list = seq_list.values.tolist()
        change_name_list = change_name_list.values.tolist()
        s_list = []
        c_list = []
        for seq in seq_list:
            seq = seq[0].replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            # while len(seq) < 512:
            #	seq = seq + '0'
            s_list.append(seq)

        for change_name in change_name_list:
            c_list.append(change_name[0])

        seq_dict = {}
        for i in range(len(s_list)):
            seq_dict[c_list[i]] = s_list[i]

        print('read seqs done')

        return seq_dict

    def get_embeddings(self, model, tokenizer, per_residue, per_protein, max_residues=3000, max_seq_len=1000, max_batch=7):
        print('get embedding start')
        cnt = 1
        results = {"residue_embs": dict(),
                   "protein_embs": dict(),
                   }
        gpu_num = 0
        start = time.time()
        batch = list()

        s_dic = self.read_seqs(self.input_path)

        for idx, change_name in enumerate(s_dic, 1):
            seq = s_dic[change_name]
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            l = []
            l.append(change_name)
            l.append(seq)
            l.append(seq_len)
            batch.append(l)

            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or idx == len(s_dic) or seq_len > max_seq_len:
                print('batch:', cnt)
                cnt = cnt + 1
                seq_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                g_num = gpu_num % 6

                device = 'cuda:' + str(g_num + 2)
                print('Using {}'.format(device))
                gpu_num += 1

                token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

                try:
                    with torch.no_grad():
                        # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                        embedding_repr = model(input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(e)
                    print("RuntimeError during embedding for {} (L={})".format(change_name, seq_len))
                    continue

                for batch_idx, identifier in enumerate(seq_ids):
                    v_len = seq_lens[batch_idx]
                    emb = embedding_repr.last_hidden_state[batch_idx, :v_len]

                    if per_residue:
                        results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                    if per_protein:
                        protein_emb = emb.mean(dim=0)
                        results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

        print('get embedding done')
        return results

    def save_embeddings(self, emb_dict, out_path):
        print('save embedding start')
        with h5py.File(str(out_path), "w") as hf:
            for sequence_id, embedding in emb_dict.items():
                # noinspection PyUnboundLocalVariable
                hf.create_dataset(sequence_id, data=embedding)

        print('save embedding done')
        return None

    def main(self):
        model, tokenizer = self.get_T5_model()
        # 	seqs = read_seqs(seq_path)
        results = self.get_embeddings(model, tokenizer, per_residue=True, per_protein=True)
        if self.per_residue:
            self.save_embeddings(results["residue_embs"], self.per_residue_path)
        if self.per_protein:
            self.save_embeddings(results["protein_embs"], self.per_protein_path)


class Smile_preprocessing():
    def __init__(self, csv_file):
        self.input_path = csv_file + "voc.csv"

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.per_atom = True
        self.per_atom_path = "./output/per_atom_embeddings.h5"  # where to store the embeddings
        self.per_smile = True
        self.per_smile_path = "./output/per_smile_embeddings.h5"  # where to store the embeddings

        # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print("Using {}".format(device))

    def get_T5_model(self):
        print('get model start')
        model = torch.load('./Smodel/Smodel_1M.pth')
        model = model.to(self.device)  # move model to GPU
        model = model.eval()  # set model to evaluation model
        tokenizer = torch.load("./Smodel/Stokenizer_1M.pth")
        print('get model done')
        return model, tokenizer

    def read_smiles(self, seq_path):
        print('read seqs start')
        seq_list = pd.read_csv(seq_path, header=None, usecols=[2])
        change_name_list = pd.read_csv(seq_path, header=None, usecols=[1])
        seq_list = seq_list.values.tolist()
        change_name_list = change_name_list.values.tolist()
        seq_list = [item[0] for item in seq_list]
        change_name_list = [item[0] for item in change_name_list]
        seq_dict = {}
        for i in range(len(seq_list)):
            seq_dict[str(change_name_list[i])] = seq_list[i]
        print('read seqs done')
        return seq_dict

    def get_embeddings(self, model, tokenizer, per_atom, per_smile, max_residues=3000, max_seq_len=1000, max_batch=1):
        print('get embedding start')
        cnt = 1
        results = {"atom_embs": dict(), "smile_embs": dict()}
        batch = list()
        s_dic = self.read_smiles(self.input_path)

        for idx, change_name in enumerate(s_dic, 1):
            seq = s_dic[change_name]
            # print(seq)
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            l = []
            l.append(change_name)
            l.append(seq)
            l.append(seq_len)
            batch.append(l)

            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or idx == len(s_dic) or seq_len > max_seq_len:
                print('batch:', cnt)
                cnt = cnt + 1
                seq_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                fe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, framework='pt', device=0)
                emb = fe(list(seqs).pop())
                emb_np = np.array(emb)
                emb_np = torch.tensor(emb_np)
                for batch_idx, identifier in enumerate(seq_ids):
                    if per_atom:
                        results["atom_embs"][identifier] = emb_np.detach().cpu().numpy().squeeze()
                    # print(emb_np.detach().cpu().numpy().squeeze())
                    # print(type(emb_np.detach().cpu().numpy().squeeze()))
                    if per_smile:
                        smile_emb = emb_np.mean(dim=1)
                        results["smile_embs"][identifier] = smile_emb.detach().cpu().numpy().squeeze()
                    # print(smile_emb.detach().cpu().numpy().squeeze())
                    # print(type(smile_emb.detach().cpu().numpy().squeeze()))
        # print(results["smile_embs"])
        print('get embedding done')
        return results

    def save_embeddings(self, emb_dict, out_path):
        print('save embedding start')
        with h5py.File(str(out_path), "w") as hf:
            for sequence_id, embedding in emb_dict.items():
                # noinspection PyUnboundLocalVariable
                hf.create_dataset(sequence_id, data=embedding)
        print('save embedding done')

        # print('save embedding as np start')
        # np.save(str(out_path), emb_dict)
        # # dd.io.save(str(out_path), emb_dict)
        # print('save embedding as np done')
        return None

    def main(self):
        model, tokenizer = self.get_T5_model()
        # 	seqs = read_seqs(seq_path)
        results = self.get_embeddings(model, tokenizer, per_atom=True, per_smile=True)
        if self.per_atom:
            self.save_embeddings(results["atom_embs"], self.per_atom_path)
        if self.per_smile:
            self.save_embeddings(results["smile_embs"], self.per_smile_path)


class Train_test_split():
    def __init__(self, file_name):
        self.name = file_name

    def main(self):
        print('read protein!')
        path = './output/per_protein_embeddings.h5'

        result = []

        p_result = {}

        f = h5py.File(path, 'r')
        for group in f.keys():
            dset = f[group]
            data = np.array(dset)
            p_result[group] = data
            # result.append(dict)

        print('read smile!')
        path = './output/per_smile_embeddings.h5'
        # a = np.load('per_smile_embeddings.h5.npy',allow_pickle=True)

        # alist = a.tolist()
        s_result = {}

        f = h5py.File(path, 'r')
        for group in f.keys():
            dset = f[group]
            data = np.array(dset)
            s_result[group] = data

        # for i in alist:
        #    s_result[str(i)] = np.array(alist[i])

        print('read interaction!')
        path = './csv_file/inter.csv'

        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        # del rows[0]

        for row in rows:
            try:
                dic = {}
                dic['0'] = row[3]
                dic['1'] = row[2]
                dic['2'] = p_result[row[2]]
                dic['3'] = row[1]
                dic['4'] = s_result[row[1]]
                result.append(dic)
            except Exception as e:
                print(e)

        result = np.array(result)
        np.save(self.name, result)

