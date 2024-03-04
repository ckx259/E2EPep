import sys
import os
import argparse
import re, torch
import torch.nn as nn
import time
from transformers import BertTokenizer
import esm
from transformers import BertModel


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads  # 64
        self.scale = att_size ** -0.5  # 1/4

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()  # [batchsize, L1, 256]

        d_k = self.att_size  # 64
        d_v = self.att_size  # 64
        batch_size = q.size(0)  # batchsize

        q = q.view(-1, q.size(-1))
        k = k.view(-1, k.size(-1))
        v = v.view(-1, v.size(-1))

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)  # [batchsize, L1, 256] -> [batchsize, L1, 4, 64]
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)  # [batchsize, L2, 256] -> [batchsize, L2, 4, 64]
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)  # [batchsize, L2, 256] -> [batchsize, L2, 4, 64]

        q = q.transpose(1, 2)  # [batchsize, L1, 4, 64] -> [batchsize, 4, L1, 64]
        v = v.transpose(1, 2)  # [batchsize, L2, 4, 64] -> [batchsize, 4, L2, 64]
        k = k.transpose(1, 2).transpose(2, 3)  # [batchsize, L2, 4, 64] -> [batchsize, 4, 64, L2]

        q = q * self.scale
        x = torch.matmul(q, k)  # [1, 4, L1, L2]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)

        x = self.att_dropout(x)
        x = x.matmul(v)  # [batchsize, 4, L1, 64]

        x = x.transpose(1, 2).contiguous()  # [batchsize, L1, 4, 64]
        x = x.view(batch_size, -1, self.num_heads * d_v)  # [batchsize, L1, 4, 64] -> [batchsize, L1, 256]

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class Encoder_cross_q(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(Encoder_cross_q, self).__init__()

        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.cross_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        x = self.cross_attention_norm(x)
        y = self.cross_attention_norm(kv)
        y = self.cross_attention(y, x, x, attn_bias)
        y = self.cross_attention_dropout(y)
        kv = kv + y

        y = self.ffn_norm(kv)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        kv = kv + y
        return kv


class Encoder_cross_v(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(Encoder_cross_v, self).__init__()

        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.cross_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, qk, attn_bias=None):
        y = self.cross_attention_norm(x)
        qk = self.cross_attention_norm(qk)
        y = self.cross_attention(y, y, qk, attn_bias)
        y = self.cross_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder_self(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(Encoder_self, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        x1 = self.self_attention_norm(x)
        x1 = self.self_attention(x1, x1, x1, attn_bias)
        x1 = self.self_attention_dropout(x1)
        x = x + x1

        x2 = self.ffn_norm(x)
        x2 = self.ffn(x2)
        x2 = self.ffn_dropout(x2)
        x3 = x + x2
        return x3


class Seq2ESM2andProt(nn.Module):
    def __init__(self, esm2_model_path, prot_model_path):
        super(Seq2ESM2andProt, self).__init__()
        self.device = args.device
        self.Esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_model_path)
        self.bert = BertModel.from_pretrained(prot_model_path)
        self.ESM2_freeze = nn.Sequential(
            nn.ELU(),
            nn.Linear(1280, 512),
        )
        self.ProtTrans_freeze = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, 512),
        )
        self.tokenizer_esm2 = self.alphabet.get_batch_converter()
        self.tokenizer_prot = BertTokenizer.from_pretrained(prot_model_path, do_lower_case=False)

    def tokenize(self, seq):
        """
        :param tuple_list: e.g., [('seq1', 'FFFFF'), ('seq2', 'AAASDA')]
        :param new_bert_seq: e.g., ['seq1', 'seq2']
        """
        tuple_list = [("seq", "{}".format(seq))]
        new_bert_seq = []
        with torch.no_grad():
            _, _, tokens = self.tokenizer_esm2(tuple_list)
            input_seq = ' '.join(seq)
            input_seq = re.sub(r"[UZOB]", "X", input_seq)
            new_bert_seq.append(input_seq)
            encoded_input = self.tokenizer_prot(new_bert_seq, return_tensors='pt', padding=True, max_length=1500)
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(self.device)
            return tokens.to(self.device), encoded_input

    def embed(self, seq):
        with torch.no_grad():
            if len(seq) < 5000:
                # [B, L_rec, D]
                token, encoded_input = self.tokenize(seq)
                emb_esm2 = self.Esm2(token, repr_layers=[33])["representations"][33][..., 1:-1, :]
                output = self.bert(**encoded_input)
                output = output[0]
                emb_prot = output[..., 1:-1, :]
                emb_esm2 = emb_esm2.reshape(-1, emb_esm2.size(-1))
                emb_prot = emb_prot.reshape(-1, emb_prot.size(-1))
                emb_esm2 = self.ESM2_freeze(emb_esm2)
                emb_prot = self.ProtTrans_freeze(emb_prot)
                emb_esm2 = emb_esm2.reshape(1, -1, emb_esm2.size(-1))
                emb_prot = emb_prot.reshape(1, -1, emb_prot.size(-1))
                return emb_esm2, emb_prot
            else:
                embs_esm2 = None
                embs_prot = None
                for ind in range(0, len(seq), 5000):
                    sind = ind
                    eind = min(ind + 5000, len(seq))
                    sub_seq = seq[sind:eind]
                    print(len(sub_seq), len(seq))
                    token, encoded_input = self.tokenize(sub_seq)
                    sub_emb_esm2 = self.Esm2(token, repr_layers=[33])["representations"][33][..., 1:-1, :]
                    sub_output = self.bert(**encoded_input)
                    sub_output = sub_output[0]
                    sub_emb_prot = sub_output[..., 1:-1, :]
                    sub_emb_esm2 = sub_emb_esm2.reshape(-1, sub_emb_esm2.size(-1))
                    sub_emb_prot = sub_emb_prot.reshape(-1, sub_emb_prot.size(-1))
                    sub_emb_esm2 = self.ESM2_freeze(sub_emb_esm2)
                    sub_emb_prot = self.ProtTrans_freeze(sub_emb_prot)
                    sub_emb_esm2 = sub_emb_esm2.reshape(1, -1, sub_emb_esm2.size(-1))
                    sub_emb_prot = sub_emb_prot.reshape(1, -1, sub_emb_prot.size(-1))
                    if (None is embs_esm2) or (None is embs_prot):
                        embs_esm2 = sub_emb_esm2
                        embs_prot = sub_emb_prot
                    else:
                        embs_esm2 = torch.cat([embs_esm2, sub_emb_esm2], dim=1)
                        embs_prot = torch.cat([embs_prot, sub_emb_prot], dim=1)
                print(embs_esm2.size(), embs_prot.size())
                return embs_esm2, embs_prot


class E2EPep(nn.Module):
    def __init__(self):
        super(E2EPep, self).__init__()

        self.Linear_mid = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, 256),
        )
        self.Linear_final = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )
        self.attention_q = Encoder_cross_q(512, 512, 0, 0, 8)
        self.attention_v = Encoder_cross_v(512, 512, 0, 0, 8)

    def forward(self, esm2_embeding, protTrans_embeding):
        ESM2_F_1 = self.attention_q(esm2_embeding, protTrans_embeding)
        ProtTrans_F_1 = self.attention_q(protTrans_embeding, esm2_embeding)
        ESM2_F_2 = self.attention_v(ESM2_F_1, ProtTrans_F_1)
        ProtTrans_F_2 = self.attention_v(ProtTrans_F_1, ESM2_F_1)
        representations = torch.cat((ESM2_F_2, ProtTrans_F_2), dim=2)
        representations = representations.view(-1, representations.size(-1))
        out = self.Linear_mid(representations)
        out = self.Linear_final(out)
        return out


def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if 1 < len(name):
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if 0 < seq_list.__len__():
        ans[name] = "".join(seq_list)
    return ans


def exists(fileOrFolderPath):
    return os.path.exists(fileOrFolderPath)


def createFolder(folder):
    if not exists(folder):
        os.makedirs(folder)


def dateTag():
    time_tuple = time.localtime(time.time())
    yy = time_tuple.tm_year
    mm = "{}".format(time_tuple.tm_mon)
    dd = "{}".format(time_tuple.tm_mday)
    if len(mm) < 2:
        mm = "0" + mm
    if len(dd) < 2:
        dd = "0" + dd

    date_tag = "{}{}{}".format(yy, mm, dd)
    return date_tag


def timeTag():
    time_tuple = time.localtime(time.time())
    hour = "{}".format(time_tuple.tm_hour)
    minuse = "{}".format(time_tuple.tm_min)
    second = "{}".format(time_tuple.tm_sec)
    if len(hour) < 2:
        hour = "0" + hour
    if len(minuse) < 2:
        minuse = "0" + minuse
    if len(second) < 2:
        second = "0" + second

    time_tag = "{}:{}:{}".format(hour, minuse, second)
    return time_tag


def timeRecord(time_log, content):
    date_tag = dateTag()
    time_tag = timeTag()
    with open(time_log, 'a') as file_object:
        file_object.write("{} {} says: {}\n".format(date_tag, time_tag, content))


def parsePredProbs(outs):
    """
    :param outs [Tensor]: [*, 2 or 1]
    :return pred_probs: [*], tgts: [*]
    """

    # 1 : one probability of each sample
    # 2 : two probabilities of each sample
    __type = 1
    if outs.size(-1) == 2:
        __type = 2
        outs = outs.view(-1, 2)
    else:
        outs = outs.view(-1, 1)

    sam_num = outs.size(0)

    outs = outs.tolist()

    pred_probs = []
    for j in range(sam_num):
        out = outs[j]
        if 2 == __type:
            prob_posi = out[1]
            prob_nega = out[0]
        else:
            prob_posi = out[0]
            prob_nega = 1.0 - prob_posi

        sum = prob_posi + prob_nega

        if sum < 1e-99:
            pred_probs.append(0.)
        else:
            pred_probs.append(prob_posi / sum)

    return pred_probs


def getPredLabs(predprobs_list, thre):
    res = []
    for pre in predprobs_list:
        if pre > thre:
            res.append(1)
        else:
            res.append(0)
    return res


def load_selected_layers(model, check_point, esm2_layers_to_load, bert_layers_to_load):
    # Gets the parameters for the layer to load
    for i, layer in enumerate(esm2_layers_to_load):
        loaded_state_dict = check_point['esm2_model_layers'][i]
        model.Esm2.layers[layer].load_state_dict(loaded_state_dict)
        print(f"ESM2 Layer {layer} loaded successfully.")

    for i, layer in enumerate(bert_layers_to_load):
        loaded_state_dict = check_point['bert_model_layers'][i]
        model.bert.encoder.layer[layer].load_state_dict(loaded_state_dict)
        print(f"Bert Layer {layer} loaded successfully.")

    esm2_freeze_state = check_point['esm2_freeze']
    model.ESM2_freeze.load_state_dict(esm2_freeze_state)
    print("ESM2 freeze parameters loaded successfully.")

    bert_freeze_state = check_point['bert_freeze']
    model.ProtTrans_freeze.load_state_dict(bert_freeze_state)
    print("Bert freeze parameters loaded successfully.")


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--savefolder")
    parser.add_argument("-seq_fa", "--seq_fa")
    parser.add_argument("-sind", "--start_index", type=int, default=0)
    parser.add_argument("-eind", "--end_index", type=int, default=-1)
    parser.add_argument("-mdn", "--e2epep_model_dir_name", type=str, default='E2EPep_model')
    parser.add_argument("-fmdn", "--e2epep_finetuned_model_dir_name", type=str, default='E2EPep_finetuned_model')
    parser.add_argument("-sc", "--set_cutoff", type=float, default=-1.0)
    parser.add_argument("-cuda", type=bool, default=True)
    parser.add_argument("-dv", "--device", default='cuda:0')
    args = parser.parse_args()

    if args.savefolder is None or args.seq_fa is None:
        parser.print_help()
        exit("PLEASE INPUT YOUR PARAMETERS CORRECTLY")
    if not (args.set_cutoff == -1 or (args.set_cutoff >= 0.0 and args.set_cutoff <= 1.0)):
        exit("PLEASE INPUT CORRECT CUTOFF")


    args.cutoff = 0.5
    savefolder = args.savefolder
    createFolder(savefolder)

    timeRecord("{}/run.time".format(savefolder), "Start")

    seq_fa = args.seq_fa
    esm2m = "{}/premodel/esm2_t33_650M_UR50D.pt".format(os.path.abspath('.'))
    protm = "{}/premodel/prot_bert_bfd".format(os.path.abspath('.'))
    e2epepmp = "{}/model/{}".format(os.path.abspath('.'), args.e2epep_model_dir_name)
    finetuned_model = "{}/model/{}".format(os.path.abspath('.'), args.e2epep_finetuned_model_dir_name)

    seq_dict = loadFasta(seq_fa)
    start_index = args.start_index
    end_index = args.end_index
    if end_index <= start_index:
        end_index = len(seq_dict)

    keys = []
    for key in seq_dict:
        keys.append(key)

    print('*' * 60 + 'Test Starting' + '*' * 60)
    tot_seq_num = len(seq_dict)

    esm2_prot_pre = Seq2ESM2andProt(esm2m, protm)
    if args.cuda:
        esm2_prot_pre.to(args.device)
        checkpoint_finetuned = torch.load(finetuned_model + os.sep + 'model.pkl')
    else:
        checkpoint_finetuned = torch.load(finetuned_model + os.sep + 'model.pkl', map_location='cpu')

    load_selected_layers(esm2_prot_pre, checkpoint_finetuned, [30, 31, 32], [28, 29])

    for name, param in esm2_prot_pre.named_parameters():
        param.requires_grad = False

    for ind in range(tot_seq_num):
        if ind < start_index or ind >= end_index:
            continue
        key = keys[ind]
        seq = seq_dict[key]

        if ind % 1 == 0:
            print("The {}/{}-th {}({}) is predicting...".format(ind, tot_seq_num, key, len(seq)))

        predict_list = []
        single_predict_list = []
        pre_lab_list = []
        thre_list = []
        model = E2EPep()
        filepath = "{}/{}.pred".format(savefolder, key)

        esm2_prot_pre.eval()
        with torch.no_grad():
            esm2_emb, prot_emb = esm2_prot_pre.embed(seq)


        for k, model_file in enumerate(os.listdir(e2epepmp)):
            if args.cuda:
                checkpoint = torch.load(e2epepmp + os.sep + model_file)
                model.to(args.device)
            else:
                checkpoint = torch.load(e2epepmp + os.sep + model_file, map_location='cpu')
            if args.set_cutoff == -1.0:
                thre = checkpoint['thre']
            else:
                thre = args.set_cutoff
            thre_list.append(thre)
            # Gets the parameters for the layer to load
            selected_params = {}
            for name, param in checkpoint['model'].items():
                if 'Linear_mid' in name:
                    selected_params[name] = param
                elif 'Linear_final' in name:
                    selected_params[name] = param
                elif 'attention_q' in name:
                    selected_params[name] = param
                elif 'attention_v' in name:
                    selected_params[name] = param

            model.load_state_dict(selected_params)

            model.eval()
            with torch.no_grad():
                if args.cuda:
                    esm2_emb, prot_emb = esm2_emb.cuda(), prot_emb.cuda()
                out = model(esm2_emb, prot_emb)
                out = torch.softmax(out, dim=-1)
                predict_list.append(parsePredProbs(out))
                pre_lab_list.append(getPredLabs(parsePredProbs(out), thre))


        with open(filepath, 'w') as file_object:
            # Gets the length of the sublist, assuming that all sublists are of the same length
            list_length = len(pre_lab_list[0])
            print(len(thre_list))
            file_object.write(
                "Index\tAA\tPNums\tState\n")

            # List parsing was used to compare the number of ones and zeros at each index position and generate new lists
            result = []
            nums = []
            for j in range(list_length):
                res = 0
                for k in range(len(pre_lab_list)):
                    if pre_lab_list[k][j] == 1:
                        res += 1
                nums.append(res)
                if res > (len(thre_list) // 2):
                    result.append('B')
                else:
                    result.append('N')

            for i in range(list_length):
                file_object.write(
                    "{}\t{}\t{}\t{}\n".format(i, seq[i], nums[i], result[i]))
            file_object.close()
        # exit()
        torch.cuda.empty_cache()
