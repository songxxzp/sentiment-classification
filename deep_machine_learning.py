import math
import torch
import torch.nn.functional as F

from torch import Tensor
from gensim.models import KeyedVectors

train_set = "./Dataset/train.txt"
valid_set = "./Dataset/validation.txt"
test_set = "./Dataset/test.txt"

word2vec_model = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)

class Tokenizer:
    def __init__(self, index_to_key):
        self.index_to_key = index_to_key
        self.key_to_index = {}
        for index, key in enumerate(index_to_key):
            self.key_to_index[key] = index
    
    def encode(self, word_list):
        for word in word_list:
            if word not in self.key_to_index:
                self.key_to_index[word] = len(self.index_to_key)
                self.index_to_key.append(word)

        token_list = list(map(lambda x:self.key_to_index[x], word_list))
        return token_list

    def decode(self, token_list):
        word_list = list(map(lambda x:self.index_to_key[x], token_list))
        return word_list
    
    def max_token_id(self):
        return len(self.index_to_key)

tokenizer = Tokenizer(word2vec_model.index_to_key)

def process_fn(sample):
    sample = sample.strip().rstrip()
    label, sentence = sample.split('\t')
    word_list = sentence.split()
    token_list = list(tokenizer.encode(word_list))
    return {
        "label": int(label),
        "token_list": token_list,
        "sentence": "".join(word_list)
    }


print("loading")

with open(train_set, "r", encoding="utf-8") as file:
    dataset_train = list(map(process_fn, file.readlines()))

with open(valid_set, "r", encoding="utf-8") as file:
    dataset_valid = list(map(process_fn, file.readlines()))
    
with open(test_set, "r", encoding="utf-8") as file:
    dataset_test = list(map(process_fn, file.readlines()))

print("loaded")


freq_model = torch.zeros((2, tokenizer.max_token_id()))

with torch.no_grad():
    for sample in dataset_train:
        freq_model[sample["label"]][sample["token_list"]] += 1


def build_prob_model(laplace_factor=1):
    prob_model = torch.zeros(freq_model.shape)
    with torch.no_grad():
        prob_model[0][:] = torch.log((1 * laplace_factor + freq_model[0][:]) / (2 * laplace_factor + freq_model[0][:] + freq_model[1][:]))
        prob_model[1][:] = torch.log((1 * laplace_factor + freq_model[1][:]) / (2 * laplace_factor + freq_model[0][:] + freq_model[1][:]))
    return prob_model


def eval(dataset=dataset_valid, calibration_factor=1, log=True):
    acc = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    TPs, FPs, FNs, TNs = [], [], [], []

    logits = []
    targets = []

    for sample in dataset:
        logit = torch.tensor([0.0, math.log(calibration_factor)])
        logit[0] = prob_model[0][sample["token_list"]].sum()
        logit[1] = prob_model[1][sample["token_list"]].sum()

        logits.append(logit.tolist())
        targets.append([0, 1] if sample["label"] else [1, 0])
        prob = F.softmax(logit, dim=0)

        predict = bool(prob[1] > prob[0])

        if predict == bool(sample["label"]):
            acc += 1

        if sample["label"] == 1:
            if predict:
                TP += 1
                TPs.append(sample)
            else:
                FN += 1
                FNs.append(sample)
        else:
            if predict:
                FP += 1
                FPs.append(sample)
            else:
                TN += 1
                TNs.append(sample)

    logits = torch.tensor(logits, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float)
    loss = F.cross_entropy(logits, targets)

    acc = acc / len(dataset)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 =  2 / (1 / precision + 1 / recall)

    if log:
        print("loss:", loss.tolist())
        print("    p    n")
        print("Y", str(TP).zfill(4), str(FP).zfill(4))
        print("N", str(FN).zfill(4), str(TN).zfill(4))

        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)
        print("acc:", acc)
        print()
    
    return TPs, FPs, FNs, TNs, acc, f1


def calibrate(dataset=dataset_train):
    calibration_factor = 1
    max_acc = 0
    max_acc_calibration_factor = 1
    max_f1 = 0
    max_f1_calibration_factor = 1

    print("calibrating")

    for i in range(40):
        calibration_factor = 0.2 + i / 20
        TPs, FPs, FNs, TNs, acc, f1 = eval(dataset, calibration_factor, False)
        if acc > max_acc:
            max_acc = acc
            max_acc_calibration_factor = calibration_factor
        if acc > max_f1:
            max_f1 = acc
            max_f1_calibration_factor = calibration_factor
    
    return max_acc_calibration_factor


def experiment_on_laplace_factor(dataset=dataset_train):
    global prob_model

    laplace_factor = 1
    max_acc = 0
    max_acc_laplace_factor = 1
    max_f1 = 0
    max_f1_laplace_factor = 1
    
    
    for i in range(40):
        laplace_factor = 0.5 + i / 20
        prob_model = build_prob_model(laplace_factor)
        TPs, FPs, FNs, TNs, acc, f1 = eval(dataset, 1, False)
        if acc > max_acc:
            max_acc = acc
            max_acc_laplace_factor = laplace_factor
        if acc > max_f1:
            max_f1 = acc
            max_f1_laplace_factor = laplace_factor
    
    return max_acc_laplace_factor

if __name__ == "__main__":
    laplace_factor = 1
    laplace_factor = experiment_on_laplace_factor(dataset_valid)
    print("laplace factor:", laplace_factor)
    prob_model = build_prob_model(laplace_factor)

    calibration_factor = calibrate(dataset_valid)
    print("calibration factor:", calibration_factor)
    calibration_factor = 1

    print("on train:")
    TPs, FPs, FNs, TNs, acc, f1 = eval(dataset_train, calibration_factor)
    print()
    print("on valid:")
    TPs, FPs, FNs, TNs, acc, f1 = eval(dataset_valid, calibration_factor)
    print()
    print("on test:")
    TPs, FPs, FNs, TNs, acc, f1 = eval(dataset_test, calibration_factor)
