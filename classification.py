train_set = "./Dataset/train.txt"
valid_set = "./Dataset/validation.txt"
test_set = "./Dataset/test.txt"

def process_fn(sample):
    sample = sample.strip().rstrip()
    label, sentence = sample.split('\t')
    word_list = sentence.split()
    return {
        "label": int(label),
        "word_list": word_list,
        "sentence": "".join(word_list)
    }

with open(train_set, "r", encoding="utf-8") as file:
    dataset_train = list(map(process_fn, file.readlines()))

with open(valid_set, "r", encoding="utf-8") as file:
    dataset_valid = list(map(process_fn, file.readlines()))
    
with open(test_set, "r", encoding="utf-8") as file:
    dataset_test = list(map(process_fn, file.readlines()))


freq_model = {}
prob_model = {}

for sample in dataset_train:
    for word in sample["word_list"]:
        if word not in freq_model:
            freq_model[word] = [0, 0]
        freq_model[word][sample["label"]] += 1


def build_prob_model(laplace_factor = 1):
    prob_model = {}
    for word in freq_model:
        prob_model[word] = [(1 * laplace_factor + freq_model[word][0]) / (2 * laplace_factor + sum(freq_model[word])), (1 * laplace_factor + freq_model[word][1]) / (2 * laplace_factor + sum(freq_model[word]))]
    return prob_model


def eval(dataset=dataset_valid, calibration_factor=1, log=True):
    acc = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    TPs, FPs, FNs, TNs = [], [], [], []

    for sample in dataset:
        prob = [1, 1]
        for word in sample["word_list"]:
            if word in prob_model:
                if prob_model[word][0] and prob_model[word][1]:
                    prob[0] *= prob_model[word][0]
                    prob[1] *= prob_model[word][1]

        predict = int(prob[1] * calibration_factor > prob[0])
        
        if predict == sample["label"]:
            acc += 1
        
        if sample["label"] == 1:
            if predict == 1:
                TP += 1
                TPs.append(sample)
            else:
                FN += 1
                FNs.append(sample)
        else:
            if predict == 1:
                FP += 1
                FPs.append(sample)
            else:
                TN += 1
                TNs.append(sample)

    acc = acc / len(dataset)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 =  2 / (1 / precision + 1 / recall)

    if log:
        print("    p    n")
        print("Y", str(TP).zfill(4), str(FP).zfill(4))
        print("N", str(FN).zfill(4), str(TN).zfill(4))

        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)
        print("acc:", acc)
    
    return TPs, FPs, FNs, TNs, acc, f1


def calibrate(dataset=dataset_train):
    calibration_factor = 1
    max_acc = 0
    max_acc_calibration_factor = 1
    max_f1 = 0
    max_f1_calibration_factor = 1

    print("calibrating")

    for i in range(40):
        calibration_factor = 0.6 + i / 50
        TPs, FPs, FNs, TNs, acc, f1 = eval(dataset, calibration_factor, False)
        if acc > max_acc:
            max_acc = acc
            max_acc_calibration_factor = calibration_factor
        if acc > max_f1:
            max_f1 = acc
            max_f1_calibration_factor = calibration_factor
    
    return max_f1_calibration_factor


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
    
    return max_f1_laplace_factor

if __name__ == "__main__":
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
