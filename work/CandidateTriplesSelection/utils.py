from collections import Counter
import random
import numpy as np
import paddle


def read(data_path):
    all_sample_text1, all_sample_text2, all_sample_labels = [], [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text1, text2, label = line.strip().split('\t')
            all_sample_text1.append(text1)
            all_sample_text2.append(text2)
            all_sample_labels.append(label)
    for idx in range(len(all_sample_labels)):
        yield {"text1": all_sample_text1[idx], "text2": all_sample_text2[idx], "label": all_sample_labels[idx]}


def read_test(all_sample_text1, all_sample_text2):
    for idx in range(len(all_sample_text1)):
        yield {"text1": all_sample_text1[idx], "text2": all_sample_text2[idx], "label": '0'}


def convert_example_to_feature(example, tokenizer, label2id, max_seq_len=512):
    features = tokenizer(example["text1"], example["text2"], max_seq_len=max_seq_len)
    label_ids = label2id[example["label"]]
    return features["input_ids"], features["token_type_ids"], label_ids


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ClassificationScore(object):  # 单标签K分类
    def __init__(self, id2tag):
        self.id2tag = id2tag
        self.all_pred_labels = []
        self.all_true_labels = []
        self.all_correct_labels = []

    def reset(self):
        self.all_pred_labels.clear()
        self.all_true_labels.clear()
        self.all_correct_labels.clear()

    def compute(self, pred_count, real_count, correct_count):
        recall = 0 if real_count == 0 else (correct_count / real_count)
        precision = 0 if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def get_result(self):
        result = {}
        pred_counter = Counter(self.all_pred_labels)
        real_counter = Counter(self.all_true_labels)
        correct_counter = Counter(self.all_correct_labels)
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter[label]
            correct_count = correct_counter[label]
            precision, recall, f1 = self.compute(pred_count, real_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}


        # real_total_count = len(self.all_true_labels)
        # pred_total_count = len(self.all_pred_labels)
        # correct_total_count = len(self.all_correct_labels)
        # recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        # result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result

    def update(self, true_labels, pred_labels):
        """
        :param true_labels: tensor, shape=(bs)
        :param pred_labels: tensor, shape=(bs)
        """

        if isinstance(true_labels, paddle.Tensor):
            true_labels = true_labels.numpy().tolist()
        if isinstance(pred_labels, paddle.Tensor):
            pred_labels = pred_labels.numpy().tolist()

        for true_label, pred_label in zip(true_labels, pred_labels):
            self.all_true_labels.append(self.id2tag[true_label])
            self.all_pred_labels.append(self.id2tag[pred_label])
            if true_label == pred_label:
                self.all_correct_labels.append(self.id2tag[pred_label])

    def format_print(self, result):
        metric = result["Total"]
        print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")