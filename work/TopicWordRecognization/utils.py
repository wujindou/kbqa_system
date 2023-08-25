from collections import Counter
import random
import numpy as np
import paddle


def read(data_path):
    all_sample_words, all_sample_labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        tmp_sample_words, tmp_sample_labels = [], []
        for line in f.readlines():
            if line == '\n' and tmp_sample_words and tmp_sample_words:
                all_sample_words.append(tmp_sample_words)
                all_sample_labels.append(tmp_sample_labels)
                tmp_sample_words, tmp_sample_labels = [], []
            else:
                word, label = line.strip().split(' ')[0], line.strip().split(' ')[1]
                tmp_sample_words.append(word)
                tmp_sample_labels.append(label)
    for idx in range(len(all_sample_words)):
        yield {"words": all_sample_words[idx], "labels": all_sample_labels[idx]}


def convert_example_to_feature(example, tokenizer, label2id, pad_default_tag=0, max_seq_len=512):
    features = tokenizer(example["words"], is_split_into_words=True, max_seq_len=max_seq_len)
    label_ids = [label2id[label] for label in example["labels"][:max_seq_len-2]]
    label_ids = [label2id[pad_default_tag]] + label_ids + [label2id[pad_default_tag]]
    assert len(features["input_ids"]) == len(label_ids)
    return features["input_ids"], features["token_type_ids"], label_ids


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class SeqEntityScore(object):       # 实体识别
    def __init__(self, id2tag):
        self.id2tag = id2tag
        self.real_entities = []
        self.pred_entities = []
        self.correct_entities = []

    def reset(self):
        self.real_entities.clear()
        self.pred_entities.clear()
        self.correct_entities.clear()

    def compute(self, real_count, pred_count, correct_count):
        recall = 0 if real_count == 0 else (correct_count / real_count)
        precision = 0 if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def get_result(self):
        result = {}
        real_counter = Counter([item[0] for item in self.real_entities])
        pred_counter = Counter([item[0] for item in self.pred_entities])
        correct_counter = Counter([item[0] for item in self.correct_entities])
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter.get(label, 0)
            correct_count = correct_counter.get(label, 0)
            recall, precision, f1 = self.compute(real_count, pred_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        real_total_count = len(self.real_entities)
        pred_total_count = len(self.pred_entities)
        correct_total_count = len(self.correct_entities)
        recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result

    def get_entities_bio(self, seq):
        # 解析一个tag序列，提取其中的实体，例如：
        # 原始序列：[B-Person, I-Person, I-Person, O, O, B-Organization, I-Organization]
        # 提出实体：[[Person, 0, 2], [Organization, 5, 6]]
        entities = []
        entity = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = self.id2tag[tag]

            if tag.startswith("B-"):
                if entity[2] != -1:     # 读到"B"时如果发现缓存有实体则记录
                    entities.append(entity)
                entity = [-1, -1, -1]
                entity[1] = indx
                entity[0] = tag.split('-', maxsplit=1)[1]
                entity[2] = indx
                if indx == len(seq) - 1:
                    entities.append(entity)
            elif tag.startswith('I-') and entity[1] != -1:
                _type = tag.split('-', maxsplit=1)[1]
                if _type == entity[0]:
                    entity[2] = indx
                if indx == len(seq) - 1:
                    entities.append(entity)
            else:       # 读到"O"
                if entity[2] != -1:     # 读到"O"时如果发现缓存有实体则记录
                    entities.append(entity)
                entity = [-1, -1, -1]
        return entities

    def update(self, real_paths, pred_paths):

        if isinstance(real_paths, paddle.Tensor):
            real_paths = real_paths.numpy()
        if isinstance(pred_paths, paddle.Tensor):
            pred_paths = pred_paths.numpy()

        for real_path, pred_path in zip(real_paths, pred_paths):
            # 获取真实序列和预测序列中的实体
            real_ents = self.get_entities_bio(real_path)
            pred_ents = self.get_entities_bio(pred_path)
            # 保存相应的实体
            self.real_entities.extend(real_ents)
            self.pred_entities.extend(pred_ents)
            self.correct_entities.extend([pred_ent for pred_ent in pred_ents if pred_ent in real_ents])

    def format_print(self, result, print_detail=False):
        def print_item(entity, metric):
            if entity != "Total":
                print(f"Entity: {entity} - Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")
            else:
                print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")

        print_item("Total", result["Total"])
        if print_detail:
            for key in result.keys():
                if key == "Total":
                    continue
                print_item(key, result[key])
            print("\n")