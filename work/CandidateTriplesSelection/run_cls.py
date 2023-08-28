import logging
from work.CandidateTriplesSelection.utils import read, convert_example_to_feature, set_seed, ClassificationScore, read_test
from work.CandidateTriplesSelection.model import ErnieCLS
from functools import partial
from work.config import CLSConfig

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple

logger = logging.getLogger()


def train():
    train_ds = load_dataset(read, data_path=train_path, lazy=False)  # 文件->example
    dev_ds = load_dataset(read, data_path=dev_path, lazy=False)

    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_seq_len=max_seq_len)

    train_ds = train_ds.map(trans_func, lazy=False)  # example->feature
    dev_ds = dev_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(axis=0, dtype='int64'),
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
    train_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn,
                                        return_list=True)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn,
                                      return_list=True)

    ernie = ErnieModel.from_pretrained(model_name)
    model = ErnieCLS(ernie, len(label2id), dropout=0.1)

    num_training_steps = len(train_loader) * num_epoch
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(),
                                       weight_decay=weight_decay, apply_decay_param_fun=lambda x: x in decay_params,
                                       grad_clip=grad_clip)

    loss_model = paddle.nn.CrossEntropyLoss()
    cls_metric = ClassificationScore(id2label)

    global_step, cls_best_f1 = 0, 0.
    model.train()
    for epoch in range(1, num_epoch + 1):
        for batch_data in train_loader:
            input_ids, token_type_ids, labels = batch_data
            logits = model(input_ids, token_type_ids=token_type_ids)

            loss = loss_model(logits, labels)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % log_step == 0:
                print(
                    f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if global_step > 0 and global_step % eval_step == 0:

                cls_results = evaluate(model, dev_loader, cls_metric)
                cls_result = cls_results["1"]
                model.train()
                cls_f1 = cls_result["F1"]
                if cls_f1 > cls_best_f1:
                    paddle.save(model.state_dict(), f"{save_path}/ernie_cls_best.pdparams")
                if cls_f1 > cls_best_f1:
                    print(f"\ncls best F1 performence has been updated: {cls_best_f1:.5f} --> {cls_f1:.5f}")
                    cls_best_f1 = cls_f1
                print(
                    f'\ncls evalution result: precision: {cls_result["Precision"]:.5f}, recall: {cls_result["Recall"]:.5f},  F1: {cls_result["F1"]:.5f}, current best {cls_best_f1:.5f}\n')

            global_step += 1


def evaluate(model, data_loader, metric):
    model.eval()
    metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)
        pred_labels = logits.argmax(axis=-1)
        metric.update(pred_labels=pred_labels, true_labels=labels)
    results = metric.get_result()

    return results

def load_cls_model(model_path):
    loaded_state_dict = paddle.load(model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    model = ErnieCLS(ernie, len(label2id), dropout=0.1)
    model.load_dict(loaded_state_dict)
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    model.eval()
    return model,tokenizer

def predict(model,tokenizer, input_text1, input_text2):
    # model_path = f"{save_path}/ernie_cls_best.pdparams"
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_seq_len=max_seq_len)

    if isinstance(input_text1, str) and isinstance(input_text2, str):
        features = tokenizer(input_text1, input_text2, max_seq_len=max_seq_len)
        input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0)
        token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids)
        pred_label = id2label[logits.argmax(axis=-1).item()]
        # print(pred_label)
        return pred_label
    elif isinstance(input_text1, list) and isinstance(input_text2, list):
        test_ds = load_dataset(read_test, all_sample_text1=input_text1, all_sample_text2=input_text2, lazy=False)
        test_ds = test_ds.map(trans_func, lazy=False)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            Stack(axis=0, dtype='int64'),
        ): fn(samples)
        test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=batch_size, shuffle=False)
        test_loader = paddle.io.DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn,
                                           return_list=True)

        pred_results = []
        for idx, batch_data in enumerate(test_loader):
            input_ids, token_type_ids, _ = batch_data
            logits = model(input_ids, token_type_ids=token_type_ids)
            pred_labels = logits.argmax(axis=-1).tolist()
            pred_results.extend([id2label[label] for label in pred_labels])
        # print(pred_results)
        return pred_results
    else:
        raise TypeError('整错了!!Please use pair of str or pair of list to predict!')


model_name = "ernie-1.0"
max_seq_len = 512
batch_size = 8
label2id = {"0": 0, "1": 1}
id2label = {0: "0", 1: "1"}

num_epoch = 10
learning_rate = 2e-5
weight_decay = 0.01
warmup_proportion = 0.1
max_grad_norm = 1.0
log_step = 50
eval_step = 400  # (len(train_loader)//batch_size)//20
seed = 1000

train_path = './data/train.tsv'
dev_path = './data/dev.tsv'
save_path = "./checkpoint"

# envir setting
set_seed(seed)
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device("gpu:0")

if __name__ == '__main__':
    # train()

    pred_model_path = CLSConfig().best_model_path

    # input_text1 = '若泽·萨尔内的总统是谁？'
    # input_text2 = '若泽·萨尔内总统'

    input_text1 = ['若泽·萨尔内的总统是谁？', '闻一多全集是哪个出版社出版的？', '闻一多全集是哪个出版社出版的？', ]
    input_text2 = ['若泽·萨尔内总统', '闻一多全集出版社', '闻一多全集出版时间', ]

    pred_results = predict(pred_model_path, input_text1, input_text2)
    print(pred_results)
