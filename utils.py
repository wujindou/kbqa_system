from work.TopicWordRecognization.run_ner import predict as ner_predict
from work.CandidateTriplesSelection.run_cls import predict as cls_predict
from work.CandidateTriplesLookup.knowledge_retrieval import entity_linking, search_triples_by_index
from work.AnswerRanking.ranking import span_question, score_similarity
from work.config import KGConfig, CLSConfig, NERConfig
import jieba
import gensim
import datetime
import json
import re
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer# , ErnieModel
from paddlenlp.data import Stack, Pad, Tuple
from tools import * 

KGconfig = KGConfig()
KGconfig.mention2entity_clean_path='/kaggle/input/kbqadata/nlpcc-iccpol-2016_mention2id_clean.json'
KGconfig.knowledge_graph_path='/kaggle/input/kbqadata/nlpcc-iccpol-2016.kbqa.kb/nlpcc-iccpol-2016.kbqa.kb'

mention2entity_clean_path = KGconfig.mention2entity_clean_path
knowledge_graph_path = KGconfig.knowledge_graph_path

print('正在加载mention2entity表', datetime.datetime.now())
with open(mention2entity_clean_path, 'r', encoding='utf-8') as f:
    mention2entity_dict = json.loads(f.read())

print('正在加载知识库', datetime.datetime.now())
forward_KG_f = open(knowledge_graph_path, 'rb')
print('知识库加载完毕', datetime.datetime.now())

KGconfig.forward_index_path='/kaggle/input/kbqadata/forward_index.json'
print('正在加载索引表', datetime.datetime.now())
forward_index_path = KGconfig.forward_index_path
with open(forward_index_path, 'r', encoding='utf-8') as f:
    forward_index = json.loads(f.read())
print('索引表加载完毕', datetime.datetime.now())

word2vec_model = load_word2vec()

input_mention = '史蒂芬霍金'
rela_ents = entity_linking(mention2entity_dict, input_mention)
print('匹配到知识库中的候选实体：', rela_ents)

rel_triples= search_triples_by_index(rela_ents, forward_index, forward_KG_f)
print('共检索到{}条三元组'.format(len(rel_triples)))
print('打印20条以内的三元组：')