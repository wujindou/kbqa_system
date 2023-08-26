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

from work.config import Word2VecConfig
from gensim.models import KeyedVectors
wd_config = Word2VecConfig()
wd_config.model_path = '/kaggle/input/kbqadata/sgns.target.word-character/sgns.target.word-character'
def load_word2vec():
    word2vec_model_path = wd_config.model_path # 词向量文件的位置
    print('正在预加载word2vec词向量，预计2min', datetime.datetime.now())
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')
    print('word2vec词向量加载完毕', datetime.datetime.now())
    return word2vec_model
word2vec_model = load_word2vec()

input_mention = '史蒂芬霍金'
rela_ents = entity_linking(mention2entity_dict, input_mention)
print('匹配到知识库中的候选实体：', rela_ents)