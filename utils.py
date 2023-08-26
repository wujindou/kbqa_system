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

def make_KG_index(knowledge_graph_path, forward_index_path):
    """
    读KG文件，用第一个实体为key构建单向索引，索引格式为字典，{mention:{'start_pos':int, 'length':int}, ...}
    利用索引读KG时：
    with open(knowledge_graph_path, 'rb') as f:
        f.seek(223)
        readresult = f.read(448).decode('utf-8')
    """
    def make_index(graph_path, index_path):
        print('begin to read KG', datetime.datetime.now())
        index_dict = dict()
        with open(graph_path, 'r', encoding='utf-8') as f:
            previous_entity = ''
            previous_start = 0
            while True:
                start_pos = f.tell()
                line = f.readline()
                if not line:
                    break
                entity = line.split(' ||| ')[0]
                if entity != previous_entity and previous_entity:
                    tmp_dict = dict()
                    tmp_dict['start_pos'] = previous_start
                    tmp_dict['length'] = start_pos - previous_start
                    index_dict[previous_entity] = tmp_dict
                    previous_start = start_pos
                previous_entity = entity
        print('finish reading KG, begin to write', datetime.datetime.now())
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(index_dict, ensure_ascii=False))
        print('finish writing', datetime.datetime.now())
    make_index(knowledge_graph_path, forward_index_path)