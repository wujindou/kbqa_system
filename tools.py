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

import Levenshtein
import re
import unicodedata
def entity_linking(mention2entity_dict, input_mention):
    """
    输入问句的NER结果input_mention，找到mention2entity_dict中与其相关度高的几个mention，返回它们的entitis
    使用一些规则以适配更多的mention
    :param mention2entity_dict:
    :param input_mention:
    :return:
    """
    if input_mention == 'NONE':         # 对于查不到的，返回候选实体为空列表，下面保持一致
        return []

    input_mention = input_mention.replace(" ", "")      # mention2entity中的mention已经去空格了，这里要对NER结果也去空格
    relative_entities = mention2entity_dict.get(input_mention, [])    # 先尝试直接查
    if not relative_entities:                                   # 直接查查不到，进入模糊查询
        # 保存模糊查询结果，模糊查询势必会遍历整个知识库，匹配所有认为相似的mention并计算各自编辑距离，在比较编辑距离后选取最小的那几个mention
        fuzzy_query_relative_entities = dict()
        input_mention = unify_char_format(input_mention)
        for mention_key in mention2entity_dict.keys():
            prim_mention = mention_key
            _find = False

            # 先做数据格式的处理
            mention_key = unify_char_format(mention_key)

            if len(mention_key) == 0:
                continue

            if '\\' == mention_key[-1]: 
                    mention_key = mention_key[:-1] + '"'

            # 组合型的mention
            if ',' in mention_key or '、' in mention_key or '\\\\' in mention_key or ';' in mention_key or ('或' in mention_key and '或' not in input_mention):
                mention_splits = re.split(r'[,;、或]|\\\\', mention_key)
                for _mention in mention_splits:
                    if (len(input_mention) < 6 and Levenshtein.distance(input_mention, _mention) <= 1) \
                            or (len(input_mention) >= 6 and Levenshtein.distance(input_mention, _mention) <= 4) \
                            or (len(input_mention) >= 20 and Levenshtein.distance(input_mention, _mention) <= 10):
                        _find = True
                        fuzzy_query_relative_entities[prim_mention] = Levenshtein.distance(input_mention, _mention)
            # 非组合型的mention
            else:
                if (len(input_mention) < 6 and Levenshtein.distance(input_mention, mention_key) <= 1) \
                            or (len(input_mention) >= 6 and Levenshtein.distance(input_mention, mention_key) <= 4) \
                            or (len(input_mention) >= 20 and Levenshtein.distance(input_mention, mention_key) <= 10):
                    _find = True
                    fuzzy_query_relative_entities[prim_mention] = Levenshtein.distance(input_mention, mention_key)

        if fuzzy_query_relative_entities:               # 模糊查询查到了结果
            min_key = min(fuzzy_query_relative_entities.keys(), key=fuzzy_query_relative_entities.get)         # 最小编辑距离的那几个mention
            min_similar_score = fuzzy_query_relative_entities[min_key]
            for prim_mention in fuzzy_query_relative_entities.keys():
                if fuzzy_query_relative_entities[prim_mention] == min_similar_score:
                    relative_entities.extend(mention2entity_dict[prim_mention])
                    # print('在模糊查询时找到mention的匹配,主题词和映射表的mention分别为：', input_mention, prim_mention)
        else:                                           # 模糊查询仍然查不到结果
            # print('模糊查询仍查不到结果：', input_mention)
            pass
    if input_mention not in relative_entities:          # 对于一些常用词，不再mention2entity表中，也加入进来
        relative_entities.append(input_mention)
    return relative_entities

def unify_char_format(string):
    """
    用于将两个字符串做对比之前，先把字符串做规范化
    :param string:
    :return:
    """
    string = unicodedata.normalize('NFKC', string)             
    string = string.replace('【', '[').replace('】', ']')      
    string = string.lower()                               
    return string

def search_triples_by_index(relative_entitis, index, raw_graph_f):
    """
    :param relative_entitis: list
    :param index: dict
    :param raw_graph: the file-pointer of the raw graph file, and the content need to be post-process
    :return: list of all the triples relative to the input_triples entitis  双重列表
    """
    relative_triples = []
    for entity in relative_entitis:
        index_entity = index.get(entity, None)
        if index_entity:
            read_index, read_size = index[entity]['start_pos'], index[entity]['length']
            raw_graph_f.seek(read_index)
            readresult = raw_graph_f.read(read_size).decode('utf-8')
            for line in readresult.strip().split('\n'):
                triple = line.strip().split(' ||| ')
                relative_triples.append(triple)
    return relative_triples

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

def span_question(question, ner_result):
    """
    用于答案排序阶段，删去问句中与答案排序无关的信息，如主题词、疑问词等
    """
    question = question.replace(ner_result, '').replace('《', '').replace('》', '')
    for delete_word in ['我想知道','我想请问','请问你','请问','你知道','谁知道','知道','谁清楚','我很好奇','你帮我问问','有没有人看过','有没有人'
                        '怎么','这个','有多少个','有哪些','哪些','哪个','多少','几个','谁','被谁','还有'
                        ,'吗','呀','啊','吧','着','的','是','呢','了','？','?','什么']:
        question = question.replace(delete_word, '')

    return question

span_res = span_question('襄城公主的老公是谁？', '襄城公主')
print(span_res)

def score_similarity(word2vec_model, string1, string2):
    """
    比较两个字符串的相似度，从字符覆盖度、w2v相似度做综合评分，用于答案排序时，问句和三元组关系名的比较
    :return: 相似度得分
    """
    return char_overlap(string1, string2) + word2vec_sim(word2vec_model, string1, string2)


def char_overlap(string1, string2):
    char_intersection = set(string1) & set(string2)
    char_union = set(string1) | set(string2)
    return len(char_intersection) / len(char_union)


def word2vec_sim(word2vec_model, string1, string2):
    # 阅读n_similarity的源代码，是对两组词向量分别取平均值并做L2归一化，然后求内积
    words1 = jieba.cut(string1)
    words2 = jieba.cut(string2)

    de_seg1 = []
    de_seg2 = []
    vocabs =list(word2vec_model.key_to_index.keys())
    for seg in words1:
        if seg not in vocabs:
            _ws = [_w for _w in seg if _w in vocabs]
            de_seg1.extend(_ws)
        else:
            de_seg1.append(seg)
    for seg in words2:
        if seg not in vocabs:
            _ws = [_w for _w in seg if _w in vocabs]
            de_seg1.extend(_ws)
        else:
            de_seg2.append(seg)
    if de_seg1 and de_seg2:
        score = word2vec_model.n_similarity(de_seg1, de_seg2)
    else:
        score = 0
    return score