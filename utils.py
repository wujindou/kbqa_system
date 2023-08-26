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

ner_config = NERConfig()
cls_config = CLSConfig()
ner_config.best_model_path='/kaggle/input/kbqadata/ernie_ner_best.pdparams'
cls_config.best_model_path='/kaggle/input/kbqadata/ernie_cls_best.pdparams'

def pipeline_predict(question):
    ner_results = ner_predict(ner_config.best_model_path, question)
    ner_results = set([_result.replace("《", "").replace("》", "") for _result in ner_results])
    # ner_results是一个set，可能有0个、1个或多个元素。如果是0个元素尝试以下规则看能否提取出实体
    if not ner_results:
        if '《' in question and '》' in question:
            ner_results = re.search(r'(.*)的.*是.*', question).group(1)
        elif re.search(r'', question):  
            ner_results = re.search(r'(.*)的.*是.*', question).group(1)
        else:
            print('没有提取出主题词！')
            return()

    print('■识别到的主题词：', ner_results, datetime.datetime.now())

    candidate_entities = []
    for mention in ner_results:
        candidate_entities.extend(entity_linking(mention2entity_dict, mention))
    print('■找到的候选实体：', candidate_entities, datetime.datetime.now())

    forward_candidate_triples = search_triples_by_index(candidate_entities, forward_index, forward_KG_f)
    candidate_triples = forward_candidate_triples
    candidate_triples = list(filter(lambda x: len(x) == 3, candidate_triples))
    candidate_triples_num = len(candidate_triples)
    print('■候选三元组共{}条'.format(candidate_triples_num), datetime.datetime.now())
    show_num = 20 if candidate_triples_num > 20 else candidate_triples_num
    print('■展示前{}条候选三元组：{}'.format(show_num, candidate_triples[:show_num]))

    candidate_triples_labels = cls_predict(cls_config.best_model_path, [question]*len(candidate_triples), [triple[0]+triple[1] for triple in candidate_triples])
    predict_triples = [candidate_triples[i] for i in range(len(candidate_triples)) if candidate_triples_labels[i] == '1']
    print('■三元组粗分类结果，保留以下三元组：', predict_triples)

    predict_answers = [_triple[2] for _triple in predict_triples]
    if len(predict_answers) == 0:
        print('■知识库中没有检索到相关知识，请换一个问题试试......')
        return()
    elif len(set(predict_answers)) == 1:  # 预测的答案只有一个，尽管提供答案的三元组可能有多个
        print('■预测答案唯一，直接输出......')
        best_triple = predict_triples[0]
        best_answer = predict_answers[0]
        print('■最佳答案：', best_answer)
    else:  # 预测出多个答案，需要排序
        print('■检测到多个答案，正在进行答案排序......')
        max_ner = ''  # 用所有ner结果中最长的那个去分割问句
        for _ner in ner_results:
            if len(_ner) > len(max_ner):
                max_ner = _ner
        fine_question = span_question(question, max_ner)
        rel_scores = [score_similarity(word2vec_model, _triple[1].replace(' ', ''), fine_question) for _triple in
                      predict_triples]
        triples_with_score = list(zip(map(tuple, predict_triples), rel_scores))
        triples_with_score.sort(key=lambda x: x[1], reverse=True)
        print('■三元组排序结果：\n{}'.format("\n".join([str(pair[0]) + '-->' + str(pair[1]) for pair in triples_with_score])))
        best_answer = triples_with_score[0][0][-1]
        print('■最佳答案：', best_answer)

question ='运动员达斯汀·约翰逊在2010PGA锦标赛取得了什么成绩？'
pipeline_predict(question)