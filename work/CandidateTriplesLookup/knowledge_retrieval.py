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
            # mention_key = re.sub(r'\(.*\)|\[.*\]|<.*>|\(.*?$|\[.*?$|<.*?$', '', mention_key)   # 去掉mention中带有的括号对或前括号
            # if not re.search(r'\w', mention_key) and not re.search(u'[\u4e00-\u9fa5]', mention_key):    # 有的mention去掉后就没了，那就只能去括号不去内容
            #     mention_key = re.sub(r'[\(\)\[\]<>]', '', prim_mention)

            if len(mention_key) == 0:   # 上面的操作把mention删没了
                continue

            if '\\' == mention_key[-1]:                     # mention最后如果是\符号，则八成是后引号
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


# 根据相关实体和索引查知识库，返回相关的三元组
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


def unify_char_format(string):
    """
    用于将两个字符串做对比之前，先把字符串做规范化
    :param string:
    :return:
    """
    string = unicodedata.normalize('NFKC', string)              # 全角转半角，中文标点转英文标点
    string = string.replace('【', '[').replace('】', ']')         # unicodedata.normalize并不能转中括号
    string = string.lower()                                     # 英文全部转小写
    return string

