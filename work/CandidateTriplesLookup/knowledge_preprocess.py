import json
import datetime
import re


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


def clean_mention2entity(mention2entity_path, mention2entity_clean_path):
    """
    对mention2entity处理，包括：去掉mention的空格
    处理后直接存成dict类型，json保存
    """
    mention2entity_clean = dict()
    with open(mention2entity_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if ' ||| ' in line:
                mention = line.split(' ||| ')[0]
                entity = line.split(' ||| ')[1].strip()
                mention = mention.replace(' ', '')
                entity = entity.split('\t')
                # 增加对几个认为发现的错误mention2id信息的删除
                if mention in ['-', '——', '－－', '无·', '']:
                    continue
                if mention in ['无', '男']:
                    clean_entity = [x for x in entity if re.match(mention+r'(\(|$)', x)]
                    entity = clean_entity
                mention2entity_clean[mention] = entity
    with open(mention2entity_clean_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(mention2entity_clean))
