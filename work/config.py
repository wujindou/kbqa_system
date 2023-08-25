import os

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class KGConfig():
    def __init__(self):
        self.mention2entity_path = os.path.join(root_path, 'data/data120603/nlpcc-iccpol-2016.kbqa.kb.mention2id')
        self.mention2entity_clean_path = os.path.join(root_path, 'data/data120603/nlpcc-iccpol-2016_mention2id_clean.json')
        self.forward_index_path = os.path.join(root_path, 'data/data120603/forward_index.json')
        self.knowledge_graph_path = os.path.join(root_path, 'data/data120603/nlpcc-iccpol-2016.kbqa.kb')


class CLSConfig():
    def __init__(self):
        self.best_model_path = os.path.join(root_path, "data/data122049/ernie_cls_best.pdparams")


class NERConfig():
    def __init__(self):
        self.best_model_path = os.path.join(root_path, "data/data122049/ernie_ner_best.pdparams")


class Word2VecConfig():
    def __init__(self):
        self.model_path = os.path.join(root_path, 'data/data122049/sgns.target.word-character')
