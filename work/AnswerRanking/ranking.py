import jieba


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
    for seg in words1:
        if seg not in word2vec_model.vocab:
            _ws = [_w for _w in seg if _w in word2vec_model.vocab]
            de_seg1.extend(_ws)
        else:
            de_seg1.append(seg)
    for seg in words2:
        if seg not in word2vec_model.vocab:
            _ws = [_w for _w in seg if _w in word2vec_model.vocab]
            de_seg1.extend(_ws)
        else:
            de_seg2.append(seg)
    if de_seg1 and de_seg2:
        score = word2vec_model.n_similarity(de_seg1, de_seg2)
    else:
        score = 0
    return score
