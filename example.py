from usif import word2prob,my_word2vec,sent_uSIF
import numpy as np


sentence_list = ['我 去 超市 买 手机','我 要 买 手机','买 草莓 和 苹果','周末 和 朋友 去 公园 玩','我 周末 和 朋友 出去 玩']

vec = my_word2vec('/Users/liangrong/Deep_learning/pretrained_models/sgns.sogou.word')
prob = word2prob('./data/word_count.txt')
my_usif = sent_uSIF(vec,prob)
my_vectors = my_usif.embed(sentence_list)

def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])

def cosine_similarity(x, y, norm=False):

    assert len(x) == len(y)
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos

print('usif')
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[1],cosine_similarity(list(my_vectors[2]),list(my_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[1],cosine_similarity(list(my_vectors[0]),list(my_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[2],cosine_similarity(list(my_vectors[0]),list(my_vectors[2]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[3],sentence_list[4],cosine_similarity(list(my_vectors[3]),list(my_vectors[4]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[4],cosine_similarity(list(my_vectors[2]),list(my_vectors[4]))))


from tfidf import sent_tfidf

my_tfidf = sent_tfidf(vec)
tfidf_vectors = my_tfidf.embed(sentence_list)


print('tfidf')
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[1],cosine_similarity(list(tfidf_vectors[2]),list(tfidf_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[1],cosine_similarity(list(tfidf_vectors[0]),list(tfidf_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[2],cosine_similarity(list(tfidf_vectors[0]),list(tfidf_vectors[2]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[3],sentence_list[4],cosine_similarity(list(tfidf_vectors[3]),list(tfidf_vectors[4]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[4],cosine_similarity(list(tfidf_vectors[2]),list(tfidf_vectors[4]))))

from tfidf import sent_average

my_average = sent_average(vec)
average_vectors = my_average.embed(sentence_list)

print('average')
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[1],cosine_similarity(list(average_vectors[2]),list(average_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[1],cosine_similarity(list(average_vectors[0]),list(average_vectors[1]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[0],sentence_list[2],cosine_similarity(list(average_vectors[0]),list(average_vectors[2]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[3],sentence_list[4],cosine_similarity(list(average_vectors[3]),list(average_vectors[4]))))
print("第一句 {} // 第二句 {},得分 {}".format(sentence_list[2],sentence_list[4],cosine_similarity(list(average_vectors[2]),list(average_vectors[4]))))
