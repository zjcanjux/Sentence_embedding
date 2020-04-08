# Sentence_embedding
Unsupervised sentence embedding method 非监督的句向量编码  
## 主要实现一些非监督的句向量编码方法  
已经实现的有:  
1.average  最简单的句向量方法，即基于词向量的求和平均  
2.tfidf    基于tfidf的加权平均，即每个词的词向量乘以其的tfidf值  
3.usif     是文章 Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline 中的方法  
目前用的词向量出自，https://github.com/Embedding/Chinese-Word-Vectors  
## 相似度测试结果  
usif  
第一句 买 草莓 和 苹果 // 第二句 我 要 买 手机,得分 -0.51795  
第一句 我 去 超市 买 手机 // 第二句 我 要 买 手机,得分 0.53539  
第一句 我 去 超市 买 手机 // 第二句 买 草莓 和 苹果,得分 -0.63173  
第一句 周末 和 朋友 去 公园 玩 // 第二句 我 周末 和 朋友 出去 玩,得分 0.64078  
第一句 买 草莓 和 苹果 // 第二句 我 周末 和 朋友 出去 玩,得分 -0.64849  
tfidf  
第一句 买 草莓 和 苹果 // 第二句 我 要 买 手机,得分 0.41765  
第一句 我 去 超市 买 手机 // 第二句 我 要 买 手机,得分 0.68814  
第一句 我 去 超市 买 手机 // 第二句 买 草莓 和 苹果,得分 0.42734  
第一句 周末 和 朋友 去 公园 玩 // 第二句 我 周末 和 朋友 出去 玩,得分 0.75556  
第一句 买 草莓 和 苹果 // 第二句 我 周末 和 朋友 出去 玩,得分 0.28242  
average  
第一句 买 草莓 和 苹果 // 第二句 我 要 买 手机,得分 0.61223  
第一句 我 去 超市 买 手机 // 第二句 我 要 买 手机,得分 0.89204  
第一句 我 去 超市 买 手机 // 第二句 买 草莓 和 苹果,得分 0.60939  
第一句 周末 和 朋友 去 公园 玩 // 第二句 我 周末 和 朋友 出去 玩,得分 0.88357  
第一句 买 草莓 和 苹果 // 第二句 我 周末 和 朋友 出去 玩,得分 0.46401  

## requirements
scikit-learn  
torchtext  
numpy

## To-do
- Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features

