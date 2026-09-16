#!/usr/bin/env python
# coding: utf-8

# ============================================================================
# BGE-M3 嵌入（Embedding）模型使用示例
#
# 在 RAG（检索增强生成）流程中的定位：
#   文档 → 切分(chunk) → 【Embedding 向量化】 → 向量库 → 检索 → LLM 生成
#   本脚本演示最核心的一步：把文本转成向量，并用向量相似度判断语义相关性。
#
# BGE-M3 的三个 "M" 能力：
#   1. Multi-Linguality（多语言）：支持 100+ 种语言，中英混合无需额外处理
#   2. Multi-Functionality（多功能）：同时输出 dense(稠密) / sparse(稀疏) / colbert(多向量) 三种表示
#      - dense_vecs    : 单向量，语义检索，本脚本使用的方式
#      - lexical_weights: 稀疏权重，等价于学习式 BM25，适合关键词精确匹配
#      - colbert_vecs  : 多向量交互，精度最高但存储和计算开销最大
#   3. Multi-Granularity（多粒度）：最长支持 8192 token，短句到长文档均可处理
# ============================================================================

# In[1]:


# ---------------------------------------------------------------------------
# 步骤 1：下载模型权重
#   国内服务器直接从 HuggingFace 下载容易超时，这里用 ModelScope（魔搭）镜像。
#   - 'BAAI/bge-m3'：模型在 ModelScope 上的仓库 ID
#   - cache_dir：模型落盘目录。放在 /root/autodl-tmp 是 AutoDL 的数据盘，
#     空间大且不会占用系统盘；bge-m3 约 2.2GB，务必别放 /root 下
#   - 返回的 model_dir 是实际下载路径，供后续加载使用
# ---------------------------------------------------------------------------
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='/root/autodl-tmp/models')


# In[1]:


# ---------------------------------------------------------------------------
# 步骤 2：加载模型
#   BGEM3FlagModel 是 FlagEmbedding 提供的高层封装，比裸 transformers 更好用：
#   - 可直接通过 encode() 拿到 dense/sparse/colbert 三种向量，无需自己写池化逻辑
#   - use_fp16=True：使用半精度推理，显存占用减半、速度更快，
#     精度损失极小（通常 <0.5%），生产环境一般默认开启
# ---------------------------------------------------------------------------
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('/root/autodl-tmp/models/BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# ---------------------------------------------------------------------------
# 步骤 3：准备待编码文本
#   这里构造了两组文本模拟 RAG 中的典型场景：
#   - sentences_1 扮演「用户 query」
#   - sentences_2 扮演「知识库文档 chunk」
#   预期语义对应关系：query[0]↔doc[0]（都讲 BGE M3）、query[1]↔doc[1]（都讲 BM25）
# ---------------------------------------------------------------------------
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# ---------------------------------------------------------------------------
# 步骤 4：文本向量化（encode）
#   - batch_size=12：每批处理 12 条文本，显存不足时可调小（如 4/8）
#   - max_length=8192：截断长度上限。bge-m3 支持 8192，但长度越长显存和耗时越高；
#     若文档 chunk 只有几百字，设成 512 能显著提速
#   - encode() 返回一个字典，['dense_vecs'] 取出稠密向量
#     （另外还有 ['lexical_weights'] 稀疏权重、['colbert_vecs'] 多向量）
#   - 注意：第二条 encode 未传参，会使用默认 max_length，效果相同
# ---------------------------------------------------------------------------
embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']

# ---------------------------------------------------------------------------
# 步骤 5：计算相似度 —— RAG 检索的本质就是这一步
#   embeddings_1 形状：(2, 1024)  即 2 条 query，每条 1024 维
#   embeddings_2 形状：(2, 1024)  即 2 条 doc，每条 1024 维
#   @ 是矩阵乘法，(2,1024) @ (1024,2) → (2,2) 的相似度矩阵：
#            doc0    doc1
#   query0 [ s00     s01 ]
#   query1 [ s10     s11 ]
#   bge-m3 输出的向量已做 L2 归一化，因此点积结果直接等于余弦相似度，
#   取值范围 [-1, 1]，越大越相关
# ---------------------------------------------------------------------------
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
# 对角线 0.6265 / 0.6780 明显高于非对角线 0.3477 / 0.3499，
# 说明模型正确地把语义相同的 query 和 doc 拉近了 —— 这就是向量检索能生效的原因

# ---------------------------------------------------------------------------
# 【RAG 实践要点】
# 1. 相似度阈值：余弦相似度 >0.5 通常认为相关，但具体阈值需用业务数据实测标定
# 2. 生产环境不要用暴力矩阵乘法：几万条以上就该上 FAISS / Milvus 等向量库做 ANN 检索
# 3. query 与 doc 的处理：bge-m3 无需加 "query:" / "passage:" 指令前缀，
#    这点与 bge-large-zh 等老版本模型不同（后者必须加前缀否则效果显著下降）
# 4. 三种向量可组合使用：dense + sparse 混合检索（Hybrid Search）通常比单用 dense 效果好
# ---------------------------------------------------------------------------
