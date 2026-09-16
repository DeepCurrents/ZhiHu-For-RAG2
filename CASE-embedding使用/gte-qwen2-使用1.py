#!/usr/bin/env python
# coding: utf-8

# ============================================================================
# GTE-Qwen2 嵌入模型使用示例（SentenceTransformer 高层封装版）
#
# 与 bge-m3使用.py 的对比：
#   两个脚本都在做「文本 → 向量 → 相似度」，区别在「怎么加载模型、怎么编码」：
#   - bge-m3  → 用 FlagEmbedding 的 BGEM3FlagModel，一次输出 dense/sparse/colbert
#   - gte-Qwen2 → 用 sentence-transformers 的 SentenceTransformer，只输出单向量
#
# GTE-Qwen2 的特点：
#   基于 Qwen2 大模型改造的 Embedding 模型，属于「LLM-based Embedding」路线，
#   靠大模型的语义理解能力换来更强的检索效果，代价是显存和推理开销更大。
#   本文件用 1.5B 小版本跑通流程；7B 版本效果更好但需要更大显存。
#
# 同目录的 gte-qwen2-使用2.py 是「不用封装、手写池化」的等价实现，
# 建议两个文件对照着看，能理解封装层到底替你做了什么。
# ============================================================================

# In[2]:


# ---------------------------------------------------------------------------
# 步骤 1：下载模型权重
#   同样走 ModelScope 镜像，避免国内直连 HuggingFace 超时。
#   注意这里有两行，只保留一行、注释掉另一行，用来切换模型规格：
#   - iic/gte_Qwen2-7B-instruct   ：效果更好，显存需求高（fp16 约 15GB+）
#   - iic/gte_Qwen2-1.5B-instruct ：轻量版（当前启用），入门/调试用
#     24G 显存的 4090 两个都能跑，但 1.5B 启动快很多，适合先跑通流程
# ---------------------------------------------------------------------------
#模型下载
from modelscope import snapshot_download
#model_dir = snapshot_download('iic/gte_Qwen2-7B-instruct', cache_dir='/root/autodl-tmp/models')
model_dir = snapshot_download('iic/gte_Qwen2-1.5B-instruct', cache_dir='/root/autodl-tmp/models')


# In[1]:


# ---------------------------------------------------------------------------
# 步骤 2：加载模型
#   SentenceTransformer 是 sentence-transformers 库的统一入口，
#   它能直接读 HuggingFace 格式的模型目录，自动处理池化(pooling)和归一化。
#
#   两个参数说明：
#   - model_dir：模型本地路径。注意 ModelScope 会把仓库名里的 "." 转成 "___"，
#     所以下载的是 1.5B，但目录名是 1___5B —— 这不是笔误，是 ModelScope 的命名规则
#   - trust_remote_code=True：允许执行模型仓库里自带的 Python 代码。
#     GTE-Qwen2 用了自定义的网络结构，不加这个参数会加载失败。
#     安全提示：只对可信来源的模型开启（BAAI/iic 这类官方仓库是安全的）
# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

model_dir = "/root/autodl-tmp/models/iic/gte_Qwen2-1___5B-instruct"
model = SentenceTransformer(model_dir, trust_remote_code=True)

# ---------------------------------------------------------------------------
# 步骤 3：设置最大序列长度
#   这里设 8192，意味着超长文本会被完整编码，不会截断。
#   代价：注意力计算随长度呈平方增长，长文本会让显存和耗时急剧上升。
#   实务建议：RAG 里 chunk 一般只有 300~1000 token，把这里设成 1024 就够了，
#   能省下大量显存 —— 真正需要 8192 的场景很少
# ---------------------------------------------------------------------------
# In case you want to reduce the maximum length:
model.max_seq_length = 8192

# ---------------------------------------------------------------------------
# 步骤 4：准备 query 和 document
#   同样模拟 RAG 里「1 条查询 vs N 条候选文档」的检索场景
# ---------------------------------------------------------------------------
queries = [
    "how much protein should a female eat",
    "summit define",
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# ---------------------------------------------------------------------------
# 步骤 5：编码 —— 注意 query 和 document 走的是不同分支
#   query 端传了 prompt_name="query"，document 端什么都没传。
#
#   为什么必须区分？
#   GTE-Qwen2 是「指令式(instruct)」模型：它要求在查询前面拼接一段任务指令
#   （如 "Instruct: Given a web search query, retrieve relevant passages\nQuery: ..."），
#   才能发挥出训练时的效果。SentenceTransformer 把这段指令预置在了模型的
#   config 里，用 prompt_name="query" 就能自动套用，不用自己拼字符串。
#
#   关键点：文档端不加指令、查询端才加。两边搞反或都加，检索效果会明显下降 ——
#   这是新手最容易踩的坑。gte-qwen2-使用2.py 里可以看到指令被手动拼出来的样子
# ---------------------------------------------------------------------------
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# ---------------------------------------------------------------------------
# 步骤 6：计算相似度
#   (2, d) @ (d, 2) → (2, 2) 相似度矩阵，逻辑与 bge-m3 完全一致。
#   这里额外乘 100 把分数放大到 0~100，纯粹为了肉眼看着直观，不影响排序结果。
#   注意：SentenceTransformer 输出的向量默认已归一化，所以点积即余弦相似度
# ---------------------------------------------------------------------------
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())
# [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
# 解读：
#   70.01 → query0(女性蛋白质摄入) 对 doc0(蛋白质指南)  ：高相关 ✓
#    8.18 → query0 对 doc1(summit 定义)                  ：不相关 ✓
#   14.62 → query1(summit 定义) 对 doc0(蛋白质指南)      ：不相关 ✓
#   77.71 → query1 对 doc1(summit 定义)                  ：高相关 ✓
#   对角线分数远高于非对角线，说明模型区分能力正常

# ---------------------------------------------------------------------------
# 【RAG 实践要点】
# 1. 别直接用这个脚本的分数当阈值：乘 100 之后数值范围变了，
#    实际业务里应统一用未缩放的余弦相似度（0~1）来定阈值
# 2. 显存换效果：GTE-Qwen2 走的是「大模型当 Embedding」的路线，
#    效果通常优于 bge-m3，但推理慢一个量级。
#    选型建议：离线建库可以用大模型，线上实时查询用 bge-m3 这类轻量模型
# 3. 建库时用的编码方式必须和查询时一致：包括模型、max_seq_length、
#    以及是否加 prompt。中途换模型会导致整个向量库失效，必须重建
# 4. 批量编码务必用 encode(..., batch_size=N) 而不是 for 循环单条编码，
#    能充分利用 GPU 并行，通常有数倍到十几倍加速
# ---------------------------------------------------------------------------
