# ============================================================================
# ChatPDF + FAISS：最小可用的完整 RAG 链路
#
# 这个脚本是整个仓库里最值得精读的一份，因为它把 RAG 的主链路一次跑通了：
#
#   PDF 解析  →  文本切分  →  向量化  →  建向量库  →  检索  →  拼 Prompt  →  LLM 生成  →  标注来源
#   (PyPDF2)   (Splitter)  (agicto)    (FAISS)   (相似度)   (模版)      (agicto)    (页码溯源)
#
# 接口说明：本脚本通过 agicto 聚合平台调用模型，使用 OpenAI 兼容协议。
#   Base URL: https://api.agicto.cn/v1/
#   需要环境变量 AGICTO_API_KEY（详见下方配置区）
#
# 对照 bge-m3使用.py 理解：
#   bge-m3 那个脚本只演示了「向量化 + 算相似度」这两步，
#   本脚本则把「向量化」真正用到了工程化的检索流程里 —— 差别在于：
#   1. 文本量大，必须切分成 chunk（bge-m3 脚本里是 4 条短句，不需要切）
#   2. 向量量大，不能靠矩阵乘法暴力比对，必须用 FAISS 做近邻检索
#   3. 检索结果要喂给 LLM 生成答案，并给出出处
#
# 另外这个脚本额外做了「页码溯源」：回答完问题后告诉用户答案来自 PDF 第几页，
# 这是 RAG 落地时非常实际的需求（让用户能核实答案），大部分教程都不讲。
# ============================================================================

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle

# ---------------------------------------------------------------------------
# 配置区：API Key 与模型名统一从环境变量读取，不硬编码在代码里（避免提交到 git 泄露）
#
# 运行前需先在服务器上配置：
#   export AGICTO_API_KEY="sk-xxxxxxxx"
# Key 去 https://agicto.com 控制台申请；模型名可在 https://agicto.com/model 查询
#
# 为什么是 agicto 而不是 DashScope 原生？
#   agicto 是聚合平台，走 OpenAI 兼容协议（/v1/chat/completions、/v1/embeddings），
#   所以这里用 OpenAIEmbeddings / ChatOpenAI 而不是 DashScopeEmbeddings / Tongyi。
#   好处是以后想换其他厂商，只改 base_url 和模型名即可，代码不用动。
#
# 注意：embedding 模型一旦更换，已建好的向量库就作废了 ——
#      不同模型产出的向量不在同一语义空间，必须先删掉 vector_db 重新建库
# ---------------------------------------------------------------------------
AGICTO_API_KEY = os.getenv('AGICTO_API_KEY')
AGICTO_BASE_URL = os.getenv('AGICTO_BASE_URL', 'https://api.agicto.cn/v1/')
EMBEDDING_MODEL = os.getenv('AGICTO_EMBEDDING_MODEL', 'text-embedding-3-small')
CHAT_MODEL = os.getenv('AGICTO_CHAT_MODEL', 'gpt-4o-mini')

# fail-fast 校验：没配 Key 就直接抛错，而不是等到调用时才报奇怪的错
if not AGICTO_API_KEY:
    raise ValueError("请设置环境变量 AGICTO_API_KEY")


def build_embeddings():
    """
    构造向量化模型（embedding）

    单独抽成函数，是因为建库和加载库两处都要用，
    且必须保证两处参数完全一致 —— 参数不一致会导致查询向量和库内向量不匹配
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=AGICTO_BASE_URL,
        api_key=AGICTO_API_KEY,
        # 关键参数：关掉本地 token 预处理
        # 默认情况下 langchain 会用 tiktoken 把文本切成一串 token id 再发给接口，
        # 但很多第三方聚合网关（包括 agicto）只接受原始字符串形式的 input，
        # 传 token 数组会直接报参数解析错误，所以必须设为 False
        check_embedding_ctx_length=False,
    )


# ---------------------------------------------------------------------------
# 函数 1：从 PDF 提取文本，并记录「每个字符属于第几页」
#
# 为什么搞这么麻烦，不直接 pdf.pages[i].extract_text()？
#   因为后面要按固定长度把全文切成 chunk，一个 chunk 很可能横跨 2 页。
#   如果我们只保留「页码 → 文本」的对应关系，切分后就无法反查 chunk 来自哪页了。
#   所以这里退一步，维护「字符位置 → 页码」的映射：
#   char_page_mapping[i] 表示 text[i] 这个字符出自第几页。
#   切分后取 chunk 对应区间的页码众数，就能推断出 chunk 的主页码。
#
# 返回：
#   text               —— 全文拼接后的字符串
#   char_page_mapping  —— 与 text 等长的列表，逐字符记页码
# ---------------------------------------------------------------------------
def extract_text_with_page_numbers(pdf) -> Tuple[str, List[Tuple[str, int]]]:
    """
    从PDF中提取文本并记录每个字符对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
    """
    text = ""
    char_page_mapping = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            # 为当前页面的每个字符记录页码
            # 这里用 extend + 重复列表的方式，让 mapping 长度始终和 text 对齐
            char_page_mapping.extend([page_number] * len(extracted_text))
        else:
            # 扫描版 PDF（图片型）提取不到文字，需要先做 OCR 才能处理
            print(f"No text found on page {page_number}.")

    return text, char_page_mapping


# ---------------------------------------------------------------------------
# 函数 2：切分文本 + 向量化 + 建库 + 存盘
#
# 完整做了四件事：
#   ① 切分 chunk  ② 调 agicto 生成向量  ③ 建 FAISS 索引  ④ 持久化到磁盘
#
# 参数 save_path 传了就会存盘，下次可直接 load，不用重新花钱调 embedding API
# ---------------------------------------------------------------------------
def process_text_with_splitter(text: str, char_page_mapping: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    #
    # separators 是「优先级递减」的分隔符列表，切分时会依次尝试：
    #   先按空行切 → 太长就按换行切 → 再太长按句号切 → 再按空格 → 最后硬切
    # 这样能尽量保证切出来的块是语义完整的段落，不会把一句话从中间劈开
    #
    # chunk_size=1000：每块目标长度（字符数，不是 token 数）
    #   调参经验：太小 → 语义不完整、检索命中率高但答不出；太大 → 噪声多、稀释语义
    #   中文一般 300~800 比较合适，这里是配合 1000 的示例值
    # chunk_overlap=200：相邻块重叠 200 字符
    #   作用：防止关键句子正好落在切分边界上，被劈成两半导致两边都检索不到
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 创建嵌入模型
    # 这里用的是 agicto 聚合平台上的在线 embedding 服务，不是本地模型。
    # 用在线服务的好处：不用下模型、不用占显存；坏处：要花钱、要联网、有 QPS 限制。
    # 想换成本地模型（如 bge-m3），只需把 build_embeddings() 换成 HuggingFaceEmbeddings(...)，
    # 后面所有代码都不用改 —— 这就是 LangChain 抽象出 Embeddings 接口的价值
    embeddings = build_embeddings()
    
    # 从文本块创建知识库
    # from_texts 内部做了三件事：对每个 chunk 调 embedding API → 构建 FAISS 索引 → 返回封装对象
    # FAISS 用的是 L2 距离建索引，检索时内部会转换，效果等价于余弦相似度
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")

    # -----------------------------------------------------------------------
    # 页码溯源：把每个 chunk 映射回它在原 PDF 里的页码
    #
    # 思路：split_text 是顺序切分的，所以可以按 chunk 长度在全文里「顺序累加」
    # 还原出每个 chunk 在原文中的起止位置，再用这个区间去查 char_page_mapping。
    #
    # 注意这里的实现是简化版：chunk_overlap 会导致位置有偏差，
    # 但对于「定位到大致第几页」这个需求来说精度足够
    # -----------------------------------------------------------------------
    page_info = {}
    current_pos = 0
    
    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        
        # 找到这个文本块中字符对应的页码
        chunk_pages = char_page_mapping[chunk_start:chunk_end]
        
        # 取页码的众数（出现最多的页码）作为该块的页码
        if chunk_pages:
            # 统计每个页码出现的次数
            page_counts = {}
            for page in chunk_pages:
                page_counts[page] = page_counts.get(page, 0) + 1
            
            # 找到出现次数最多的页码
            # 用众数而非首字符所在页，是为了处理「chunk 横跨两页」的情况：
            # 哪个页贡献的字符多，就认为这块主要来自哪一页
            most_common_page = max(page_counts, key=page_counts.get)
            page_info[chunk] = most_common_page
        else:
            page_info[chunk] = 1  # 默认页码
        
        current_pos = chunk_end
    
    # 把映射挂到 knowledgeBase 对象上，方便后续查询时直接取用
    knowledgeBase.page_info = page_info
    print(f'页码映射完成，共 {len(page_info)} 个文本块')
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        # 会生成两个文件：index.faiss（向量索引）+ index.pkl（原始文本和元数据）
        # 注意：这里只存了向量，没存 embedding 模型本身，所以加载时必须传入同一个模型
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        # page_info 是 Python 字典，FAISS 自己不会存，所以用 pickle 单独存一份
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")
    
    return knowledgeBase


# ---------------------------------------------------------------------------
# 函数 3：从磁盘加载已建好的向量库
#
# 典型用法：建库是一次性开销（费钱费时），之后每次问答都直接 load，秒级启动
# 关键约束：加载时必须用和建库时「完全相同」的 embedding 模型，
#          否则查询向量和库里的向量不在同一语义空间，检索结果会完全错乱
# ---------------------------------------------------------------------------
def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将调用 build_embeddings() 创建一个
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = build_embeddings()
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    # 安全提示：这个参数为 True 意味着加载时会执行 pickle 反序列化，
    # 恶意构造的 index.pkl 可以执行任意代码。只加载自己生成的向量库，别加载来路不明的
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase


# ===========================================================================
# 主流程开始
# ===========================================================================

# 读取PDF文件
# 样例文档是一份银行内部考核办法，属于典型的「私有知识库」场景 ——
# 这类内容大模型训练时没见过，正是 RAG 要解决的问题
pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, char_page_mapping = extract_text_with_page_numbers(pdf_reader)
#print('page_numbers=',page_numbers)


# In[9]:


print(f"提取的文本长度: {len(text)} 个字符。")
    
# 处理文本并创建知识库，同时保存到磁盘
# 注意：这一步会真实调用 embedding API 并按字符数计费。
# 所以第一次跑完之后，后续调试建议走 load_knowledge_base，不要重复建库
save_dir = "./vector_db"
knowledgeBase = process_text_with_splitter(text, char_page_mapping, save_path=save_dir)

# 示例：如何加载已保存的向量数据库
# 注释掉以下代码以避免在当前运行中重复加载
# ⚠️ 重要：改用 agicto 的 embedding 模型后，旧的 vector_db 必须重建 ——
#    旧的库是用 DashScope text-embedding-v1 建的，向量维度和语义空间都不同，
#    直接加载去检索会得到完全错乱的结果（不报错，但答案全是错的，很难排查）。
#    所以首次运行请先删掉旧库：rm -rf ./vector_db
"""
# 从磁盘加载向量数据库（embedding 会自动用 build_embeddings() 创建）
loaded_knowledgeBase = load_knowledge_base("./vector_db")
# 使用加载的知识库进行查询
docs = loaded_knowledgeBase.similarity_search("客户经理每年评聘申报时间是怎样的？")

# 直接使用FAISS.load_local方法加载（替代方法）
# loaded_knowledgeBase = FAISS.load_local("./vector_db", build_embeddings(), allow_dangerous_deserialization=True)
# 注意：使用这种方法加载时，需要手动加载页码信息
"""


# In[11]:


# ---------------------------------------------------------------------------
# 生成环节：把检索到的上下文拼进 Prompt 交给 LLM
#
# 这就是 RAG 的 "A"（Augmented，增强）：LLM 本身不知道这份考核办法的内容，
# 我们把检索到的原文塞进 Prompt，让它「开卷答题」，答案就有依据了
# ---------------------------------------------------------------------------
# 构造对话模型（LLM）
# 这里用 ChatOpenAI 而不是 Tongyi —— 因为 agicto 走的是 OpenAI 兼容协议。
# 模型名由环境变量 AGICTO_CHAT_MODEL 决定，默认 gpt-4o-mini（便宜、够用）。
# 想换模型只改环境变量即可，比如：
#   export AGICTO_CHAT_MODEL="gpt-4o"        # 效果更好、更贵
#   export AGICTO_CHAT_MODEL="deepseek-chat" # 性价比高
#   export AGICTO_CHAT_MODEL="qwen-turbo"    # 通义，agicto 上同样可调
# temperature=0 表示「尽量确定性输出」——
# RAG 场景要的是忠实于检索内容，而不是发挥创造力，温度调高反而容易编造
llm = ChatOpenAI(
    model=CHAT_MODEL,
    base_url=AGICTO_BASE_URL,
    api_key=AGICTO_API_KEY,
    temperature=0,
)

# 设置查询问题
query = "客户经理被投诉了，投诉一次扣多少分"
#query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    # k=10 表示取最相关的 10 个 chunk 作为上下文。
    # k 的取舍：太小 → 漏掉关键信息；太大 → 噪声增多 + Prompt 变长导致成本上升、
    #            且 LLM 容易被无关内容干扰。实务上常配合 Rerank 精排来兼顾召回和精度
    docs = knowledgeBase.similarity_search(query,k=10)

    # 构建上下文
    # 把 10 个 chunk 用空行拼成一段长文本，作为 Prompt 里的「参考资料」
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构建提示
    # 这是最朴素的 RAG Prompt，只有「上下文 + 问题」。
    # 生产环境通常还会加：角色设定、引用要求（必须标注来源）、
    # 以及最重要的一句「若上下文中没有答案，请回答不知道」——
    # 不加这句，模型很容易在检索失败时靠训练记忆编造答案（幻觉）
    prompt = f"""根据以下上下文回答问题:

{context}

问题: {query}"""

    # 直接调用 LLM
    # ChatOpenAI 的 invoke() 返回的是 AIMessage 对象（不止正文，还带 token 用量等元信息），
    # 所以要取 .content 才是模型回答的文本本身 —— 这点和 Tongyi 直接返回字符串不同
    response = llm.invoke(prompt)
    print(response.content)
    print("来源:")

    # 记录唯一的页码
    # 用 set 去重：多个 chunk 可能来自同一页，只展示一次避免刷屏
    unique_pages = set()

    # 显示每个文档块的来源页码
    for doc in docs:
        #print('doc=',doc)
        text_content = getattr(doc, "page_content", "")
        # 用 chunk 原文去 page_info 里反查页码（key 就是 chunk 文本本身）
        # strip() 要和建库时的 key 保持一致，否则查不到
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")

# ---------------------------------------------------------------------------
# 【RAG 实践要点】
# 1. 这个脚本是「朴素 RAG」的完整形态，也是所有优化工作的基线。
#    遇到效果不好的问题，先定位是「检索没召回」还是「召回了但没答对」：
#      - 检索问题 → 优化切分、换更好的 embedding、加 Rerank
#      - 生成问题 → 优化 Prompt、换更强的 LLM
# 2. 成本敏感场景注意：embedding 按输入 token 计费，重复建库纯属浪费。
#    定稿后的向量库应该持久化并纳入版本管理
# 3. similarity_search 返回的是「向量最近的 k 个」，不一定「都是相关的」。
#    更好的做法是用 similarity_search_with_score 拿到相似度分数，
#    低于阈值的结果直接丢弃，避免把无关内容当上下文喂给 LLM
# 4. 页码溯源用的是「字符位置累加 + 取众数」的简化算法，
#    受 chunk_overlap 影响会有偏差。如果要精确到页，
#    建议改用 FAISS.from_documents 并给每个 chunk 挂 metadata，更可靠
# 5. 如果 PDF 是扫描件（图片型），extract_text() 拿不到任何文字，
#    需要先接 OCR（如 PaddleOCR / 阿里云文档解析），本脚本无法直接处理
# ---------------------------------------------------------------------------
