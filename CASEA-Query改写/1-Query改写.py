# ============================================================================
# Query 改写：检索前的「查询优化」环节
#
# 为什么需要 Query 改写？
#   用户说话是「口语化、依赖上下文、带情绪」的，而向量库里存的是「书面化、
#   自包含」的文档。直接用用户原话去检索，经常召不回正确结果 ——
#   这时候问题不在向量库，而在于「查询」和「文档」的表述方式不匹配。
#   改写就是把用户的自然语言，翻译成「适合检索」的查询。
#
# 本脚本覆盖 5 类典型场景（每类对应一个方法）：
#   1. 上下文依赖型 —— "还有其他设施吗？"         → 补全成完整问题
#   2. 对比型       —— "哪个游玩时间比较长？"     → 明确出要比较的对象
#   3. 模糊指代型   —— "都什么时候开始？"         → 把"都"替换成具体主体
#   4. 多意图型     —— "门票多少钱？要预约吗？"   → 拆成多个独立子问题
#   5. 反问型       —— "这不会也要预约一个月吧？" → 还原成客观的中立问题
#
# 再加上一个 auto_rewrite_query() 做「类型识别 + 自动路由」，
# 这才是能直接接进 RAG 流水线的形态。
#
# 在 RAG 链路中的位置：
#   用户提问 → 【Query 改写】 → 向量检索 → Rerank → LLM 生成
#               ↑ 本脚本
#   它在整个链路的最前端，成本最低（一次轻量 LLM 调用），收益却很高 ——
#   先把检索这一步做对，后面所有环节才有意义。
# ============================================================================

# Query改写使用示例
# 导入依赖库
from openai import OpenAI
import os
import json

# ---------------------------------------------------------------------------
# 配置区：API Key 与模型名统一从环境变量读取，不硬编码在代码里（避免提交到 git 泄露）
#
# 运行前需先在服务器上配置：
#   export AGICTO_API_KEY="sk-xxxxxxxx"
# Key 去 https://agicto.com 控制台申请；模型名可在 https://agicto.com/model 查询
#
# 为什么改成 OpenAI SDK？
#   本脚本原来用的是 DashScope 原生 SDK（dashscope.Generation.call），
#   而 agicto 是聚合平台、走 OpenAI 兼容协议，两边协议不通用 ——
#   拿 agicto 的 Key 去调 DashScope SDK 会直接鉴权失败。
#   换成 openai SDK 后，以后想换厂商只需改 base_url 和模型名，代码不用动。
# ---------------------------------------------------------------------------
AGICTO_API_KEY = os.getenv('AGICTO_API_KEY')
AGICTO_BASE_URL = os.getenv('AGICTO_BASE_URL', 'https://api.agicto.cn/v1/')
CHAT_MODEL = os.getenv('AGICTO_CHAT_MODEL', 'gpt-4o-mini')

# fail-fast 校验：没配 Key 就直接抛错，而不是等到调用时才报奇怪的错
if not AGICTO_API_KEY:
    raise ValueError("请设置环境变量 AGICTO_API_KEY")

# OpenAI 兼容客户端：把 base_url 指向 agicto 即可，其余用法和调 OpenAI 完全一致
client = OpenAI(api_key=AGICTO_API_KEY, base_url=AGICTO_BASE_URL)

# ---------------------------------------------------------------------------
# 通用工具函数：调用 LLM 生成文本
#
# 这是所有改写方法的底层入口。两个关键设计：
#
# ① temperature=0
#    改写属于「确定性任务」——同样的输入应该永远得到同样的改写结果。
#    温度设为 0 让模型每步都选概率最高的 token，避免同一问题时而改写、
#    时而不改写，导致检索结果飘忽不定，线上很难排查
#
# ② 模型名统一从环境变量来
#    默认 gpt-4o-mini（便宜、够用）。想换模型只改环境变量，比如：
#      export AGICTO_CHAT_MODEL="deepseek-chat"  # 性价比高
#      export AGICTO_CHAT_MODEL="qwen-turbo"     # 通义，agicto 上同样可调
# ---------------------------------------------------------------------------
# 基于 prompt 生成文本
def get_completion(prompt, model=None):
    messages = [{"role": "user", "content": prompt}]
    # 不传 model 时用默认模型（便于调用方只想换一次全局默认值）
    response = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=messages,
        temperature=0,
    )
    # 失败时 openai SDK 会直接抛异常（APIError 等），不像 DashScope SDK
    # 那样把错误塞进 status_code 里静默返回 —— 所以这里不用再手动校验状态码。
    # 但仍要防一手「接口通了、choices 却为空」的边界情况
    if not response.choices:
        raise RuntimeError(f"agicto 未返回生成内容: {response}")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Query 改写的核心类：把 5 种改写策略 + 自动路由封装在一起
#
# 每个 rewrite_xxx 方法的结构完全一致，都是一个三段式模板：
#   instruction（角色 + 任务说明）→ prompt（指令 + 历史 + 问题）→ 调用 LLM
# 把 instruction 单独抽出来，是为了让 prompt 更好维护 —— 想调效果时
# 只改 instruction 那几行中文就行，不用在 f-string 里翻找
# ---------------------------------------------------------------------------
# Query改写功能
class QueryRewriter:
    # 不传 model 时使用配置文件里的默认模型 CHAT_MODEL
    def __init__(self, model=None):
        self.model = model or CHAT_MODEL

    # -----------------------------------------------------------------------
    # 类型 1：上下文依赖型
    #
    # 典型特征：出现「还有」「其他」「那」「上面说的」这类词，
    #          单独拿出来看根本不成立（"还有其他设施吗？"——什么的其他设施？）
    #
    # 为什么要改：向量检索是无状态的，它只看到这一句话。
    #            改写后把主语补全，检索才有明确方向
    #
    # 参数 conversation_history 需要把前几轮对话一起传进来，
    # 这是实现「多轮 RAG」的关键 —— 单轮 RAG 不需要这一步
    # -----------------------------------------------------------------------
    def rewrite_context_dependent_query(self, current_query, conversation_history):
        """上下文依赖型Query改写"""
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    # -----------------------------------------------------------------------
    # 类型 2：对比型
    #
    # 典型特征：出现「哪个」「比较」「更」「哪个更好」等比较词
    #
    # 为什么要改：这类问题的答案往往同时涉及多个对象，但向量检索是
    #            「一句话 → 一个向量」，如果原句里对象指代不清（"哪个"指谁？），
    #            召回的 chunk 可能只覆盖其中一个对象
    #            改写的目标是让「被比较的双方」都明确出现在查询里
    # -----------------------------------------------------------------------
    def rewrite_comparative_query(self, query, context_info):
        """对比型Query改写"""
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        
        return get_completion(prompt, self.model)
    
    # -----------------------------------------------------------------------
    # 类型 3：模糊指代型
    #
    # 典型特征：出现「都」「它」「这个」「他们」等代词
    #
    # 与类型 1 的区别：
    #   类型 1 缺的是「整句话的语境」，类型 3 缺的是「某个词指代的对象」，
    #   前者要补全整句，后者只需替换代词。所以两者的 instruction 不同
    # -----------------------------------------------------------------------
    def rewrite_ambiguous_reference_query(self, current_query, conversation_history):
        """模糊指代型Query改写"""
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    # -----------------------------------------------------------------------
    # 类型 4：多意图型 —— 唯一一个「一变多」的改写
    #
    # 典型特征：一句话里塞了多个问题，用「、」或「？」分隔
    #          例："门票多少钱？需要提前预约吗？停车费怎么收？"
    #
    # 为什么必须拆：三个问题揉成一句话编码，得到的向量是「三者的平均」——
    #              语义被稀释，检索时常常三个问题一个都答不好。
    #              拆开后每个子问题单独检索，命中率会明显提升
    #
    # 关键点：这里要求 LLM 输出 JSON 数组，是为了让结果能被程序直接消费。
    #        这也是「用 LLM 做数据处理」的常见范式 —— 让模型输出结构化数据，
    #        而不是自然语言，才能接进后续的代码流程
    # -----------------------------------------------------------------------
    def rewrite_multi_intent_query(self, query):
        """多意图型Query改写 - 分解查询"""
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以JSON数组格式输出。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请以JSON数组格式输出，例如：["问题1", "问题2", "问题3"]
"""
        
        response = get_completion(prompt, self.model)
        # LLM 不保证一定输出合法 JSON（可能带 ```json 代码块标记、或加解释文字），
        # 所以必须 try 兜底：解析失败就把整段响应当成单个问题返回，
        # 保证下游拿到的一定是可用的列表，不会因为解析异常中断整条链路
        try:
            return json.loads(response)
        except:
            return [response]
    
    # -----------------------------------------------------------------------
    # 类型 5：反问型
    #
    # 典型特征：带情绪的反问句，"这不会也要提前一个月预订吧？"
    #
    # 为什么要改：反问句里往往包含「错误的前提假设」或「否定表达」，
    #           直接拿去检索，向量会偏向情绪词（"不会"、"吧"），
    #           而不是真正的信息需求（"预订提前期是多久"）。
    #           改写的本质是「去掉情绪，还原成客观的知识性提问」
    # -----------------------------------------------------------------------
    def rewrite_rhetorical_query(self, current_query, conversation_history):
        """反问型Query改写"""
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    # -----------------------------------------------------------------------
    # 自动路由：先判断问题属于哪一类，再决定用哪个改写策略
    #
    # 这一步是「元判断」——不直接改写，而是先分类。
    # 好处是：① 每类问题都有专属的 prompt，比用一个大而全的 prompt 效果好
    #          ② 返回的 confidence 可以用来做兜底决策
    #             （置信度低时干脆不改写，用原问题检索，避免改坏）
    #
    # 注意指令里的优先级规则：多意图型 > 模糊指代型。
    # 这是必要的 —— 一句话可能同时符合两类特征（又有多问题、又有代词），
    # 不给优先级的话，LLM 的分类结果会在两类之间随机摇摆
    # -----------------------------------------------------------------------
    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回JSON格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": "置信度(0-1)"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 ###
"""
        
        response = get_completion(prompt, self.model)
        # 同样需要兜底：解析失败时返回一个「不改写」的默认结果，
        # 这样调用方永远能拿到结构完整的字典，不会 KeyError
        try:
            return json.loads(response)
        except:
            return {
                "query_type": "未知类型",
                "rewritten_query": query,
                "confidence": 0.5
            }
    
    # -----------------------------------------------------------------------
    # 完整流水线：识别类型 → 按类型调用对应的改写方法
    #
    # 注意这里做了「两阶段」处理：
    #   auto_rewrite_query 已经返回了一个 rewritten_query，
    #   但方法又按类型调了一遍专属改写函数。看起来是重复劳动，其实不是：
    #   - 第一阶段（auto_rewrite_query）的输出要求模型同时做分类和改写，
    #     任务复杂，质量不如专注单一任务的 prompt
    #   - 第二阶段用专属 prompt 重做一遍，质量更高
    #   代价是多一次 LLM 调用。如果对延迟敏感，可以只用第一阶段的结果
    #
    # 返回的字典同时保留了 original_query 和 rewritten_query，
    # 便于线上做 A/B 对比、也便于排查「是不是改写改坏了」
    # -----------------------------------------------------------------------
    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写，然后根据类型调用相应的改写方法"""
        # 首先进行自动识别
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        
        # 根据识别结果调用相应的改写方法
        # 用「包含」而不是「等于」来匹配类型，是因为 LLM 返回的
        # query_type 可能是 "上下文依赖型" 也可能是 "类型1：上下文依赖型"，
        # 用子串匹配能容忍这类格式波动，提高鲁棒性
        query_type = result.get('query_type', '')
        
        if '上下文依赖' in query_type:
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif '对比' in query_type:
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif '模糊指代' in query_type:
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif '多意图' in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif '反问' in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            # 对于其他类型，返回自动识别的改写结果
            # 识别不出类型时用第一阶段的改写结果，而不是原问题 ——
            # 保守但不至于完全没优化
            final_result = result.get('rewritten_query', query)
        
        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get('confidence', 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result
        }


# ---------------------------------------------------------------------------
# 演示入口：用「迪士尼主题乐园」客服场景串起 6 个示例
#
# 为什么用客服场景：它天然包含多轮对话、指代、比较、多问题、情绪反问，
# 是 Query 改写最能体现价值的场景（也是实际落地最多的场景）
# ---------------------------------------------------------------------------
def main():
    # 初始化Query改写器
    rewriter = QueryRewriter()    
    print("=== Query改写功能使用示例（迪士尼主题乐园） ===\n")
    
    # 示例1: 上下文依赖型Query
    # 输入 "还有其他设施吗？"，模型应结合历史改写为
    # "上海迪士尼乐园疯狂动物城园区还有哪些游乐设施？"
    print("示例1: 上下文依赖型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园的最新项目。"
AI: "上海迪士尼乐园最新推出了'疯狂动物城'主题园区，这里有朱迪警官和尼克狐的互动体验。"
用户: "这个园区有什么游乐设施？"
AI: "'疯狂动物城'园区目前有疯狂动物城警察局、朱迪警官训练营和尼克狐的冰淇淋店等设施。"
"""
    current_query = "还有其他设施吗？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_context_dependent_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例2: 对比型Query
    # "哪个" 指代不明，改写后应明确成「疯狂动物城 vs 蜘蛛侠」两个园区
    # 提示：这类问题拆成两个子查询分别检索、再合并结果，效果往往更好
    print("示例2: 对比型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园的最新项目。"
AI: "上海迪士尼乐园最新推出了疯狂动物城主题园区，还有蜘蛛侠主题园区"
"""
    current_query = "哪个游玩的时间比较长，比较有趣"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_comparative_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例3: 模糊指代型Query
    # "都" 指代「上海迪士尼和香港迪士尼」，不替换掉就无法确定检索哪个园区
    print("示例3: 模糊指代型Query")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园和香港迪士尼乐园的烟花表演。"
AI: "好的，上海迪士尼乐园和香港迪士尼乐园都有精彩的烟花表演。"
"""
    current_query = "都什么时候开始？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_ambiguous_reference_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例4: 多意图型Query
    # 一条查询含 3 个问题，期望输出 3 元素的 JSON 数组。
    # 拿到数组后，标准做法是「并行发起 3 次检索」再汇总给 LLM
    print("示例4: 多意图型Query")
    query = "门票多少钱？需要提前预约吗？停车费怎么收？"
    
    print(f"原始查询: {query}")
    
    result = rewriter.rewrite_multi_intent_query(query)
    print(f"分解结果: {result}\n")
    
    # 示例5: 反问型Query
    # "这不会也要提前一个月预订吧？" 带情绪和预设，改写后应还原为
    # "上海迪士尼乐园门票的提前预订时间是多久？" 这类中性问句
    print("示例5: 反问型Query")
    conversation_history = """
用户: "你好，我想预订下周六上海迪士尼乐园的门票。"
AI: "正在为您查询... 查询到下周六的门票已经售罄。"
用户: "售罄是什么意思？我朋友上周去还能买到当天的票。"
"""
    current_query = "这不会也要提前一个月预订吧？"
    
    print(f"对话历史: {conversation_history}")
    print(f"当前查询: {current_query}")
    
    result = rewriter.rewrite_rhetorical_query(current_query, conversation_history)
    print(f"改写结果: {result}\n")
    
    # 示例6: 自动识别Query类型
    # 批量测试自动路由的准确率。5 条 query 分别对应 5 种类型：
    #   还有其他游乐项目吗？   → 上下文依赖型
    #   哪个园区更好玩？       → 对比型
    #   都适合小朋友吗？       → 模糊指代型
    #   有什么餐厅？价格怎么样？→ 多意图型
    #   这不会也要排队两小时吧？→ 反问型
    # 注意这里调用的是 auto_rewrite_query（只做一次调用），
    # 而不是 auto_rewrite_and_execute（两次调用），批量测试时能省一半开销
    print("示例6: 自动识别Query类型")
    test_queries = [
        "还有其他游乐项目吗？",
        "哪个园区更好玩？",
        "都适合小朋友吗？",
        "有什么餐厅？价格怎么样？",
        "这不会也要排队两小时吧？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"测试查询 {i}: {query}")
        result = rewriter.auto_rewrite_query(query)
        print(f"  识别类型: {result['query_type']}")
        print(f"  改写结果: {result['rewritten_query']}")
        print(f"  置信度: {result['confidence']}\n")

if __name__ == "__main__":
    main() 

# ---------------------------------------------------------------------------
# 【RAG 实践要点】
# 1. Query 改写是「性价比最高」的一环：一次轻量 LLM 调用（qwen-turbo 成本极低），
#    却能显著提升召回率。向量库效果不好时，先试改写，再考虑换模型/换切分策略
# 2. 多轮对话场景必须传 conversation_history。单轮 RAG 不需要，
#    但只要业务里有追问，就一定有「上下文依赖」和「指代」问题
# 3. 改写的风险是「改坏」：模型可能把原本清晰的问题改成偏离原意的查询。
#    两个防护措施：
#    ① temperature=0 保证稳定；
#    ② 用返回的 confidence 做兜底，低置信度时回退到原问题检索
# 4. 每次改写都会增加一次 LLM 调用的延迟（约几百毫秒）。
#    对延迟敏感的场景可以：只对命中关键词的 query 做改写，
#    或用更小的模型（qwen-turbo）来做分类、用大模型做改写
# 5. 这个脚本只覆盖了「改写」。RAG 查询优化还有另外两个方向：
#    - 查询扩展（Query Expansion）：一个问题生成多个变体，多路召回后融合（RAG-Fusion）
#    - 假设文档嵌入（HyDE）：先让 LLM 编一个「假答案」，用假答案去检索，
#      因为答案和文档的表述风格更接近，检索效果常优于用问题本身检索
# 6. 与 2-Query联网搜索改写.py 的区别：
#    本脚本解决「怎么问得更清楚」，那个脚本解决「要不要联网 / 怎么搜网」，
#    两者都是检索前的前置处理，可以串成一条链
# ---------------------------------------------------------------------------
