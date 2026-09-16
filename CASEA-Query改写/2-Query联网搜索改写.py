# ============================================================================
# Query 联网搜索改写：判断「要不要联网」+「怎么搜」
#
# 为什么需要它？
#   不是所有问题都该走 RAG 知识库。像"明天园区天气怎么样""现在门票多少钱"
#   这类问题，答案在向量库里根本不存在（因为它是随时间变化的），
#   硬去检索只会召回到一堆过时或无关的内容，LLM 再拿着这些内容编答案。
#   正确做法是：先判断该不该联网，该联网就把问题改写成「搜索引擎友好」的形式。
#
# 本脚本覆盖 4 个环节（每个环节对应一个方法）：
#   1. identify_web_search_needs  —— 判断是否需要联网搜索
#   2. rewrite_for_web_search     —— 把口语问题改写成检索式查询
#   3. generate_search_strategy   —— 生成搜索关键词与平台/时间范围策略
#   4. auto_web_search_rewrite    —— 串起前三步，供 RAG 流水线直接调用
#
# 在 RAG 链路中的位置：
#   用户提问 → 【是否需要联网？】→ 需要 → 联网搜索 → 结果拼进 Prompt → LLM 生成
#                              → 不需要 → 走本地向量库检索（见 1-Query改写.py）
#
# 接口说明：本脚本通过 agicto 聚合平台调用模型，使用 OpenAI 兼容协议。
#   Base URL: https://api.agicto.cn/v1/
#   需要环境变量 AGICTO_API_KEY（详见下方配置区）
# ============================================================================

# Query联网搜索改写功能
# 导入依赖库
from openai import OpenAI
import os
import json
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# 配置区：API Key 与模型名统一从环境变量读取，不硬编码在代码里（避免提交到 git 泄露）
#
# 运行前需先在服务器上配置：
#   export AGICTO_API_KEY="sk-xxxxxxxx"
# Key 去 https://agicto.com 控制台申请；模型名可在 https://agicto.com/model 查询
#
# 为什么用 OpenAI SDK？
#   agicto 是聚合平台、走 OpenAI 兼容协议，和 DashScope 原生协议不通用，
#   拿 agicto 的 Key 去调 dashscope SDK 会直接鉴权失败。
# ---------------------------------------------------------------------------
AGICTO_API_KEY = os.getenv('AGICTO_API_KEY')
AGICTO_BASE_URL = os.getenv('AGICTO_BASE_URL', 'https://api.agicto.cn/v1/')
CHAT_MODEL = os.getenv('AGICTO_CHAT_MODEL', 'gpt-4o-mini')

# fail-fast 校验：没配 Key 就直接抛错，而不是等到调用时才报奇怪的错
if not AGICTO_API_KEY:
    raise ValueError("请设置环境变量 AGICTO_API_KEY")

# OpenAI 兼容客户端：把 base_url 指向 agicto 即可，其余用法和调 OpenAI 完全一致
client = OpenAI(api_key=AGICTO_API_KEY, base_url=AGICTO_BASE_URL)

# 基于 prompt 生成文本
# 不传 model 时用配置区里的默认模型；temperature=0 保证判断结果稳定可复现
def get_completion(prompt, model=None):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=messages,
        temperature=0,
    )
    # openai SDK 失败时会直接抛异常，不像 DashScope SDK 那样静默塞进 status_code，
    # 所以这里只需要防一手「接口通了、choices 却为空」的边界情况
    if not response.choices:
        raise RuntimeError(f"agicto 未返回生成内容: {response}")
    return response.choices[0].message.content

class WebSearchQueryRewriter:
    # 不传 model 时使用配置文件里的默认模型 CHAT_MODEL
    def __init__(self, model=None):
        self.model = model or CHAT_MODEL
    
    def identify_web_search_needs(self, query, conversation_history=""):
        """识别查询是否需要联网搜索"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，判断是否需要联网搜索来获取最新、最准确的信息。

需要联网搜索的情况包括：
1. 时效性信息 - 包含"最新"、"今天"、"现在"、"实时"、"当前"等时间相关词汇
2. 价格信息 - 包含"多少钱"、"价格"、"费用"、"票价"等价格相关词汇
3. 营业信息 - 包含"营业时间"、"开放时间"、"闭园时间"、"是否开放"等营业状态
4. 活动信息 - 包含"活动"、"表演"、"演出"、"节日"、"庆典"等动态信息
5. 天气信息 - 包含"天气"、"下雨"、"温度"等天气相关
6. 交通信息 - 包含"怎么去"、"交通"、"地铁"、"公交"等交通方式
7. 预订信息 - 包含"预订"、"预约"、"购票"、"订票"等预订相关
8. 实时状态 - 包含"排队"、"拥挤"、"人流量"等实时状态

请返回JSON格式：
{
    "need_web_search": true/false,
    "search_reason": "需要搜索的原因",
    "confidence": "置信度(0-1)"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 用户查询 ###
{query}

### 分析结果 ###
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return {
                "need_web_search": False,
                "search_reason": "无法解析",
                "confidence": 0.5
            }
    
    def rewrite_for_web_search(self, query, search_type="general"):
        """为联网搜索改写查询"""
        instruction = """
你是一个专业的搜索查询优化专家。请将用户的查询改写为更适合搜索引擎检索的形式。

改写技巧：
1. 添加具体地点 - 如"上海迪士尼乐园"、"香港迪士尼乐园"
2. 添加时间范围 - 如"2024年"、"今天"、"本周"
3. 使用关键词组合 - 将长句拆分为关键词
4. 添加搜索意图 - 明确搜索目的
5. 去除口语化表达 - 转换为标准搜索词
6. 添加相关词汇 - 增加同义词或相关词

请返回JSON格式：
{
    "rewritten_query": "改写后的搜索查询",
    "search_keywords": ["关键词1", "关键词2", "关键词3"],
    "search_intent": "搜索意图",
    "suggested_sources": ["建议搜索的网站类型"]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始查询 ###
{query}

### 搜索类型 ###
{search_type}

### 改写结果 ###
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return {
                "rewritten_query": query,
                "search_keywords": [query],
                "search_intent": "信息查询",
                "suggested_sources": ["官方网站", "旅游网站"]
            }
    
    def generate_search_strategy(self, query, search_type="general"):
        """生成搜索策略"""
        current_date = datetime.now().strftime("%Y年%m月%d日")
        instruction = f"""
你是一个搜索策略专家。请为用户的查询制定详细的搜索策略。

当前日期：{current_date}

搜索策略包括：
1. 主要搜索词 - 核心关键词
2. 扩展搜索词 - 相关词汇和同义词
3. 搜索网站 - 推荐的搜索平台
4. 时间范围 - 具体的搜索时间范围

请返回JSON格式：
{{
    "primary_keywords": ["主要关键词"],
    "extended_keywords": ["扩展关键词"],
    "search_platforms": ["搜索平台"],
    "time_range": "具体的时间范围"
}}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 用户查询 ###
{query}

### 搜索类型 ###
{search_type}

### 搜索策略 ###
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return {
                "primary_keywords": [query],
                "extended_keywords": [],
                "search_platforms": ["百度", "谷歌"],
                "time_range": "最近一周"
            }
    
    def auto_web_search_rewrite(self, query, conversation_history=""):
        """自动识别并改写为联网搜索查询"""
        # 第一步：识别是否需要联网搜索
        search_analysis = self.identify_web_search_needs(query, conversation_history)
        
        if not search_analysis.get('need_web_search', False):
            return {
                "need_web_search": False,
                "reason": "查询不需要联网搜索",
                "original_query": query
            }
        
        # 第二步：改写查询
        rewritten_result = self.rewrite_for_web_search(query)
        
        # 第三步：生成搜索策略
        search_strategy = self.generate_search_strategy(query)
        
        return {
            "need_web_search": True,
            "search_reason": search_analysis.get('search_reason', ''),
            "confidence": search_analysis.get('confidence', 0.5),
            "original_query": query,
            "rewritten_query": rewritten_result.get('rewritten_query', query),
            "search_keywords": rewritten_result.get('search_keywords', []),
            "search_intent": rewritten_result.get('search_intent', ''),
            "suggested_sources": rewritten_result.get('suggested_sources', []),
            "search_strategy": search_strategy
        }

def main():
    # 初始化联网搜索Query改写器
    web_searcher = WebSearchQueryRewriter()
    
    print("=== Query联网搜索识别与改写示例（迪士尼主题乐园） ===\n")
    
    # 示例1: 时效性信息查询
    print("示例1: 时效性信息查询")
    conversation_history1 = """
用户: "我想去上海迪士尼乐园玩"
AI: "上海迪士尼乐园是一个很棒的选择！"
"""
    query1 = "上海迪士尼乐园今天开放吗？现在人多不多？"
    
    print(f"对话历史: {conversation_history1}")
    print(f"当前查询: {query1}")
    
    result1 = web_searcher.auto_web_search_rewrite(query1, conversation_history1)
    
    if result1['need_web_search']:
        print(f"✓ 需要联网搜索")
        print(f"  搜索原因: {result1['search_reason']}")
        print(f"  置信度: {result1['confidence']}")
        print(f"  改写查询: {result1['rewritten_query']}")
        print(f"  搜索关键词: {result1['search_keywords']}")
        print(f"  搜索意图: {result1['search_intent']}")
        print(f"  建议来源: {result1['suggested_sources']}")
        print(f"  搜索策略:")
        print(f"    - 主要关键词: {result1['search_strategy']['primary_keywords']}")
        print(f"    - 扩展关键词: {result1['search_strategy']['extended_keywords']}")
        print(f"    - 搜索平台: {result1['search_strategy']['search_platforms']}")
        print(f"    - 时间范围: {result1['search_strategy']['time_range']}")
    else:
        print(f"✗ 不需要联网搜索")
        print(f"  原因: {result1['reason']}")
    
    print("\n" + "="*60 + "\n")
    
    # 示例2: 价格和预订信息查询
    print("示例2: 价格和预订信息查询")
    query2 = "下周六的门票多少钱？需要提前多久预订？"
    
    print(f"当前查询: {query2}")
    
    result2 = web_searcher.auto_web_search_rewrite(query2)
    
    if result2['need_web_search']:
        print(f"✓ 需要联网搜索")
        print(f"  搜索原因: {result2['search_reason']}")
        print(f"  置信度: {result2['confidence']}")
        print(f"  改写查询: {result2['rewritten_query']}")
        print(f"  搜索关键词: {result2['search_keywords']}")
        print(f"  搜索意图: {result2['search_intent']}")
        print(f"  建议来源: {result2['suggested_sources']}")
        print(f"  搜索策略:")
        print(f"    - 主要关键词: {result2['search_strategy']['primary_keywords']}")
        print(f"    - 扩展关键词: {result2['search_strategy']['extended_keywords']}")
        print(f"    - 搜索平台: {result2['search_strategy']['search_platforms']}")
        print(f"    - 时间范围: {result2['search_strategy']['time_range']}")
    else:
        print(f"✗ 不需要联网搜索")
        print(f"  原因: {result2['reason']}")

if __name__ == "__main__":
    main() 