# LangGraph ReAct Agent 学习项目

这是一个用于学习 LangGraph ReAct Agent 的简单示例项目。

核心代码结构参考自 LangChain AI 官方的 ReAct Agent 模板：

https://github.com/langchain-ai/react-agent

本项目仅用于个人学习、代码阅读和实验。请尊重原作者的工作，如果复用或修改相关代码，请保留必要的来源说明，并遵守原项目的开源协议。

## 项目简介

本项目实现了一个最小化的 ReAct Agent。

基本执行流程如下：

```text
START
  ↓
call_model
  ↓
如果模型需要调用工具 → tools → call_model
  ↓
如果模型不再调用工具 → END
```

Agent 主要包含：

- `Context`：运行时配置，例如模型、系统提示词、搜索结果数量
- `State`：Agent 状态，例如消息历史
- `search`：搜索工具
- `call_model`：调用大模型的节点
- `ToolNode`：执行工具调用
- `StateGraph`：构建 Agent 执行图

## 环境依赖

建议使用 Python 3.12。

安装依赖：

```bash
pip install -U langchain langgraph langchain-openai langchain-tavily python-dotenv
```

## 环境变量

建议使用 `.env` 文件保存 API Key：

```env
API_KEY=你的_API_Key
BASE_URL=
TAVILY_API_KEY=你的_Tavily_API_Key

MODEL=
MAX_SEARCH_RESULTS=5
```

请不要把真实 API Key 提交到 GitHub。

可以在 `.gitignore` 中加入：

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

## 运行方式

```bash
python react_agent.py
```

示例输入：

```python
HumanMessage(content="请搜索一下 LangGraph 是什么，并简要总结。")
```

正常情况下，执行过程大致是：

```text
HumanMessage
AIMessage(tool_calls=[search])
ToolMessage
AIMessage
```

也就是：

```text
用户问题 → 模型决定调用搜索工具 → 工具返回结果 → 模型生成最终回答
```

## 说明

本项目主要用于理解 LangGraph 的基本机制，包括：

- StateGraph 的构建流程
- 节点和边的定义
- 条件路由
- ReAct 循环
- 工具调用
- Runtime Context
- Messages 状态累积

后续可以继续扩展：

- 增加更多工具
- 优化搜索结果压缩
- 增加日志和调试输出
- 使用 checkpoint 保存状态
- 增加 human-in-the-loop
- 拆分为多个模块文件

## 致谢

感谢 LangChain AI 团队提供的开源项目和示例代码：

- https://github.com/langchain-ai/react-agent
- https://github.com/langchain-ai/langgraph
