# CDT503 Track-B 本地 PDF RAG 问答系统

> 当前版本：已完成可运行的本地 RAG 原型，并补充了学习指南、合成测试 PDF 和验收问题。

## 1. 需求来源区分

### 来自课程 PDF：`final_project.pdf`

- 项目属于 Track B：Natural Language Processing & Sequence Modeling。
- 允许的方向包括 conversational agents 和 Retrieval-Augmented Generation (RAG)。
- 课程评分强调功能原型、方法/架构说明、实现结果和最终演示。
- 书面报告需要覆盖：问题与设置、方法与架构、实现与结果、挑战与结论。
- 最终演示需要展示可工作的原型，并能解释方法、回答技术问题。

### 来自用户本次请求

- 使用 Python。
- 目标是本地 PDF RAG 问答系统。
- 必须使用 LangChain、Chroma、BGE-small embedding。
- CUDA 环境使用 Qwen-7B-Instruct 的 4-bit 量化本地推理；无 CUDA 时使用
  Qwen2.5-1.5B-Instruct CPU fallback，保证项目可以在普通电脑上开发和演示。
- 禁止调用 OpenAI API，因此不加入 `openai` 或 `langchain-openai` 依赖。
- 当前实现已完成 `ingest.py`、`retriever_llm.py`、`evaluate.py` 和 `app.py`；这些文件均不调用 OpenAI API。

## 2. 目录结构

```text
E:/rag_project/
├── requirements.txt              # 第一阶段依赖清单，含分组中文注释
├── PROJECT_STRUCTURE.md          # 需求来源、目录职责和后续实现边界
├── src/                          # 应用源码
│   ├── config/                   # 路径、模型名、检索参数等配置
│   ├── ingestion/                # PDF 加载、清洗、分块、元数据整理
│   ├── retrieval/                # BGE embedding、Chroma 建库和检索器
│   ├── generation/               # 本地 Qwen 加载和答案生成
│   ├── evaluation/               # 检索质量、答案质量和演示指标
│   └── ui/                       # 本地命令行/Gradio 交互入口
├── data/
│   ├── raw/                      # 原始 PDF；不在代码中硬编码绝对路径
│   ├── demo_pdfs/                # 全新合成验收 PDF 和 demo_qa_set.json
│   ├── processed/                # 清洗后或分块后的中间数据
│   └── sample_questions/         # 手工测试问题和预期依据
├── vector_store/                 # Chroma 持久化目录；运行时生成数据库文件
├── models/
│   ├── embeddings/               # BGE-small 本地缓存或软链接位置
│   └── llm/                      # Qwen-7B-Instruct 本地缓存或软链接位置
├── scripts/                      # 导入文档、建库、启动服务等命令脚本
├── tests/                        # 单元测试和端到端测试
├── docs/                         # 架构图、实验记录、报告素材和学习指南
└── logs/                         # 本地运行日志；不得写入访问令牌或隐私内容
```

## 3. 预定的数据流

```text
本地 PDF
  -> PDF loader
  -> 文本清洗与 chunking
  -> BGE-small embedding
  -> Chroma 本地持久化
  -> 相似度检索 / top-k
  -> 本地 Qwen（GPU: 7B 4-bit / CPU: 1.5B）
  -> 带来源片段的本地答案
```

## 4. 后续实现边界

核心代码的数据流和后续实验建议如下：

1. 使用 `src.ingestion.ingest` 为 512 和 1024 两个 chunk_size 建立独立 collection。
2. 使用 `src.retrieval.retriever_llm` 测试 top-k 检索和本地 Qwen 回答。
3. 使用 `src.evaluation.evaluate` 生成 RAGAS 指标 CSV 和对比图。
4. 使用 `src.ui.app` 启动 Gradio，完成上传 PDF 和问答演示。
5. 阅读 `docs/CDT503_RAG_学习与验收指南.docx`，用 `data/demo_pdfs/` 进行最终验收。

每个模块都应保留详细中文注释，特别说明：输入、输出、模型路径、设备选择、
异常处理和为什么不使用任何 OpenAI API。
