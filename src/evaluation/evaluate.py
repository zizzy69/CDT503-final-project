"""
CDT503 Track-B RAG 项目：RAGAS 本地评估脚本。

评估流程：
    test_qa_set.json
        -> 分别打开 chunk_size=512 和 chunk_size=1024 的 Chroma collection
        -> 使用 BGE-small 检索上下文
        -> 使用本地 Qwen 生成回答（CUDA 时 7B 4-bit，CPU 时 1.5B）
        -> 使用 RAGAS 的 context_recall 和 answer_relevancy 评分
        -> 保存 CSV 和柱状对比图

运行方式（在项目根目录执行）：
    python src/evaluation/evaluate.py

可选环境变量：
    TEST_SET_PATH=./data/test_qa_set.json
    TOP_K=4
    CHUNK_OVERLAP=120
    LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
    EVAL_OUTPUT_DIR=./evaluation_results

测试 JSON 支持的常见字段：
    [
        {
            "question": "问题文本",
            "reference": "标准答案"
        }
    ]

其中 reference 也可以使用 ground_truth、expected_answer 或 answer 字段。
脚本不会调用 OpenAI API；RAGAS 的评估 LLM 和 embedding 均使用本地模型。
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# 在导入项目内其他模块前把项目根目录加入 sys.path。
# 这样即使用户使用“python src/evaluation/evaluate.py”直接启动，仍然可以导入
# src.retrieval.retriever_llm，而不要求额外安装本项目为 Python package。
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 明确指定项目根目录后，再导入 retriever_llm 中已经实现并测试过的本地 RAG 组件。
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

# RAGAS 0.4.x 会在导入阶段引用旧版 LangChain 的 Vertex AI 模块。当前项目
# 完全不使用 Vertex AI；如果本地 langchain-community 已移除该可选模块，提供
# 一个仅用于 isinstance 检查的空兼容类，避免 RAGAS 在启动阶段无关地失败。
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except ModuleNotFoundError as error:
    if "langchain_community.chat_models.vertexai" not in str(error):
        raise

    vertexai_shim = types.ModuleType(
        "langchain_community.chat_models.vertexai"
    )

    class ChatVertexAI:  # noqa: N801
        """RAGAS 导入兼容占位类；本项目不会实例化或调用它。"""

    vertexai_shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_shim

import matplotlib

# 评估脚本只保存图片，不需要弹出 GUI 窗口；Agg 后端适合服务器和无桌面环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, ContextRecall
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieval.retriever_llm import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    build_prompt,
    generate_answer,
    retrieve_documents,
)


# ---------------------------------------------------------------------------
# 评估文件与输出文件配置
# ---------------------------------------------------------------------------
def resolve_test_set_path() -> Path:
    """解析测试集路径。

    默认优先读取 ./data/test_qa_set.json；如果用户把测试集放在项目根目录，
    也会自动兼容。显式设置 TEST_SET_PATH 时优先使用环境变量路径。
    """

    configured_path = os.getenv("TEST_SET_PATH")
    if configured_path:
        return Path(configured_path).resolve()

    candidates = [
        PROJECT_ROOT / "data" / "test_qa_set.json",
        PROJECT_ROOT / "test_qa_set.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # 返回默认位置，让后续错误信息清楚指出推荐的放置位置。
    return candidates[0].resolve()


TEST_SET_PATH = resolve_test_set_path()
OUTPUT_DIR = Path(
    os.getenv("EVAL_OUTPUT_DIR", str(PROJECT_ROOT / "evaluation_results"))
).resolve()
SUMMARY_CSV_PATH = OUTPUT_DIR / "ragas_chunk_comparison.csv"
DETAIL_CSV_PATH = OUTPUT_DIR / "ragas_chunk_comparison_details.csv"
PLOT_PATH = OUTPUT_DIR / "ragas_chunk_comparison.png"

# 本次对比实验固定比较两组 chunk_size；chunk_overlap 保持一致，以便尽量只观察
# chunk_size 对检索和生成质量的影响。
EXPERIMENT_CHUNK_SIZES: Tuple[int, int] = (512, 1024)


def load_test_cases(path: Path) -> List[Dict[str, str]]:
    """读取并规范化 test_qa_set.json。

    支持以下 JSON 顶层格式：
    1. 直接使用列表：[{"question": ..., "reference": ...}]
    2. 使用 data 字段：{"data": [{...}]}
    3. 使用 test_qa_set 字段：{"test_qa_set": [{...}]}

    为了让 RAGAS 的 context_recall 有可比较的标准，reference 是必需字段。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"找不到测试集：{path}\n"
            "请创建 data/test_qa_set.json，或设置 TEST_SET_PATH 环境变量。"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, dict):
        raw_cases = payload.get("data") or payload.get("test_qa_set")

        if raw_cases is None:
            # 也兼容只有一条问答记录的 JSON 对象。
            raw_cases = [payload] if "question" in payload else None
    else:
        raw_cases = None

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(
            "测试集格式错误：顶层应为问答对象列表，或包含 data/test_qa_set 列表字段。"
        )

    normalized_cases: List[Dict[str, str]] = []

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"测试集第 {index} 条不是 JSON 对象。")

        question = (
            raw_case.get("question")
            or raw_case.get("query")
            or raw_case.get("user_input")
        )

        reference = (
            raw_case.get("reference")
            or raw_case.get("ground_truth")
            or raw_case.get("expected_answer")
            or raw_case.get("answer")
        )

        # 某些测试集把标准答案保存为字符串列表；这里合并成一个文本，供 RAGAS
        # 的 context_recall 从标准答案中拆分事实声明。
        if reference is None and raw_case.get("ground_truths"):
            ground_truths = raw_case["ground_truths"]
            if isinstance(ground_truths, list):
                reference = "\n".join(str(item) for item in ground_truths)
            else:
                reference = str(ground_truths)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"测试集第 {index} 条缺少有效 question/query 字段。")

        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"测试集第 {index} 条缺少有效 reference/ground_truth 字段；"
                "context_recall 需要标准答案作为参照。"
            )

        case_id = raw_case.get("id", index)
        normalized_cases.append(
            {
                "id": str(case_id),
                "question": question.strip(),
                "reference": reference.strip(),
            }
        )

    return normalized_cases


def build_embedding_model() -> HuggingFaceEmbeddings:
    """加载与 ingest.py、retriever_llm.py 相同的 BGE-small embedding。

    检索和 answer_relevancy 必须使用同一语义空间的 embedding。这里不使用任何
    OpenAI embedding，而是直接加载本地/缓存的 BGE-small 模型。
    """

    print(
        f"[加载 BGE-small] model={EMBEDDING_MODEL_NAME}, "
        f"device={EMBEDDING_DEVICE}"
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_experiment_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    chunk_size: int,
) -> Chroma:
    """打开指定 chunk_size 对应的 Chroma collection。

    ingest.py 默认将切分参数写入 collection 名称，例如：
        pdf_rag_bge_small_cs512_co120
        pdf_rag_bge_small_cs1024_co120

    因此两个实验可以在同一个 Chroma 持久化目录中并存，互不覆盖。
    """

    collection_name = f"pdf_rag_bge_small_cs{chunk_size}_co{CHUNK_OVERLAP}"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    document_count = vector_store._collection.count()
    if document_count == 0:
        raise ValueError(
            f"Chroma collection '{collection_name}' 为空或不存在。"
            f"请先用 CHUNK_SIZE={chunk_size} 和 CHUNK_OVERLAP={CHUNK_OVERLAP} "
            "运行 ingest.py。"
        )

    print(
        f"[实验 collection] chunk_size={chunk_size}, "
        f"name={collection_name}, documents={document_count}"
    )

    return vector_store


def _model_device(model: AutoModelForCausalLM) -> torch.device:
    """获取量化模型的主设备，用于把输入张量放到正确设备。"""

    return next(model.parameters()).device


class LocalTransformersRagasLLM(BaseRagasLLM):
    """把本地 Transformers 模型适配为 RAGAS 可使用的评估 LLM。

    RAGAS 的 context_recall 和 answer_relevancy 都需要一个“评估裁判模型”读取
    问题、上下文、答案或标准答案并给出结构化判断。这里实现 BaseRagasLLM 的
    generate_text/agenerate_text 接口，让 RAGAS 的评估提示词直接交给本地 Qwen。

    这样评估阶段也不需要 OpenAI client、API key 或远程服务。
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
    ) -> None:
        # BaseRagasLLM 是 dataclass 风格的基类；先初始化其运行配置，再保存本地模型。
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def _generate_one(
        self,
        prompt_text: str,
        temperature: float | None,
        stop: List[str] | None,
    ) -> str:
        """使用本地 Qwen 生成一条 RAGAS 评估结果。"""

        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个严格按照要求输出评估结果的本地 RAG 评估助手。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ]
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            model_inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
            )

        device = _model_device(self.model)
        input_tensors = {
            key: value.to(device)
            for key, value in model_inputs.items()
            if key in {"input_ids", "attention_mask"}
            and torch.is_tensor(value)
        }
        input_length = input_tensors["input_ids"].shape[-1]

        # RAGAS 需要生成短的判断/结构化结果；评估使用的最大输出长度不宜过大，
        # 否则会显著增加两组实验的运行时间。
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": min(MAX_NEW_TOKENS, 256),
            "repetition_penalty": REPETITION_PENALTY,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        effective_temperature = 0.0 if temperature is None else float(temperature)
        if effective_temperature <= 0.0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": effective_temperature,
                    "top_k": 50,
                    "top_p": TOP_P,
                }
            )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **input_tensors,
                **generation_kwargs,
            )

        output_ids = generated_ids[0][input_length:]
        text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        # 若 RAGAS 传入 stop 字符串，则截断到第一个 stop 位置，模拟常见 LLM 接口。
        if stop:
            stop_positions = [text.find(item) for item in stop if text.find(item) >= 0]
            if stop_positions:
                text = text[: min(stop_positions)].strip()

        return text

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.01,
        stop: List[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """实现 RAGAS 的同步文本生成接口。"""

        del callbacks  # 本地离线评估不需要 LangChain callback。

        if hasattr(prompt, "to_string"):
            prompt_text = prompt.to_string()
        else:
            prompt_text = str(prompt)

        # RAGAS 可能要求 n 个候选结果；每个结果独立调用本地模型，保持接口语义。
        generations = [
            Generation(
                text=self._generate_one(
                    prompt_text=prompt_text,
                    temperature=temperature,
                    stop=stop,
                )
            )
            for _ in range(max(1, n))
        ]

        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.01,
        stop: List[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """实现 RAGAS 的异步接口。

        Transformers 的本地 generate 是同步 GPU 操作；这里保持接口兼容，直接复用
        同步实现。脚本采用逐样本评估，避免同时把多次 7B 推理压入显存。
        """

        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )


def create_ragas_embeddings(
    embedding_model: HuggingFaceEmbeddings,
) -> LangchainEmbeddingsWrapper:
    """把 LangChain BGE embedding 适配成 RAGAS 的 embedding 接口。"""

    return LangchainEmbeddingsWrapper(embedding_model)


def safe_metric_score(metric: Any, **kwargs: Any) -> float:
    """兼容 RAGAS metric 的同步/异步评分返回值。

    当前 collections API 提供 score()；如果安装的 RAGAS 版本只暴露 ascore()，
    则在没有运行事件循环的命令行环境中使用 asyncio.run() 作为后备方案。
    """

    try:
        if hasattr(metric, "score"):
            result = metric.score(**kwargs)
        elif hasattr(metric, "ascore"):
            result = asyncio.run(metric.ascore(**kwargs))
        else:
            raise AttributeError("RAGAS metric 没有 score/ascore 方法。")
    except TypeError:
        # 某些版本只接受异步评分接口；重新尝试 ascore。
        if not hasattr(metric, "ascore"):
            raise
        result = asyncio.run(metric.ascore(**kwargs))

    # Collections API 返回带 value 属性的 MetricResult；旧版本可能直接返回 float。
    value = getattr(result, "value", result)
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return float("nan")

    return numeric_value


def evaluate_one_case(
    case: Dict[str, str],
    vector_store: Chroma,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    context_recall_metric: ContextRecall,
    answer_relevancy_metric: AnswerRelevancy,
) -> Dict[str, Any]:
    """运行一条测试问答的检索、生成和两项 RAGAS 评估。"""

    question = case["question"]
    reference = case["reference"]

    # 先执行真实 RAG 流程，确保评估指标对应系统实际返回的上下文和答案。
    retrieved_documents = retrieve_documents(vector_store, question)
    retrieved_contexts = [
        document.page_content.strip()
        for document in retrieved_documents
        if document.page_content.strip()
    ]

    prompt = build_prompt(question, retrieved_documents)
    answer = generate_answer(prompt, tokenizer, model)

    if not retrieved_contexts:
        raise ValueError(f"测试问题没有可用于 RAGAS 的上下文：{question}")

    # context_recall 使用 reference 判断标准答案中的关键信息是否出现在检索上下文中。
    context_recall = safe_metric_score(
        context_recall_metric,
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    # answer_relevancy 关注回答是否切合问题，不直接判断回答的事实正确性。
    answer_relevancy = safe_metric_score(
        answer_relevancy_metric,
        user_input=question,
        response=answer,
    )

    source_names = [
        str(document.metadata.get("source", "unknown"))
        for document in retrieved_documents
    ]

    return {
        "question_id": case["id"],
        "question": question,
        "reference": reference,
        "answer": answer,
        "context_recall": context_recall,
        "answer_relevancy": answer_relevancy,
        "retrieved_contexts": "\n\n---\n\n".join(retrieved_contexts),
        "sources": " | ".join(source_names),
    }


def write_detail_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """保存每条测试样本的详细评估结果，便于报告中分析失败案例。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "chunk_size",
        "chunk_overlap",
        "question_id",
        "question",
        "reference",
        "answer",
        "context_recall",
        "answer_relevancy",
        "sources",
        "retrieved_contexts",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_score(rows: Iterable[Dict[str, Any]], metric_name: str) -> float:
    """计算某一实验组的有效平均指标。"""

    values = []
    for row in rows:
        value = row.get(metric_name)
        if value is None:
            continue

        numeric_value = float(value)
        if math.isfinite(numeric_value):
            values.append(numeric_value)

    return sum(values) / len(values) if values else float("nan")


def write_summary_csv(
    summary_rows: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    """保存两组 chunk_size 的平均指标对比结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "chunk_size",
        "chunk_overlap",
        "num_questions",
        "context_recall",
        "answer_relevancy",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_comparison(
    summary_rows: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    """绘制两组 chunk_size 的 context_recall、answer_relevancy 柱状对比图。"""

    labels = [f"chunk_size={row['chunk_size']}" for row in summary_rows]
    context_recall_values = [
        float(row["context_recall"]) for row in summary_rows
    ]
    answer_relevancy_values = [
        float(row["answer_relevancy"]) for row in summary_rows
    ]

    x_positions = list(range(len(labels)))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        context_recall_values,
        width=bar_width,
        label="context_recall",
        color="#4C78A8",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        answer_relevancy_values,
        width=bar_width,
        label="answer_relevancy",
        color="#F58518",
    )

    axis.set_title("RAGAS Evaluation: Chunk Size Comparison")
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()

    # 在柱顶显示数值，便于直接插入课程报告或演示截图。
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """执行完整的两组 chunk_size RAGAS 对比实验。"""

    test_cases = load_test_cases(TEST_SET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[测试集] {TEST_SET_PATH}")
    print(f"[测试问题数] {len(test_cases)}")
    print(f"[实验组] chunk_size={EXPERIMENT_CHUNK_SIZES}")
    print(f"[固定 chunk_overlap] {CHUNK_OVERLAP}")
    print(f"[检索 TOP_K] {TOP_K}")

    # BGE 模型只加载一次，供两个 Chroma collection 和 RAGAS answer_relevancy 共用。
    embedding_model = build_embedding_model()
    ragas_embeddings = create_ragas_embeddings(embedding_model)

    # Qwen 7B 4-bit 只加载一次；随后复用于 RAG 答案生成和 RAGAS 评估裁判。
    print(
        f"[加载本地 Qwen] model={LLM_MODEL_NAME}, "
        f"mode={'4bit' if torch.cuda.is_available() else 'CPU'}"
    )
    from src.retrieval.retriever_llm import load_quantized_qwen

    tokenizer, model = load_quantized_qwen()
    ragas_llm = LocalTransformersRagasLLM(tokenizer=tokenizer, model=model)

    # RAGAS collections API：ContextRecall 需要标准答案；AnswerRelevancy 还需要
    # embedding 来计算问题与模型反向生成问题之间的语义相似度。
    context_recall_metric = ContextRecall(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for chunk_size in EXPERIMENT_CHUNK_SIZES:
        experiment_name = f"chunk_size_{chunk_size}"
        vector_store = build_experiment_vector_store(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
        )

        experiment_rows: List[Dict[str, Any]] = []

        for case_index, case in enumerate(test_cases, start=1):
            print(
                f"[评估] {experiment_name} "
                f"{case_index}/{len(test_cases)} "
                f"question_id={case['id']}"
            )

            result_row = evaluate_one_case(
                case=case,
                vector_store=vector_store,
                tokenizer=tokenizer,
                model=model,
                context_recall_metric=context_recall_metric,
                answer_relevancy_metric=answer_relevancy_metric,
            )
            result_row.update(
                {
                    "experiment": experiment_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": CHUNK_OVERLAP,
                }
            )
            experiment_rows.append(result_row)
            detail_rows.append(result_row)

        summary_rows.append(
            {
                "experiment": experiment_name,
                "chunk_size": chunk_size,
                "chunk_overlap": CHUNK_OVERLAP,
                "num_questions": len(experiment_rows),
                "context_recall": mean_score(
                    experiment_rows,
                    "context_recall",
                ),
                "answer_relevancy": mean_score(
                    experiment_rows,
                    "answer_relevancy",
                ),
            }
        )

    # 先写明细，再写汇总，方便复现实验并定位某道题的异常分数。
    write_detail_csv(detail_rows, DETAIL_CSV_PATH)
    write_summary_csv(summary_rows, SUMMARY_CSV_PATH)
    plot_comparison(summary_rows, PLOT_PATH)

    print("\n[完成] RAGAS 对比评估结束。")
    print(f"[汇总 CSV] {SUMMARY_CSV_PATH}")
    print(f"[明细 CSV] {DETAIL_CSV_PATH}")
    print(f"[柱状图] {PLOT_PATH}")

    for row in summary_rows:
        print(
            f"{row['experiment']}: "
            f"context_recall={row['context_recall']:.4f}, "
            f"answer_relevancy={row['answer_relevancy']:.4f}"
        )


if __name__ == "__main__":
    main()
