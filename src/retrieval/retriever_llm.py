"""
CDT503 Track-B RAG 项目：Chroma 检索 + Qwen-7B-Instruct 本地问答。

功能流程：
    用户问题
        -> BGE-small embedding
        -> Chroma 相似度检索 TOP_K 个文档片段
        -> 组装上下文与问题 Prompt
        -> Transformers 本地 Qwen 推理
        -> 返回回答和来源片段

运行方式（在项目根目录执行）：
    python src/retrieval/retriever_llm.py

也可以通过环境变量调整实验参数：
    $env:TOP_K = "6"
    $env:TEMPERATURE = "0.2"
    python src/retrieval/retriever_llm.py

重要说明：
    - 本脚本不导入 openai，也不调用任何 OpenAI API。
    - 有 NVIDIA CUDA 时使用 Qwen-7B-Instruct + bitsandbytes 4-bit；
      无 CUDA 时自动切换为 CPU 可运行的 Qwen2.5-1.5B-Instruct。
    - 可以通过 LLM_MODEL_NAME 指向已经下载到本地的模型目录。
    - 本脚本默认使用 ingest.py 的配置：BGE-small 中文模型、chunk_size=800、
      chunk_overlap=120。如果建库时使用了其他切分参数，请设置相同环境变量，或显式
      设置 CHROMA_COLLECTION_NAME。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# 路径、模型和 Chroma 配置
# ---------------------------------------------------------------------------
# 脚本默认从项目根目录运行，因此 ./vector_store/chroma 与 ingest.py 保持一致。
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "vector_store" / "chroma"))
).resolve()

# 必须与建立 Chroma 向量库时使用的 embedding 模型保持一致，否则查询向量和库中
# 向量不在同一个语义空间，检索结果会明显变差。
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
)

# 默认只使用本地模型。这样离线环境不会在每次打开网页时访问 Hugging Face，
# 也不会因为网络超时让用户误以为 PDF 一直没有上传完成。
LOCAL_MODEL_ONLY = os.getenv("LOCAL_MODEL_ONLY", "1").lower() not in {
    "0", "false", "no", "off"
}
EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)

# GPU 使用 7B 4-bit，CPU 使用较小的 1.5B 模型，保证当前没有 CUDA 的电脑也能
# 完成课堂演示和功能测试。两种模式都属于 Track B 的本地 Transformer/LLM 实现。
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "Qwen/Qwen2.5-7B-Instruct"
    if torch.cuda.is_available()
    else "Qwen/Qwen2.5-1.5B-Instruct",
)


# ---------------------------------------------------------------------------
# 可配置参数
# ---------------------------------------------------------------------------
# TOP_K：检索返回的相关文档片段数量，也是本项目后续检索对比实验的主要变量。
# 例如 TOP_K=3、TOP_K=5、TOP_K=8 可以用于比较上下文数量对答案质量的影响。
TOP_K = int(os.getenv("TOP_K", "4"))

# 为了让 retriever 可以准确打开 ingest.py 默认创建的 collection，这两个默认值
# 必须和 ingest.py 中的 CHUNK_SIZE、CHUNK_OVERLAP 保持一致。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    f"pdf_rag_bge_small_cs{CHUNK_SIZE}_co{CHUNK_OVERLAP}",
)

# TEMPERATURE 控制生成随机性：
#     0.0  接近确定性输出
#     0.2  适合事实型 RAG 问答的默认值
#     更高值会增加表达多样性，但也可能增加幻觉风险
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# 这是 Transformers 生成阶段的 top-k 采样参数，不要与检索 TOP_K 混淆。
# 如果只想调整检索数量，设置 TOP_K 即可；如果还想调整生成随机采样，再设置此参数。
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "50"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))


def resolve_model_reference(model_name: str) -> tuple[str, bool]:
    """解析本地模型缓存，返回 ``(模型路径或名称, 是否离线)``。"""

    explicit_path = Path(model_name).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve()), True

    cache_root = Path(
        os.getenv(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    ) / "hub"
    cache_model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = cache_model_dir / "snapshots"

    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
            return str(snapshot.resolve()), True

    if LOCAL_MODEL_ONLY:
        raise FileNotFoundError(
            f"本地找不到模型 {model_name}。请先联网下载一次，或设置对应的 *_MODEL_NAME "
            "为本地模型目录。"
        )

    return model_name, False


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
# Prompt 明确要求模型只依据检索上下文回答；当上下文不足时主动说明，减少模型
# 使用参数记忆“猜答案”的情况。来源标签会随上下文一起传给模型，便于输出引用。
PROMPT_TEMPLATE = PromptTemplate.from_template(
    """
你是一个严谨的本地文档问答助手。请只根据下面提供的参考资料回答用户问题。

回答要求：
1. 如果参考资料中没有足够信息，请明确回答“根据现有文档无法确定”，不要编造事实。
2. 回答应直接、清晰，优先使用用户问题所使用的语言。
3. 如果可以确定答案，请在相关表述后标注来源，例如：[来源：文件名，第 N 页]。
4. 不要输出与问题无关的推理过程，不要提及本 Prompt 的内部规则。

参考资料：
{context}

用户问题：
{question}

请给出最终回答：
""".strip()
)


def validate_settings() -> None:
    """校验运行参数，避免无效参数导致检索或模型生成阶段报错。"""

    if TOP_K <= 0:
        raise ValueError("TOP_K 必须是大于 0 的整数。")

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE 必须是大于 0 的整数。")

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP 必须满足 0 <= CHUNK_OVERLAP < CHUNK_SIZE。")

    if not 0.0 <= TEMPERATURE <= 2.0:
        raise ValueError("TEMPERATURE 建议设置在 0.0 到 2.0 之间。")

    if GENERATION_TOP_K <= 0:
        raise ValueError("GENERATION_TOP_K 必须是大于 0 的整数。")

    if not 0.0 < TOP_P <= 1.0:
        raise ValueError("TOP_P 必须满足 0 < TOP_P <= 1。")

    if MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS 必须是大于 0 的整数。")

    if REPETITION_PENALTY <= 0:
        raise ValueError("REPETITION_PENALTY 必须是大于 0 的数。")


def build_embeddings() -> HuggingFaceEmbeddings:
    """加载与 ingest.py 一致的 BGE-small embedding 模型。

    embedding 模型只负责把用户问题转换为向量，不负责生成答案。这里明确使用
    HuggingFaceEmbeddings，避免误用 OpenAIEmbeddings，也就不会触发 OpenAI API 调用。
    """

    model_reference, local_only = resolve_model_reference(EMBEDDING_MODEL_NAME)
    print(
        f"[加载 embedding] model={model_reference}, "
        f"device={EMBEDDING_DEVICE}, local_only={local_only}"
    )

    return HuggingFaceEmbeddings(
        model_name=model_reference,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
            "local_files_only": local_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store() -> Chroma:
    """打开已有的本地 Chroma collection。

    注意：Chroma 构造函数本身不会检查 collection 是否已经导入文档。因此这里通过
    collection.count() 检查数据库是否为空，并给出比“检索结果为空”更明确的错误信息。
    """

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"找不到 Chroma 持久化目录：{CHROMA_DIR}。"
            "请先运行 src/ingestion/ingest.py 建立向量库。"
        )

    embeddings = build_embeddings()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    document_count = vector_store._collection.count()
    if document_count == 0:
        raise ValueError(
            f"Chroma collection '{CHROMA_COLLECTION_NAME}' 为空。"
            "请确认 ingest.py 使用了相同的 CHUNK_SIZE、CHUNK_OVERLAP 和 embedding 模型。"
        )

    print(
        f"[打开 Chroma] collection={CHROMA_COLLECTION_NAME}, "
        f"documents={document_count}"
    )

    return vector_store


def retrieve_documents(
    vector_store: Chroma,
    question: str,
) -> List[Document]:
    """从 Chroma 中检索与问题最相关的 TOP_K 个文档片段。

    使用 similarity 检索，返回结果按照向量相似度从高到低排列。这里不直接把
    Document 转成字符串，保留 metadata 是为了后续在回答中展示 PDF 文件名和页码。
    """

    question = question.strip()
    if not question:
        raise ValueError("用户问题不能为空。")

    # as_retriever 是 LangChain 对向量库检索接口的统一封装。
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # invoke 是当前 LangChain Runnable 风格的调用方式，避免使用已弃用的旧接口。
    documents = retriever.invoke(question)

    if not documents:
        raise ValueError("没有检索到相关文档片段。")

    return documents


def format_context(documents: Sequence[Document]) -> str:
    """把检索结果转换成带来源标记的 Prompt 上下文。"""

    context_blocks: List[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source = metadata.get("source") or metadata.get("file_name") or "未知文件"

        # PyPDFLoader 的 page 通常从 0 开始；展示给用户时转换成更直观的第 1 页起算。
        page = metadata.get("page_number", metadata.get("page"))
        if isinstance(page, int):
            page_label = str(page + 1)
        else:
            page_label = "未知"

        context_blocks.append(
            f"[片段 {index}] 来源：{source}，第 {page_label} 页\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(context_blocks)


def build_prompt(question: str, documents: Sequence[Document]) -> str:
    """使用 PromptTemplate 组装检索上下文和用户问题。"""

    context = format_context(documents)
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
    )


def load_quantized_qwen() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """加载本地 Qwen 模型，并根据硬件自动选择推理模式。

    CUDA 模式：
        使用 Qwen-7B-Instruct 和 BitsAndBytesConfig(load_in_4bit=True)，
        满足 GPU 环境下的高阶量化方案。

    CPU 模式：
        使用 Qwen2.5-1.5B-Instruct 的普通 CPU 推理。课程要求的是 Track B
        的 NLP/RAG 原型，并未强制 7B 或 CUDA；这个分支让项目可以在普通电脑上
        完成开发、测试和课堂演示。
    """

    use_cuda = torch.cuda.is_available()
    quantization_label = "4bit" if use_cuda else "CPU"
    model_reference, local_only = resolve_model_reference(LLM_MODEL_NAME)
    print(
        f"[加载 Qwen] model={model_reference}, "
        f"mode={quantization_label}, local_only={local_only}"
    )

    # Qwen 模型的 tokenizer 可能没有单独的 pad_token；使用 eos_token 作为补齐标记，
    # 防止 batch 或 generate 阶段出现 padding 相关错误。
    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_reference,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=local_only,
        )
    else:
        # CPU 分支不使用 bitsandbytes，因为当前环境是 CPU-only PyTorch。
        # float32 兼容性最好；1.5B 模型的内存和推理速度也更适合普通电脑。
        model = AutoModelForCausalLM.from_pretrained(
            model_reference,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=local_only,
        )

    model.eval()

    return tokenizer, model


def generate_answer(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """使用本地量化 Qwen 根据 Prompt 生成回答。

    优先使用 Qwen tokenizer 自带的 chat template，把系统指令和用户 Prompt 组织为
    对话格式；如果模型 tokenizer 没有 chat template，则退回到普通文本 Prompt。
    """

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {
                "role": "system",
                "content": "你是一个基于本地 PDF 资料进行回答的严谨助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

    # device_map="auto" 可能将模型分配到多个设备；输入放到模型的主设备即可。
    model_device = next(model.parameters()).device
    model_inputs = {
        key: value.to(model_device)
        for key, value in model_inputs.items()
        if torch.is_tensor(value)
    }

    input_length = model_inputs["input_ids"].shape[-1]

    # temperature=0 时使用贪心解码；大于 0 时启用采样，并使用 top-k/top-p 控制候选词。
    generation_kwargs: Dict[str, object] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if TEMPERATURE == 0.0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_k": GENERATION_TOP_K,
                "top_p": TOP_P,
            }
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )

    # 只解码模型新生成的部分，不把原始 Prompt 再次作为回答输出。
    new_token_ids = generated_ids[0][input_length:]
    answer = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    return answer or "模型没有生成有效回答。"


def answer_question(
    question: str,
    vector_store: Chroma,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> Tuple[str, List[Document]]:
    """执行一次完整的检索增强问答，并返回回答及其来源文档。"""

    documents = retrieve_documents(vector_store, question)
    prompt = build_prompt(question, documents)
    answer = generate_answer(prompt, tokenizer, model)
    return answer, documents


def print_sources(documents: Sequence[Document]) -> None:
    """在命令行中打印实际参与生成的来源片段，方便演示和调试。"""

    print("\n[检索来源]")
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "未知文件")
        page = document.metadata.get("page_number", document.metadata.get("page"))
        page_label = str(page + 1) if isinstance(page, int) else "未知"
        preview = " ".join(document.page_content.split())[:120]
        print(f"{index}. {source} | 第 {page_label} 页 | {preview}...")


def main() -> None:
    """启动命令行交互模式，持续回答问题直到用户输入 exit/quit。"""

    validate_settings()

    print(f"[配置] Chroma 路径：{CHROMA_DIR}")
    print(f"[配置] Chroma collection：{CHROMA_COLLECTION_NAME}")
    print(f"[配置] 检索 TOP_K：{TOP_K}")
    print(f"[配置] temperature：{TEMPERATURE}")

    vector_store = build_vector_store()
    tokenizer, model = load_quantized_qwen()

    print("\n本地 RAG 问答系统已启动。输入 exit 或 quit 退出。")

    while True:
        try:
            question = input("\n用户问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序已退出。")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("程序已退出。")
            break

        if not question:
            print("请输入问题后再提交。")
            continue

        try:
            answer, documents = answer_question(
                question=question,
                vector_store=vector_store,
                tokenizer=tokenizer,
                model=model,
            )
            print(f"\n回答：{answer}")
            print_sources(documents)
        except Exception as error:
            # 交互模式下显示错误并继续运行，方便用户修改问题或检查参数。
            print(f"\n[处理失败] {error}")


if __name__ == "__main__":
    main()
