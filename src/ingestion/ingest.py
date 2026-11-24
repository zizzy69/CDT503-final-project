"""
CDT503 Track-B RAG 项目：PDF 文档导入与 Chroma 向量库构建脚本。

功能流程：
    ./data/*.pdf
        -> LangChain PyPDFLoader
        -> RecursiveCharacterTextSplitter
        -> BGE-small 本地 embedding
        -> Chroma 本地持久化向量数据库

运行方式（在项目根目录 E:/rag_project 执行）：
    python src/ingestion/ingest.py

可通过环境变量做实验对比，例如：
    $env:CHUNK_SIZE = "500"
    $env:CHUNK_OVERLAP = "80"
    python src/ingestion/ingest.py

说明：
    - 本脚本不导入 openai，也不调用任何 OpenAI API。
    - BGE 模型第一次运行时可能需要从 Hugging Face 下载；下载完成后由
      sentence-transformers 在本地加载和推理。
    - 默认使用中文 BGE-small：BAAI/bge-small-zh-v1.5。也可以通过
      EMBEDDING_MODEL_NAME 指向本地模型目录。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 这里使用“项目根目录”作为相对路径基准，而不是使用当前 Python 文件所在目录。
# 这样可以保证用户从项目根目录执行脚本时，输入和输出路径与项目结构一致：
#
#     ./data/                   PDF 输入目录
#     ./vector_store/chroma/    Chroma 持久化目录
#
# 如果从其他目录执行本脚本，请先切换到项目根目录，或通过环境变量传入绝对路径。
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "vector_store" / "chroma"))
).resolve()


# ---------------------------------------------------------------------------
# 可配置的实验参数
# ---------------------------------------------------------------------------
# chunk_size 和 chunk_overlap 是后续检索效果对比实验的关键变量。
#
# 推荐实验思路：
#     组合 A：chunk_size=500,  chunk_overlap=80
#     组合 B：chunk_size=800,  chunk_overlap=120
#     组合 C：chunk_size=1200, chunk_overlap=200
#
# 通过环境变量修改参数，不需要改动代码：
#     CHUNK_SIZE=500 CHUNK_OVERLAP=80 python src/ingestion/ingest.py
#
# PowerShell 示例：
#     $env:CHUNK_SIZE = "500"
#     $env:CHUNK_OVERLAP = "80"
#     python src/ingestion/ingest.py
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


# BGE-small 中文模型。若项目 PDF 主要为英文，可将其改为 BAAI/bge-small-en-v1.5，
# 或使用 EMBEDDING_MODEL_NAME 环境变量指定本地模型目录。
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
)

# 本项目要求本地模型推理。默认优先查找 Hugging Face 本地缓存，避免网页上传
# 后因为网络超时而长时间停留在“等待上传”。如果确实需要联网下载，可显式设置：
#     $env:LOCAL_MODEL_ONLY = "0"
LOCAL_MODEL_ONLY = os.getenv("LOCAL_MODEL_ONLY", "1").lower() not in {
    "0", "false", "no", "off"
}

# embedding 设备默认自动选择 CUDA/CPU；也可以显式设置：
#     EMBEDDING_DEVICE=cpu
#     EMBEDDING_DEVICE=cuda
EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)

# Chroma collection 名称可通过环境变量覆盖。
# 默认名称中包含 chunk 参数，便于不同切分方案使用不同 collection，避免实验数据混淆。
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    f"pdf_rag_bge_small_cs{CHUNK_SIZE}_co{CHUNK_OVERLAP}",
)


def resolve_model_reference(model_name: str) -> tuple[str, bool]:
    """把模型名解析为本地 snapshot 路径，并返回是否只能离线读取。

    Hugging Face 的 ``from_pretrained`` 即使模型已经缓存，也可能先发起 HEAD
    请求检查远端文件。本机网络不可用时，这个请求会重试多次，网页看起来就像
    上传卡住。解析到 snapshot 后直接传本地绝对路径，并设置 local_files_only，
    可以跳过这一步。
    """

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
            f"本地找不到 embedding 模型 {model_name}。"
            "请先联网下载一次，或设置 EMBEDDING_MODEL_NAME 为本地模型目录。"
        )

    return model_name, False


def validate_settings() -> None:
    """校验运行参数，尽早发现配置错误，避免生成不可用的向量库。"""

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE 必须是大于 0 的整数。")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP 不能小于 0。")

    # overlap 等于或大于 chunk_size 会造成切分无法正常向前推进。
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE。")

    if not COLLECTION_NAME.strip():
        raise ValueError("CHROMA_COLLECTION_NAME 不能为空。")


def find_pdf_files(data_dir: Path) -> List[Path]:
    """递归查找 data_dir 下的全部 PDF 文件。

    使用 rglob 而不是只读取 data_dir 的第一层，允许用户按主题或来源建立子目录，
    例如：data/course_notes/week01.pdf、data/manuals/product.pdf。

    Args:
        data_dir: PDF 根目录。

    Returns:
        按路径排序后的 PDF 文件列表。

    Raises:
        FileNotFoundError: 输入目录不存在时抛出。
    """

    if not data_dir.exists():
        raise FileNotFoundError(f"找不到 PDF 输入目录：{data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"PDF 输入路径不是目录：{data_dir}")

    # suffix 统一转小写，从而同时支持 .pdf、.PDF 等写法。
    pdf_files = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )

    return pdf_files


def load_pdf_documents(pdf_files: List[Path], data_dir: Path) -> List[Document]:
    """使用 LangChain PyPDFLoader 加载 PDF，并补充可追踪的来源元数据。

    PyPDFLoader 默认会把 PDF 的每一页转换成一个 Document。保留页级文档有两个好处：
    1. 后续检索结果可以返回页码，方便问答时展示来源；
    2. 处理长 PDF 时不会先把整本书拼成一个超长字符串。

    Args:
        pdf_files: 待加载的 PDF 文件列表。
        data_dir: 用于计算相对路径的 PDF 根目录。

    Returns:
        PDF 页面对应的 LangChain Document 列表。
    """

    documents: List[Document] = []

    for pdf_path in pdf_files:
        # PyPDFLoader 是 LangChain 的本地 PDF loader，只读取本地文件，不调用远程 LLM。
        loader = PyPDFLoader(str(pdf_path))
        page_documents = loader.load()

        # source 使用相对于 ./data 的路径，避免把用户机器的绝对路径暴露给检索结果。
        relative_source = pdf_path.relative_to(data_dir).as_posix()

        for document in page_documents:
            document.metadata["source"] = relative_source
            document.metadata["file_name"] = pdf_path.name
            # PyPDFLoader 通常提供 page 字段；若缺失则保留 None，避免伪造页码。
            document.metadata["page_number"] = document.metadata.get("page")

        documents.extend(page_documents)
        print(f"[加载完成] {relative_source} -> {len(page_documents)} 页")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """使用 RecursiveCharacterTextSplitter 对页面文本进行递归切分。

    RecursiveCharacterTextSplitter 会优先尝试段落、换行、空格等边界，尽量避免把
    语义完整的内容从中间截断。chunk_overlap 用于让相邻片段共享少量上下文，降低
    关键信息恰好落在两个 chunk 边界时的检索损失。

    注意：这里不设置 length_function=len 以外的 tokenizer 计数方式，因此参数单位是
    Python 字符数，不是 token 数。后续如果需要严格按模型 token 数对比，可再替换为
    tokenizer-aware 的 length_function。
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # 将 chunk 在原文中的顺序保存到 metadata，便于之后调试检索结果和制作报告图表。
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

    return chunks


def build_embeddings() -> HuggingFaceEmbeddings:
    """构造 BGE-small 本地 embedding 模型。

    这里使用 LangChain 的 HuggingFaceEmbeddings 适配器，而不是 OpenAIEmbeddings。
    模型可以是 Hugging Face 模型名，也可以是已经下载到本地的模型目录。
    normalize_embeddings=True 便于使用余弦相似度进行向量检索。
    """

    model_reference, local_only = resolve_model_reference(EMBEDDING_MODEL_NAME)
    print(
        f"[加载 embedding] model={model_reference}, "
        f"device={EMBEDDING_DEVICE}, local_only={local_only}"
    )

    model_kwargs = {
        "device": EMBEDDING_DEVICE,
        "local_files_only": local_only,
    }

    return HuggingFaceEmbeddings(
        model_name=model_reference,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def make_document_ids(chunks: List[Document]) -> List[str]:
    """为每个 chunk 生成稳定 ID，避免重复执行时完全依赖随机 ID。

    ID 由来源、页码、chunk 内容和切分参数共同计算：
    - 相同 PDF、相同切分参数、相同内容会得到相同 ID；
    - 修改 chunk_size/chunk_overlap 后，默认会进入不同 collection；
    - 使用 SHA-1 只用于生成去重标识，不用于安全认证。
    """

    ids: List[str] = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = str(chunk.metadata.get("page_number", "unknown"))
        content = chunk.page_content
        raw_id = (
            f"{source}|{page}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{content}"
        ).encode("utf-8")
        digest = hashlib.sha1(raw_id).hexdigest()
        ids.append(f"pdf_chunk_{digest}")

    return ids


def save_to_chroma(chunks: List[Document]) -> Chroma:
    """将切分后的文档写入本地 Chroma 向量数据库。

    Chroma 在指定 persist_directory 后会自动持久化数据，不需要调用旧版本中的
    vector_store.persist()。这样可以避免因 Chroma 版本差异导致的 AttributeError。

    Args:
        chunks: 已完成文本切分的 Document 列表。

    Returns:
        已写入数据的 Chroma 实例，便于 main() 输出统计信息或后续复用。
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings()
    document_ids = make_document_ids(chunks)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 使用显式 ID 写入，便于重复运行脚本时由 Chroma 根据 ID 进行更新/去重。
    vector_store.add_documents(documents=chunks, ids=document_ids)

    return vector_store


def main() -> None:
    """执行完整的 PDF -> chunks -> embeddings -> Chroma 导入流程。"""

    validate_settings()
    pdf_files = find_pdf_files(DATA_DIR)

    if not pdf_files:
        raise FileNotFoundError(
            f"目录 {DATA_DIR} 下没有找到 PDF 文件，请先放入至少一个 .pdf 文件。"
        )

    print(f"[配置] data_dir={DATA_DIR}")
    print(f"[配置] chroma_dir={CHROMA_DIR}")
    print(f"[配置] collection={COLLECTION_NAME}")
    print(f"[配置] chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")

    documents = load_pdf_documents(pdf_files, DATA_DIR)
    if not documents:
        raise ValueError("PDF 已找到，但没有提取到任何页面文本。")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("文本切分结果为空，无法建立向量数据库。")

    save_to_chroma(chunks)

    print("\n[完成] PDF 文档已写入 Chroma。")
    print(f"[统计] PDF 文件数：{len(pdf_files)}")
    print(f"[统计] 页面 Document 数：{len(documents)}")
    print(f"[统计] 文本 chunk 数：{len(chunks)}")
    print(f"[输出] Chroma 路径：{CHROMA_DIR}")


if __name__ == "__main__":
    main()
