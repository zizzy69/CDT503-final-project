"""
CDT503 Track-B RAG 项目：Gradio 本地 PDF 问答演示界面。

功能：
    1. 上传一个或多个 PDF 文件，并自动调用 ingest 流程建立 Chroma 向量库；
    2. 输入问题，通过 BGE-small + Chroma 检索相关片段；
    3. CUDA 环境使用 Qwen-7B-Instruct 4-bit，CPU 环境使用 Qwen2.5-1.5B-Instruct；
    4. 在网页中展示回答和 PDF 来源页码。

运行方式（在项目根目录执行）：
    python src/ui/app.py

可选环境变量：
    PROJECT_ROOT=E:/rag_project
    GRADIO_SERVER_NAME=127.0.0.1
    GRADIO_SERVER_PORT=7860

本界面不导入 openai，也不调用任何 OpenAI API。Qwen 和 BGE 都在本地加载和推理。
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

# 允许用户直接执行 python src/ui/app.py 时导入项目内的 src 包。
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

import gradio as gr

from src.ingestion import ingest as ingest_module
from src.retrieval import retriever_llm


# ---------------------------------------------------------------------------
# 全局运行状态
# ---------------------------------------------------------------------------
@dataclass
class RAGRuntime:
    """保存网页服务进程中可复用的本地 RAG 组件。

    模型加载非常耗时且会占用较多显存，因此网页启动时不立即加载 Qwen。
    用户第一次提交问题时才调用 get_runtime() 完成惰性初始化，后续问题复用
    同一个 tokenizer、模型和 embedding，适合课堂演示。
    """

    embeddings: Any
    vector_store: Any
    tokenizer: Any
    model: Any


_runtime: RAGRuntime | None = None
_runtime_lock = threading.Lock()
_ingest_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 上传与本地建库
# ---------------------------------------------------------------------------
def _get_uploaded_path(file_item: Any) -> Path:
    """兼容不同 Gradio 版本返回的文件对象格式。"""

    if isinstance(file_item, (str, Path)):
        return Path(file_item)

    # Gradio 新版本可能返回 FileData 对象，通常通过 path 属性提供本地临时文件路径。
    file_path = getattr(file_item, "path", None)
    if file_path:
        return Path(file_path)

    # 某些版本返回包含 path 的字典。
    if isinstance(file_item, dict) and file_item.get("path"):
        return Path(file_item["path"])

    raise TypeError(f"无法识别 Gradio 上传文件类型：{type(file_item)!r}")


def _copy_uploaded_pdfs(file_items: Sequence[Any]) -> List[Path]:
    """把 Gradio 临时文件复制到项目的 data/raw/ 目录。

    使用 Path.name 丢弃用户上传路径中的目录部分，避免路径穿越；目标文件名只
    来自 PDF 的文件名。若同名文件已经存在，则直接覆盖，之后依赖 ingest.py
    生成的稳定 chunk ID 避免向量库中产生重复记录。
    """

    raw_dir = ingest_module.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[Path] = []

    for file_item in file_items:
        source_path = _get_uploaded_path(file_item)

        if source_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"只支持 PDF 文件，收到：{source_path.name or source_path}"
            )

        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"找不到上传文件：{source_path}")

        # 只保留文件名，避免把临时目录或用户本机路径写入项目目录。
        target_path = raw_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path)

    return copied_files


def ingest_uploaded_files(file_items: Any) -> str:
    """处理上传文件并自动运行完整 ingest 流程。

    这里直接复用 ingest.py 的函数，而不是在 app.py 中复制一套 PDF 解析逻辑，
    确保命令行建库和网页建库使用同样的 PyPDFLoader、文本切分器、BGE embedding
    和 Chroma collection 配置。
    """

    if not file_items:
        return "请选择至少一个 PDF 文件。"

    # Gradio 单文件模式可能返回单个对象；统一转换为列表，便于后续处理。
    if not isinstance(file_items, (list, tuple)):
        file_items = [file_items]

    try:
        with _ingest_lock:
            copied_files = _copy_uploaded_pdfs(file_items)

            # 先校验 ingest 参数，避免在解析多个 PDF 后才发现配置错误。
            ingest_module.validate_settings()

            all_pdf_files = ingest_module.find_pdf_files(
                ingest_module.DATA_DIR
            )

            if not all_pdf_files:
                return "上传成功，但 data 目录下没有可处理的 PDF 文件。"

            # ingest.py 会处理 data/ 下的全部 PDF，保证同一向量库包含此前上传的资料。
            documents = ingest_module.load_pdf_documents(
                all_pdf_files,
                ingest_module.DATA_DIR,
            )
            chunks = ingest_module.split_documents(documents)

            if not chunks:
                return "PDF 已上传，但没有提取到可建立向量的文本。"

            ingest_module.save_to_chroma(chunks)

            # 如果问答模型已经初始化，重新打开 collection 以确保后续查询读取最新数据。
            global _runtime
            if _runtime is not None:
                _runtime.vector_store = retriever_llm.build_vector_store()

        uploaded_names = ", ".join(path.name for path in copied_files)

        return (
            "### 导入完成\n"
            f"已上传：`{uploaded_names}`\n\n"
            f"已处理 PDF：{len(all_pdf_files)} 个，"
            f"文本片段：{len(chunks)} 个。\n\n"
            f"Chroma 已保存至：`{ingest_module.CHROMA_DIR}`"
        )

    except Exception as error:
        return f"### 导入失败\n`{error}`"


# ---------------------------------------------------------------------------
# 本地 RAG 模型初始化与问答
# ---------------------------------------------------------------------------
def get_runtime() -> RAGRuntime:
    """惰性加载并缓存 BGE、Chroma 和 Qwen 4-bit 运行组件。"""

    global _runtime

    if _runtime is not None:
        return _runtime

    # 多个浏览器请求同时到达时只允许一个请求加载 7B 模型。
    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        embeddings = retriever_llm.build_embeddings()
        vector_store = retriever_llm.build_vector_store()
        tokenizer, model = retriever_llm.load_quantized_qwen()

        _runtime = RAGRuntime(
            embeddings=embeddings,
            vector_store=vector_store,
            tokenizer=tokenizer,
            model=model,
        )

    return _runtime


def format_sources(documents: Sequence[Any]) -> str:
    """把检索到的 Document 元数据转换为网页可读的来源列表。"""

    if not documents:
        return "未检索到来源片段。"

    source_lines: List[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source = metadata.get("source") or metadata.get("file_name") or "未知文件"
        page = metadata.get("page_number", metadata.get("page"))

        # PyPDFLoader 的页码通常从 0 开始，展示给课堂观众时转换成从 1 开始。
        page_label = str(page + 1) if isinstance(page, int) else "未知"
        preview = " ".join(document.page_content.split())[:160]

        source_lines.append(
            f"{index}. **{source}**，第 {page_label} 页\n"
            f"   > {preview}..."
        )

    return "\n".join(source_lines)


def answer_question(question: str) -> Tuple[str, str, str]:
    """响应网页提问，返回回答、来源和状态文本。"""

    question = (question or "").strip()

    if not question:
        return "", "", "请输入问题后再提交。"

    try:
        runtime = get_runtime()

        answer, documents = retriever_llm.answer_question(
            question=question,
            vector_store=runtime.vector_store,
            tokenizer=runtime.tokenizer,
            model=runtime.model,
        )

        sources = format_sources(documents)
        return answer, sources, "回答完成。"

    except Exception as error:
        return "", "", f"处理失败：{error}"


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    """创建适合课堂演示的简洁 Gradio 页面。"""

    css = """
    .demo-container {
        max-width: 1100px;
        margin: 0 auto;
    }
    .status-box {
        min-height: 70px;
    }
    """

    with gr.Blocks(
        title="CDT503 Local PDF RAG",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        with gr.Column(elem_classes=["demo-container"]):
            gr.Markdown(
                """
                # CDT503 本地 PDF RAG 问答系统

                上传课程资料后提问。系统使用 **BGE-small + Chroma + 本地 Qwen**，
                CUDA 环境启用 7B 4-bit，CPU 环境使用 1.5B
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 上传 PDF 文档")

                    upload_files = gr.File(
                        label="选择 PDF 文件",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath",
                    )

                    upload_status = gr.Markdown(
                        "等待上传 PDF 文件。",
                        elem_classes=["status-box"],
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 2. 输入问题")

                    question = gr.Textbox(
                        label="问题",
                        placeholder="例如：这个项目的主要研究目标是什么？",
                        lines=4,
                    )

                    with gr.Row():
                        ask_button = gr.Button(
                            "提交问题",
                            variant="primary",
                        )
                        clear_button = gr.ClearButton(
                            components=[question],
                            value="清空问题",
                        )

            status = gr.Markdown("系统状态：等待操作。")

            gr.Markdown("### 回答")
            answer = gr.Markdown(
                "回答将在这里显示。"
            )

            gr.Markdown("### 检索来源")
            sources = gr.Markdown(
                "检索到的 PDF 片段和页码将在这里显示。"
            )

            # 选择文件后自动启动 ingest；无需额外点击“建立向量库”。
            upload_files.change(
                fn=ingest_uploaded_files,
                inputs=upload_files,
                outputs=upload_status,
            )

            # 点击提交问题后进行检索和本地 Qwen 推理。
            ask_button.click(
                fn=answer_question,
                inputs=question,
                outputs=[answer, sources, status],
            )

            question.submit(
                fn=answer_question,
                inputs=question,
                outputs=[answer, sources, status],
            )

            gr.Markdown(
                "---\n"
                "**本地组件：** LangChain · Chroma · BGE-small · Local Qwen"
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # share=False 确保课堂演示只在本机提供服务，不创建公网分享链接。
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=True,
        show_error=True,
    )
