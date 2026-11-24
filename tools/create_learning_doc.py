"""Create a beginner-friendly CDT503 RAG learning and acceptance guide.

This script intentionally creates the document from plain text and tables so it
can be regenerated whenever the project structure changes.
"""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "CDT503_RAG_学习与验收指南.docx"


def set_cell_shading(cell, fill: str) -> None:
    """Give a table cell a light background color."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    """Replace a cell's content and apply the document font."""
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    run.bold = bold
    set_run_font(run, 9.5)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_run_font(run, size: float = 10.5, color: str | None = None) -> None:
    """Set East Asian and Latin fonts so Word and LibreOffice agree."""
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Aptos")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Aptos")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_paragraph(doc, text: str = "", style: str | None = None, size: float = 10.5,
                  color: str | None = None, bold: bool = False, space_after: float = 5):
    paragraph = doc.add_paragraph(style=style)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    run.bold = bold
    set_run_font(run, size, color)
    return paragraph


def add_bullet(doc, text: str, level: int = 0):
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    set_run_font(run, 10)
    return paragraph


def add_code(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.25)
    paragraph.paragraph_format.right_indent = Inches(0.1)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.0
    for index, line in enumerate(text.splitlines()):
        run = paragraph.add_run(line)
        set_run_font(run, 8.5, "333333")
        if index < len(text.splitlines()) - 1:
            run.add_break()
    return paragraph


def add_heading(doc, text: str, level: int = 1):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(8 if level == 1 else 5)
    paragraph.paragraph_format.space_after = Pt(4)
    for run in paragraph.runs:
        set_run_font(run, 15 if level == 1 else 12, "1F4E79")
        run.bold = True
    return paragraph


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    if widths:
        for cell, width in zip(table.rows[0].cells, widths):
            cell.width = Inches(width)
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, bold=True)
        set_cell_shading(cell, "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], str(value))
            if widths:
                cells[index].width = Inches(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def configure_document(doc: Document):
    section = doc.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    for name, size, color in [("Title", 23, "1F4E79"), ("Heading 1", 15, "1F4E79"),
                              ("Heading 2", 12, "2F6F8F")]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("CDT503 Track B | Local PDF RAG | Learning and acceptance guide")
    set_run_font(run, 8, "777777")


def build_document():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("CDT503 本地 PDF RAG 学习与验收指南")
    set_run_font(run, 23, "1F4E79")
    run.bold = True
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(14)
    run = subtitle.add_run("面向不了解大模型的同学：从 PDF 到可解释答案")
    set_run_font(run, 12, "5A6B7A")

    add_paragraph(doc, "先给结论", style="Heading 1")
    add_paragraph(doc, "这个项目已经满足 CDT503 Track B 的基本方向：它不是只训练一个简单分类器，而是把本地 PDF、文本检索、向量数据库、Transformer 大模型和 Gradio 演示界面连接成一个可运行的问答系统。它具备冲击 A 的工程深度，但任何课程成绩都不能只由代码保证。最终成绩还取决于报告是否解释清楚、实验是否使用匹配的数据、演示是否稳定，以及老师对结果质量的评价。", size=10.5)
    add_paragraph(doc, "本指南把项目拆成容易理解的步骤，并给出新的测试 PDF、验收问题和 A 档准备清单。文档中会区分两类内容：课程评分要求是老师的要求；本地模型、禁止 OpenAI API 等是本项目的实现约束。", size=10.5)

    add_heading(doc, "1 项目与课程要求的对应关系", 1)
    add_table(doc,
              ["课程或评分关注点", "项目中的实现", "当前判断"],
              [
                  ("Track B：NLP、Transformer、LLM 或 RAG", "本地 PDF RAG：先检索文本片段，再让 Qwen 生成答案。", "满足基本要求"),
                  ("不能只是下载数据并运行一个基础模型", "项目包含 PDF 解析、切分、向量化、持久化、检索、生成、Web UI 和评估。", "工程复杂度明显高于基线"),
                  ("实验和结果分析", "对比 chunk_size=512 与 1024，并计算 context_recall、answer_relevancy，保存 CSV 和柱状图。", "具备 A 档实验雏形"),
                  ("部署或可交互演示", "Gradio 界面支持上传 PDF 和输入问题。", "具备加分亮点"),
                  ("高级模型实现", "GPU 可使用 Qwen 7B 4bit；CUDA 不可用时自动切换本地 Qwen 1.5B CPU 模式。", "能运行，但必须在报告中诚实说明硬件差异"),
              ], widths=[1.7, 3.5, 1.5])
    add_paragraph(doc, "A 不是自动得到的标签。要让老师更容易给 A，需要把系统运行证据、两组实验结果、失败案例、限制和改进计划放进报告与演示中。", bold=True, color="A53A2A")

    add_heading(doc, "2 先把大模型想成一个会读书的问答助手", 1)
    add_paragraph(doc, "如果完全不了解大模型，可以把系统想成图书馆：")
    add_bullet(doc, "PDF 是图书馆里的书。系统先把书里的文字读出来。")
    add_bullet(doc, "Embedding 是把一句话变成一串数字坐标。意思相近的句子，坐标通常也更接近。")
    add_bullet(doc, "Chroma 是按数字坐标查找相似内容的书架索引。它不负责写答案，只负责找资料。")
    add_bullet(doc, "Retriever 是图书管理员：根据问题找出最相关的几个片段。top-k=4 表示默认取四段。")
    add_bullet(doc, "Qwen 是负责组织语言的本地模型。它会看到问题和检索片段，然后生成回答。")
    add_bullet(doc, "RAG 的关键是先找资料再回答，所以回答可以有文档依据，不必完全依赖模型记忆。")
    add_paragraph(doc, "一句话公式：用户问题 -> 向量检索相关片段 -> 把片段和问题放进 prompt -> 本地 Qwen 生成答案。", bold=True, color="1F4E79")
    add_code(doc, "PDF files\n    |  PyPDFLoader\n    v\nText pages -> RecursiveCharacterTextSplitter -> chunks\n                                      |\n                                      v\n                             BGE-small embedding\n                                      |\n                                      v\n                              Chroma vector store\n                                      |\nUser question -> same embedding -> top-k chunks\n                                      |\n                                      v\n                    prompt(context + question)\n                                      |\n                                      v\n                 local Qwen -> answer + source names")

    add_heading(doc, "3 项目使用的技术", 1)
    add_table(doc,
              ["技术", "它做什么", "为什么放在这里"],
              [
                  ("Python", "编写数据处理、模型推理、评估和界面。", "课程项目容易运行，也方便解释。"),
                  ("LangChain", "把 PDF 加载、文本切分、检索和提示词步骤组织起来。", "减少重复代码，保持 RAG 流程清晰。"),
                  ("PyPDFLoader", "逐页读取 PDF 文本，并保留文件名、页码等来源信息。", "问答时可以展示来源。"),
                  ("RecursiveCharacterTextSplitter", "把长文本切成有重叠的小片段。", "片段太长难检索，太短又可能失去上下文。"),
                  ("BGE-small", "把文本转换为 embedding 向量。", "本地运行、速度较快，适合课堂实验。"),
                  ("Chroma", "保存向量和原文，并按相似度查找。", "不调用云端数据库，适合本地 RAG。"),
                  ("Transformers", "加载并运行 Qwen 模型。", "使用 Hugging Face 的标准本地模型接口。"),
                  ("bitsandbytes", "GPU 情况下支持 4bit 量化。", "减少 Qwen 7B 的显存占用。"),
                  ("Gradio", "提供上传 PDF、输入问题、显示答案的网页。", "让老师可以直接演示，而不必看命令行。"),
                  ("RAGAS 与 matplotlib", "计算检索和答案指标，并绘制对比图。", "把主观演示变成可比较的实验。"),
              ], widths=[1.65, 3.55, 1.5])

    add_heading(doc, "4 项目目录结构", 1)
    add_code(doc, "E:\\rag_project\n|-- data\n|   |-- raw\\                 原始 PDF，例如课程资料\n|   `-- demo_pdfs\\           本指南新生成的验收 PDF\n|-- src\n|   |-- ingestion\\ingest.py  读取 PDF、切分、生成向量、写入 Chroma\n|   |-- retrieval\\retriever_llm.py\n|   |                          检索 top-k 并调用本地 Qwen 生成答案\n|   |-- evaluation\\evaluate.py\n|   |                          对比两个 chunk_size 并输出指标和图\n|   `-- ui\\app.py             Gradio 上传和问答界面\n|-- vector_store\\chroma\\     持久化的向量数据库\n|-- evaluation_results\\        CSV 和柱状图输出\n|-- test_qa_set.json             RAGAS 测试问题和标准答案\n|-- requirements.txt             Python 依赖\n`-- docs\\CDT503_RAG_学习与验收指南.docx  本学习文档")
    add_paragraph(doc, "最重要的理解是：vector_store 不是模型本身，而是资料索引；Qwen 也不是训练出来的新模型，而是项目运行时加载的本地预训练模型。", bold=True)

    add_heading(doc, "5 四个主要 Python 文件分别做什么", 1)
    add_heading(doc, "5.1 ingest.py：建立知识库", 2)
    add_paragraph(doc, "运行 ingest.py 时，系统递归寻找 data 文件夹中的 PDF。每个 PDF 先被拆成页面文本，再由 RecursiveCharacterTextSplitter 切成片段。每个片段通过 BGE-small 变成向量，最后保存到 Chroma。")
    add_bullet(doc, "chunk_size：每个片段的目标长度。512 通常更细，1024 通常保留更多上下文。")
    add_bullet(doc, "chunk_overlap：相邻片段重叠的长度，避免一句话正好被切断。")
    add_bullet(doc, "collection_name：不同切分实验要使用不同 collection，防止新实验覆盖旧实验。")
    add_heading(doc, "5.2 retriever_llm.py：检索并生成", 2)
    add_paragraph(doc, "它先从 Chroma 读取指定 collection，再用相同的 BGE-small 将用户问题向量化，找到 top-k 片段。之后把片段放进 prompt，交给本地 Qwen。GPU 模式加载 Qwen 7B 4bit；当前电脑 CUDA=false 时，会自动使用 Qwen 1.5B CPU 备用模式。")
    add_heading(doc, "5.3 evaluate.py：做对比实验", 2)
    add_paragraph(doc, "它读取测试问题、分别访问 512 和 1024 的向量库、生成答案，然后用 RAGAS 计算 context_recall 和 answer_relevancy，并把结果保存为 CSV 和图片。测试集必须和被检索的 PDF 内容匹配，否则分数没有解释意义。")
    add_heading(doc, "5.4 app.py：把项目变成网页", 2)
    add_paragraph(doc, "上传 PDF 后，界面会复制文件并调用 ingest 流程。输入问题后，界面调用检索和本地 Qwen，显示回答和来源。CPU 模式第一次加载可能较慢，演示前应先预热一次。")

    add_heading(doc, "6 如何运行和理解一次完整实验", 1)
    add_paragraph(doc, "在 VS Code 终端中，先选择项目虚拟环境作为 Python Interpreter。下面命令假设当前目录是 E:\\rag_project。Windows PowerShell 可以逐条执行：")
    add_code(doc, ".\\venv\\Scripts\\Activate.ps1\npython -m pip install -r requirements.txt\npython -m src.ingestion.ingest --chunk-size 512 --chunk-overlap 120\npython -m src.ingestion.ingest --chunk-size 1024 --chunk-overlap 120\npython -m src.ui.app")
    add_paragraph(doc, "如果执行 python -m src.ui.app 后出现本地网址，在浏览器打开它即可。CUDA=false 时不要误以为程序坏了：这表示没有可用 NVIDIA CUDA，程序会用 CPU 备用模型，速度会慢一些。")
    add_paragraph(doc, "评估前先检查 test_qa_set.json 的问题确实能在当前 data PDF 中找到答案，再执行：")
    add_code(doc, "python -m src.evaluation.evaluate")
    add_paragraph(doc, "本项目原来的 test_qa_set.json 是通用测试数据。如果它和当前 PDF 内容不一致，RAGAS 分数会偏低，不能据此判断系统一定很差。验收演示应使用本指南后面的新 PDF 和问题，或者另外制作与课程 PDF 一一对应的测试集。", bold=True, color="A53A2A")

    add_heading(doc, "7 A 档需要展示的证据", 1)
    add_table(doc,
              ["证据", "演示时要说清楚"],
              [
                  ("架构图", "为什么先检索再生成；Chroma 和 Qwen 的职责不同。"),
                  ("两组切分实验", "512 和 1024 的差异、指标表、柱状图，以及你对差异的解释。"),
                  ("来源信息", "答案引用了哪一个 PDF、哪一页或哪一个片段。"),
                  ("失败案例", "问题过于模糊、PDF 没有相关内容、英文资料配中文问题时会发生什么。"),
                  ("硬件说明", "当前 CUDA=false，所以本次运行是 CPU Qwen 1.5B；有 GPU 时才是 Qwen 7B 4bit。"),
                  ("工程可靠性", "重复运行不会覆盖错误 collection，文件路径可配置，依赖和结果文件齐全。"),
              ], widths=[1.7, 5.0])
    add_paragraph(doc, "一个很有价值的限制分析是：BGE-small 对跨语言问题不一定稳定。如果资料是英文而问题是中文，检索可能失误。你可以在报告中展示这个现象，并说明未来会测试多语言 embedding 或加入问题翻译步骤。诚实报告限制通常比假装所有问题都正确更有说服力。")

    add_heading(doc, "8 新测试 PDF 的验收演示", 1)
    add_paragraph(doc, "data/demo_pdfs 文件夹中有三份全新、互相独立的虚构资料。它们没有使用课程 PDF 的原文，适合先验证管线是否真正读到了新资料。建议先清楚地上传或重新 ingest 这三份文件，再使用下面的问题。")
    add_table(doc,
              ["PDF", "适合验证的内容"],
              [
                  ("demo_rag_handbook.pdf", "RAG 的步骤、chunk overlap、top-k 和离线模型。"),
                  ("demo_campus_robot.pdf", "虚构机器人 Rover-6 的容量、充电、传感器和安全规则。"),
                  ("demo_research_policy.pdf", "虚构实验室的评估规则、阈值和复现实验流程。"),
              ], widths=[2.3, 4.4])
    add_paragraph(doc, "推荐验收顺序：先问资料中明确出现的事实，再问跨文档对比，最后问一个资料中没有的问题，观察系统是否会说无法从资料确认。", bold=True)
    add_code(doc, "1. What are the three main stages of the demo RAG pipeline?\n2. What is the default top-k value in the demo handbook?\n3. What is the maximum payload of Rover-6?\n4. Which sensor must be checked before autonomous movement?\n5. What is the minimum answer relevancy target in the research policy?\n6. Compare the 512 and 1024 chunk experiments.\n7. Who won the 2025 campus robotics championship?\n   Expected behavior: the documents do not contain this fact, so the system should say it cannot confirm it.")

    add_heading(doc, "9 最终提交前检查清单", 1)
    for item in [
        "在一台没有 CUDA 的电脑上成功运行，并在报告里说明 CPU fallback。",
        "至少保留两组 chunk_size 实验结果，包含 CSV、柱状图和解释。",
        "测试问题和 PDF 内容匹配，避免用无关测试集制造没有意义的低分。",
        "演示一次上传 PDF、一次直接提问、一次显示来源、一次回答未知问题。",
        "报告中给出系统限制：跨语言检索、CPU 速度、模型大小和生成幻觉风险。",
        "提交前检查 requirements.txt、源代码、README 或本指南、测试 PDF、测试问题都在项目目录。",
    ]:
        add_bullet(doc, item)

    add_heading(doc, "10 你可以用来答辩的三句话", 1)
    add_paragraph(doc, "第一句：我们的系统采用 RAG，把文档检索和本地语言模型生成组合起来，减少模型脱离文档回答的风险。")
    add_paragraph(doc, "第二句：我们通过 chunk_size=512 和 1024 做了可重复的工程实验，用 context_recall 和 answer_relevancy 比较检索上下文与回答质量。")
    add_paragraph(doc, "第三句：当前设备 CUDA=false，因此演示使用本地 CPU 备用模型；GPU 环境会启用 Qwen 7B 4bit，这个差异和系统限制已经在报告中记录。")

    add_heading(doc, "结论", 1)
    add_paragraph(doc, "从课程方向看，这个项目已经超过最基本的 Track B 要求，并具备 A 档所需要的工程复杂度。它能否得到 A，关键不在于再堆更多库，而在于把问题定义、系统原理、对比实验、失败案例和运行证据讲清楚。只要使用匹配的测试资料，现场演示稳定，并诚实说明 CPU 与 GPU 的差异，这个项目会是一个有说服力的课程项目。")
    add_paragraph(doc, "参考范围：CDT503 final_project.pdf 中关于 Track B 与 grading differentiation 的课程说明；本项目源代码和本目录下生成的 demo_pdfs。", size=9, color="666666")

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
