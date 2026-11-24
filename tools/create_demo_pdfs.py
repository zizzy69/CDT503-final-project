"""Create independent synthetic PDFs and a matching QA set for final demos.

The PDFs deliberately contain facts that are not present in the course PDF.
That makes them useful for checking whether ingestion and retrieval really
work on newly supplied documents.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo_pdfs"
QA_PATH = OUT_DIR / "demo_qa_set.json"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DemoTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#1F4E79"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "DemoSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#5B6770"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "DemoH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=colors.HexColor("#1F4E79"),
            spaceBefore=8, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "DemoBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.5, leading=15, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "DemoSmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=colors.HexColor("#666666"),
        ),
        "bullet": ParagraphStyle(
            "DemoBullet", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.5, leading=15, leftIndent=15, firstLineIndent=-9, spaceAfter=4,
        ),
    }


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 11 * mm, "CDT503 local PDF RAG demo data - synthetic document")
    canvas.drawRightString(192 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def p(text, style):
    return Paragraph(text, style)


def bullet(text, style):
    return Paragraph(f"- {text}", style)


def table(rows, widths):
    wrapped = []
    st = styles()
    for row_index, row in enumerate(rows):
        wrapped.append([p(str(cell), st["body"]) for cell in row])
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build_pdf(path: Path, title: str, subtitle: str, story):
    st = styles()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=19 * mm,
        title=title, author="CDT503 project",
    )
    full_story = [p(title, st["title"]), p(subtitle, st["subtitle"])] + story
    doc.build(full_story, onFirstPage=page_footer, onLaterPages=page_footer)


def handbook_story():
    st = styles()
    return [
        p("Purpose", st["h1"]),
        p("This fictional handbook describes a small local retrieval augmented generation system called Northstar Notes. It is an independent test document for the CDT503 project.", st["body"]),
        p("The three stages", st["h1"]),
        bullet("Ingest: read a PDF, split its text into overlapping chunks, and store each chunk with an embedding.", st["bullet"]),
        bullet("Retrieve: embed the user question and find the four most similar chunks in Chroma.", st["bullet"]),
        bullet("Generate: place the retrieved context and question in a prompt, then ask a local language model to answer.", st["bullet"]),
        p("Default settings", st["h1"]),
        table([
            ["Setting", "Northstar value", "Reason"],
            ["Embedding", "BGE-small", "Converts text into local numeric vectors."],
            ["Vector store", "Chroma", "Keeps vectors and source text on disk."],
            ["Default top-k", "4", "Four chunks are sent to the generator."],
            ["Chunk size", "512 characters", "Small enough for focused retrieval."],
            ["Chunk overlap", "80 characters", "Preserves context near chunk boundaries."],
        ], [42 * mm, 43 * mm, 75 * mm]),
        p("A useful warning", st["h1"]),
        p("Northstar Notes must not invent an answer when the retrieved context does not support it. For example, this handbook does not state who won a robotics championship. A careful system should say that the handbook does not contain that information.", st["body"]),
        p("Local operation", st["h1"]),
        p("The demonstration uses local models and does not send PDF text to OpenAI or another hosted API. On a GPU machine, the generator may use a quantized model. On a CPU-only machine, a smaller local model is used so the workflow can still run.", st["body"]),
        p("Document identifier: NSN-204", st["small"]),
    ]


def robot_story():
    st = styles()
    return [
        p("Overview", st["h1"]),
        p("Rover-6 is a fictional indoor campus robot used to deliver library books. The specifications in this document are deliberately new test facts.", st["body"]),
        p("Key specifications", st["h1"]),
        table([
            ["Property", "Value"],
            ["Maximum payload", "18 kilograms"],
            ["Maximum indoor speed", "1.5 meters per second"],
            ["Battery runtime", "72 minutes under normal load"],
            ["Charging time", "36 minutes from 20 percent to 80 percent"],
            ["Lidar detection range", "12 meters"],
            ["Emergency stop distance", "1.2 meters from a person"],
        ], [70 * mm, 90 * mm]),
        p("Operating procedure", st["h1"]),
        bullet("Before autonomous movement, check the lidar status and confirm that the emergency stop button is released.", st["bullet"]),
        bullet("If a person is detected within 1.2 meters, stop immediately and wait for a clear path.", st["bullet"]),
        bullet("Do not carry more than 18 kilograms, even if the shelf compartment appears to have extra space.", st["bullet"]),
        bullet("Return to the charging station when the estimated battery runtime falls below 12 minutes.", st["bullet"]),
        p("Maintenance", st["h1"]),
        p("The lidar lens is cleaned every Monday and Thursday. A technician records the battery health after every 50 delivery trips. The robot is not approved for outdoor rain, stairs, or passenger transport.", st["body"]),
        p("Document identifier: R6-OPS-118", st["small"]),
    ]


def policy_story():
    st = styles()
    return [
        p("Scope", st["h1"]),
        p("The fictional Meridian Lab uses this policy to evaluate local PDF question answering systems. The policy is an independent document for the CDT503 acceptance demo.", st["body"]),
        p("Scoring policy", st["h1"]),
        table([
            ["Metric or rule", "Required value"],
            ["Context recall weight", "60 percent"],
            ["Answer relevancy weight", "40 percent"],
            ["Minimum answer relevancy target", "0.70"],
            ["Minimum overall score", "0.75"],
            ["Maximum acceptable unsupported-answer rate", "5 percent"],
            ["Fixed random seed", "503"],
        ], [100 * mm, 60 * mm]),
        p("Required comparison", st["h1"]),
        bullet("Run one experiment with chunk_size 512 and another with chunk_size 1024.", st["bullet"]),
        bullet("Keep chunk_overlap fixed at 120 characters so the comparison changes only one main variable.", st["bullet"]),
        bullet("Use the same questions, embedding model, top-k value, and generator settings in both experiments.", st["bullet"]),
        p("Reproducibility checklist", st["h1"]),
        bullet("Record the model names and whether the run used CPU or GPU.", st["bullet"]),
        bullet("Save the raw metric values to CSV before drawing a bar chart.", st["bullet"]),
        bullet("Keep at least one failure example and explain whether the error came from retrieval or generation.", st["bullet"]),
        bullet("Do not use a hosted OpenAI API for this local-only experiment.", st["bullet"]),
        p("Document identifier: ML-EVAL-503", st["small"]),
    ]


def make_qa():
    return [
        {"question": "What are the three main stages in the Northstar Notes pipeline?", "ground_truth": "The three stages are ingest, retrieve, and generate.", "source": "demo_rag_handbook.pdf"},
        {"question": "What is the default top-k value in the Northstar Notes handbook?", "ground_truth": "The default top-k value is 4.", "source": "demo_rag_handbook.pdf"},
        {"question": "What chunk overlap does Northstar Notes use?", "ground_truth": "Northstar Notes uses an 80-character chunk overlap.", "source": "demo_rag_handbook.pdf"},
        {"question": "What is the maximum payload of Rover-6?", "ground_truth": "Rover-6 has a maximum payload of 18 kilograms.", "source": "demo_campus_robot.pdf"},
        {"question": "Which sensor must be checked before Rover-6 moves autonomously?", "ground_truth": "The lidar status must be checked before autonomous movement.", "source": "demo_campus_robot.pdf"},
        {"question": "How far from a person must Rover-6 stop?", "ground_truth": "Rover-6 must stop when a person is within 1.2 meters.", "source": "demo_campus_robot.pdf"},
        {"question": "What are the weights of context recall and answer relevancy in the Meridian policy?", "ground_truth": "Context recall has a 60 percent weight and answer relevancy has a 40 percent weight.", "source": "demo_research_policy.pdf"},
        {"question": "What is the minimum answer relevancy target?", "ground_truth": "The minimum answer relevancy target is 0.70.", "source": "demo_research_policy.pdf"},
        {"question": "Which chunk sizes must be compared in the Meridian Lab experiment?", "ground_truth": "The experiment must compare chunk_size 512 and chunk_size 1024.", "source": "demo_research_policy.pdf"},
        {"question": "Who won the 2025 campus robotics championship?", "ground_truth": "The demo documents do not contain this information, so it cannot be confirmed from the provided context.", "source": "not_in_documents"},
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_pdf(OUT_DIR / "demo_rag_handbook.pdf", "Northstar Notes RAG Handbook", "Synthetic document NSN-204 | Independent retrieval test", handbook_story())
    build_pdf(OUT_DIR / "demo_campus_robot.pdf", "Rover-6 Campus Robot Operations", "Synthetic document R6-OPS-118 | Independent retrieval test", robot_story())
    build_pdf(OUT_DIR / "demo_research_policy.pdf", "Meridian Lab RAG Evaluation Policy", "Synthetic document ML-EVAL-503 | Independent retrieval test", policy_story())
    QA_PATH.write_text(json.dumps(make_qa(), indent=2), encoding="utf-8")
    for path in sorted(OUT_DIR.glob("*.pdf")):
        print(path)
    print(QA_PATH)


if __name__ == "__main__":
    main()
