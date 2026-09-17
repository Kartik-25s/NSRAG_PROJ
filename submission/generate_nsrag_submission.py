from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
ASSETS_DIR = SUBMISSION_DIR / "assets"
PPTX_PATH = SUBMISSION_DIR / "nsrag_submission_deck.pptx"


COLORS = {
    "navy": RGBColor(0x0B, 0x1F, 0x33),
    "teal": RGBColor(0x1F, 0x7A, 0x8C),
    "gold": RGBColor(0xF2, 0xA5, 0x41),
    "red": RGBColor(0xC4, 0x45, 0x36),
    "green": RGBColor(0x2A, 0x9D, 0x8F),
    "ink": RGBColor(0x23, 0x2F, 0x3E),
    "slate": RGBColor(0x4E, 0x5D, 0x6C),
    "paper": RGBColor(0xF7, 0xF4, 0xEA),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
    "light_teal": RGBColor(0xD9, 0xEE, 0xF2),
    "light_gold": RGBColor(0xFC, 0xE7, 0xC3),
    "light_red": RGBColor(0xF6, 0xD7, 0xD3),
    "light_green": RGBColor(0xD4, 0xF0, 0xEA),
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics() -> dict:
    return {
        "phase1": load_json(ROOT / "phase1/results/naive_rag_hotpotqa_scores.json"),
        "phase2": load_json(ROOT / "phase2/results/decomposer_eval_metrics.json"),
        "phase3": load_json(ROOT / "phase3/results/retrieval_metrics.json"),
        "phase4": load_json(ROOT / "phase4/results/nsrag_hotpotqa_metrics.json"),
    }


def ensure_dirs() -> None:
    SUBMISSION_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)


def build_architecture_figure() -> Path:
    out = ASSETS_DIR / "nsrag_architecture.png"
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F4EA")
    ax.set_facecolor("#F7F4EA")

    def box(x, y, w, h, title, body, fc, ec="#0B1F33"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.18",
            linewidth=2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + 0.18, y + h - 0.38, title, fontsize=15, fontweight="bold", color="#0B1F33")
        ax.text(x + 0.18, y + h - 0.78, body, fontsize=11, color="#23303E", va="top", wrap=True)

    box(0.4, 2.5, 2.0, 2.4, "Input Question", "Complex multi-hop query from HotpotQA or future scientific domains.", "#E7F2F4")
    box(3.0, 2.5, 2.4, 2.4, "Phase 2\nDecomposer", "Groq-backed DAG planner creates typed sub-questions with dependencies.", "#D9EEF2")
    box(6.1, 4.7, 2.6, 1.6, "Dense Retrieval", "Qdrant + BGE embeddings over indexed passages.", "#D4F0EA")
    box(6.1, 2.7, 2.6, 1.6, "KG Retrieval", "Optional Neo4j traversal for entity relations.", "#FCE7C3")
    box(6.1, 0.7, 2.6, 1.6, "Symbolic Solver", "SymPy branch for math and physics expressions.", "#F6D7D3")
    box(9.3, 2.5, 2.6, 2.4, "Phase 4\nAgent", "Topological hop execution, evidence traces, utility scoring, final synthesis.", "#E7F2F4")
    box(12.3, 2.5, 2.3, 2.4, "Phase 5\nVerifier", "Equation checks, chain-of-verification, attribution guard.", "#D9EEF2")

    arrow_style = dict(arrowstyle="->", color="#0B1F33", linewidth=2)
    ax.annotate("", xy=(3.0, 3.7), xytext=(2.4, 3.7), arrowprops=arrow_style)
    ax.annotate("", xy=(6.1, 5.5), xytext=(5.4, 4.2), arrowprops=arrow_style)
    ax.annotate("", xy=(6.1, 3.5), xytext=(5.4, 3.7), arrowprops=arrow_style)
    ax.annotate("", xy=(6.1, 1.5), xytext=(5.4, 3.2), arrowprops=arrow_style)
    ax.annotate("", xy=(9.3, 3.7), xytext=(8.7, 5.5), arrowprops=arrow_style)
    ax.annotate("", xy=(9.3, 3.7), xytext=(8.7, 3.5), arrowprops=arrow_style)
    ax.annotate("", xy=(9.3, 3.7), xytext=(8.7, 1.5), arrowprops=arrow_style)
    ax.annotate("", xy=(12.3, 3.7), xytext=(11.9, 3.7), arrowprops=arrow_style)

    ax.text(7.5, 7.45, "NSRAG Pipeline Implemented in the Current Repository", ha="center", va="center", fontsize=22, fontweight="bold", color="#0B1F33")
    ax.text(7.5, 6.95, "Strong decomposition + retrieval backbone, with Phase 4/5 completing the verifiable agent story.", ha="center", va="center", fontsize=12, color="#4E5D6C")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_results_figure(metrics: dict) -> Path:
    out = ASSETS_DIR / "nsrag_results_overview.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("#F7F4EA")

    p2_labels = ["DAG", "Hop +/-1", "Domain"]
    p2_values = [
        metrics["phase2"]["dag_validity_rate"],
        metrics["phase2"]["hop_accuracy_within1"],
        metrics["phase2"]["domain_tag_accuracy"],
    ]
    axes[0].bar(p2_labels, p2_values, color=["#1F7A8C", "#2A9D8F", "#F2A541"])
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Phase 2: Decomposer", fontsize=14, fontweight="bold", color="#0B1F33")
    for idx, val in enumerate(p2_values):
        axes[0].text(idx, val + 2, f"{val:.1f}", ha="center", fontsize=11, color="#23303E")

    p3_labels = ["R@5", "R@10", "MRR x100"]
    p3_values = [
        metrics["phase3"]["Recall@5"],
        metrics["phase3"]["Recall@10"],
        metrics["phase3"]["MRR"] * 100,
    ]
    axes[1].bar(p3_labels, p3_values, color=["#1F7A8C", "#2A9D8F", "#F2A541"])
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Phase 3: Retrieval", fontsize=14, fontweight="bold", color="#0B1F33")
    for idx, val in enumerate(p3_values):
        label = f"{val:.1f}" if idx < 2 else f"{metrics['phase3']['MRR']:.4f}"
        axes[1].text(idx, val + 2, label, ha="center", fontsize=11, color="#23303E")

    labels = ["EM", "F1"]
    achieved = [metrics["phase4"]["EM"], metrics["phase4"]["F1"]]
    targets = [10.0, 20.0]
    x = [0, 1]
    width = 0.35
    axes[2].bar([v - width / 2 for v in x], achieved, width=width, color="#1F7A8C", label="Achieved")
    axes[2].bar([v + width / 2 for v in x], targets, width=width, color="#F2A541", label="Target")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 26)
    axes[2].set_title("Phase 4: Agent", fontsize=14, fontweight="bold", color="#0B1F33")
    axes[2].legend(frameon=False, fontsize=10)
    for idx, val in enumerate(achieved):
        axes[2].text(idx - width / 2, val + 0.6, f"{val:.2f}", ha="center", fontsize=11, color="#23303E")
    for idx, val in enumerate(targets):
        axes[2].text(idx + width / 2, val + 0.6, f"{val:.0f}", ha="center", fontsize=11, color="#23303E")

    for ax in axes:
        ax.set_facecolor("#F7F4EA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#A9B2BA")
        ax.spines["bottom"].set_color("#A9B2BA")
        ax.tick_params(colors="#23303E")

    fig.suptitle("Achieved Results From Current Repository Artifacts", fontsize=20, fontweight="bold", color="#0B1F33", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def set_slide_background(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=24, color=None,
                bold=False, font_name="Aptos", align=PP_ALIGN.LEFT) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    run = p.runs[0]
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color or COLORS["ink"]


def add_bullets(slide, left, top, width, height, bullets, font_size=20, color=None) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for idx, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.bullet = True
        p.space_after = Pt(7)
        run = p.runs[0]
        run.font.name = "Aptos"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color or COLORS["ink"]


def add_header(slide, title, subtitle=None) -> None:
    header = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(0),
        Inches(0),
        Inches(13.33),
        Inches(0.55),
    )
    header.fill.solid()
    header.fill.fore_color.rgb = COLORS["navy"]
    header.line.fill.background()
    add_textbox(slide, Inches(0.45), Inches(0.07), Inches(8.7), Inches(0.35), title, 28, COLORS["white"], True, "Aptos Display")
    if subtitle:
        add_textbox(slide, Inches(9.15), Inches(0.1), Inches(3.6), Inches(0.28), subtitle, 11, COLORS["paper"], False, "Aptos", PP_ALIGN.RIGHT)


def add_footer_tag(slide, text="NSRAG | submission-ready snapshot") -> None:
    pill = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(9.8),
        Inches(7.02),
        Inches(3.05),
        Inches(0.36),
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = COLORS["teal"]
    pill.line.fill.background()
    add_textbox(slide, Inches(9.95), Inches(7.09), Inches(2.75), Inches(0.18), text, 10, COLORS["white"], False, "Aptos", PP_ALIGN.CENTER)


def add_card(slide, left, top, width, height, fill_rgb, title, body, title_color=None) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    shape.line.color.rgb = COLORS["white"] if fill_rgb != COLORS["white"] else COLORS["slate"]
    shape.line.width = Pt(1.2)
    add_textbox(slide, left + Inches(0.18), top + Inches(0.12), width - Inches(0.35), Inches(0.38), title, 18, title_color or COLORS["navy"], True, "Aptos Display")
    add_textbox(slide, left + Inches(0.18), top + Inches(0.52), width - Inches(0.35), height - Inches(0.7), body, 14, COLORS["ink"], False, "Aptos")


def build_deck(metrics: dict, architecture_path: Path, results_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["navy"])
    accent1 = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(9.4), Inches(-0.2), Inches(4.3), Inches(2.0))
    accent1.fill.solid()
    accent1.fill.fore_color.rgb = COLORS["gold"]
    accent1.line.fill.background()
    accent2 = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(10.2), Inches(5.8), Inches(3.5), Inches(2.2))
    accent2.fill.solid()
    accent2.fill.fore_color.rgb = COLORS["teal"]
    accent2.line.fill.background()
    add_textbox(slide, Inches(0.7), Inches(0.9), Inches(8.6), Inches(1.9), "Neuro-Symbolic-RAG", 28, COLORS["gold"], True, "Aptos Display")
    add_textbox(slide, Inches(0.7), Inches(1.75), Inches(8.8), Inches(1.4), "A Multi-Hop Agent Framework for Verifiable Scientific Reasoning", 26, COLORS["white"], True, "Aptos Display")
    add_textbox(slide, Inches(0.72), Inches(3.0), Inches(7.7), Inches(1.2), "Codebase review, achieved results, and submission framing based on the current repository state.", 17, COLORS["paper"])
    add_textbox(slide, Inches(0.72), Inches(4.0), Inches(5.7), Inches(0.4), "Phase 4 is implemented and evaluated; Phase 5 verifier is implemented and pending final benchmark reporting.", 14, COLORS["paper"])
    add_textbox(slide, Inches(0.72), Inches(4.48), Inches(4.8), Inches(0.3), "Presenter: Student Name", 12, COLORS["paper"])
    add_card(
        slide,
        Inches(0.72),
        Inches(5.15),
        Inches(4.05),
        Inches(1.2),
        COLORS["light_teal"],
        "Current headline",
        f"Phase 2 and 3 are strong. Phase 4 is near target with EM {metrics['phase4']['EM']:.1f} and F1 {metrics['phase4']['F1']:.2f}.",
    )
    add_textbox(slide, Inches(0.75), Inches(6.8), Inches(4.5), Inches(0.3), "Prepared on April 7, 2026", 11, COLORS["paper"])

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "Why This Project", "problem -> system -> evidence")
    add_card(
        slide,
        Inches(0.5),
        Inches(1.0),
        Inches(4.0),
        Inches(2.0),
        COLORS["light_gold"],
        "Research problem",
        "Multi-hop and scientific questions need more than a single retrieval step. Systems must decompose, route, reason, and verify.",
    )
    add_card(
        slide,
        Inches(4.66),
        Inches(1.0),
        Inches(4.05),
        Inches(2.0),
        COLORS["light_teal"],
        "NSRAG answer",
        "Use neural retrieval for evidence, symbolic structure for hop planning, and formal verification for trustworthiness.",
    )
    add_card(
        slide,
        Inches(8.88),
        Inches(1.0),
        Inches(3.95),
        Inches(2.0),
        COLORS["light_green"],
        "What the repo already proves",
        "This is not just an idea. The codebase already implements every major layer from decomposition to verification.",
    )
    add_textbox(slide, Inches(0.65), Inches(3.45), Inches(11.7), Inches(0.5), "Submission stance: present this as a strong prototype with validated modules and a nearly-complete end-to-end story.", 20, COLORS["navy"], True, "Aptos Display")
    add_bullets(
        slide,
        Inches(0.8),
        Inches(4.1),
        Inches(12.0),
        Inches(2.3),
        [
            "Phase 2 gives explicit and interpretable multi-hop plans.",
            "Phase 3 provides the neuro-symbolic bridge: dense retrieval, optional graph traversal, and SymPy reasoning.",
            "Phase 4 integrates the pipeline into an agent with hop traces, confidence, and final synthesis.",
            "Phase 5 adds a verifier stack, which makes the project stronger than a pure generation baseline.",
        ],
        18,
    )
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "What The Codebase Is", "repo structure at a glance")
    add_card(
        slide,
        Inches(0.55),
        Inches(1.0),
        Inches(4.15),
        Inches(5.6),
        COLORS["white"],
        "Primary modules",
        "phase1/: naive RAG baseline and FAISS index\n\nphase2/: symbolic decomposer and DAG planner\n\nphase3/: retrieval router over Qdrant, Neo4j, and SymPy\n\nphase4/: multi-hop agent with hop traces and synthesis\n\nphase5/: hallucination guard and formal verifier",
    )
    add_card(
        slide,
        Inches(4.95),
        Inches(1.0),
        Inches(3.85),
        Inches(2.65),
        COLORS["light_teal"],
        "Supporting assets",
        "eval/: shared EM/F1 harness\n\ndatasets/ and benchmarks/: HotpotQA, MuSiQue, 2Wiki, TheoremQA loaders\n\nvector_db/ and baselines/: external resources and comparisons",
    )
    add_card(
        slide,
        Inches(4.95),
        Inches(3.95),
        Inches(3.85),
        Inches(2.65),
        COLORS["light_gold"],
        "Core stack",
        "Groq for LLM calls\nBAAI bge-base-en-v1.5 for embeddings\nQdrant for local vector storage\nOptional Neo4j graph retrieval\nSymPy for symbolic solving and verification",
    )
    add_card(
        slide,
        Inches(9.05),
        Inches(1.0),
        Inches(3.75),
        Inches(5.6),
        COLORS["light_red"],
        "Important nuance",
        "The architecture is stronger than the current headline score.\n\nThe repo already contains early knowledge alignment, graph hooks, solver hooks, and verification gates, but some of those paths are scaffolded or optional in the present benchmark run.\n\nThat means the project is best framed as a complete prototype with one remaining integration bottleneck.",
    )
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "System Pipeline", "implemented architecture")
    slide.shapes.add_picture(str(architecture_path), Inches(0.45), Inches(0.9), width=Inches(12.4))
    add_textbox(slide, Inches(0.72), Inches(6.55), Inches(11.4), Inches(0.45), "Interpretation: the decomposition and retrieval backbone is already convincing; the current gap is turning that structure into stronger exact final answers.", 16, COLORS["navy"], True, "Aptos")
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "Phase-Wise Progress", "achieved results as of now")
    rows, cols = 6, 4
    table = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.15), Inches(12.3), Inches(4.9)).table
    headers = ["Phase", "Module", "Achieved Result", "Status"]
    widths = [0.85, 2.55, 6.7, 2.2]
    for idx, width in enumerate(widths):
        table.columns[idx].width = Inches(width)
    for col, text in enumerate(headers):
        cell = table.cell(0, col)
        cell.text = text
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLORS["white"]
        p.runs[0].font.size = Pt(16)
        p.alignment = PP_ALIGN.CENTER

    entries = [
        ["1", "Naive baseline", f"HotpotQA 1000-sample baseline: EM {metrics['phase1']['EM']}, F1 {metrics['phase1']['F1']}", "Complete"],
        ["2", "Decomposer", f"DAG validity {metrics['phase2']['dag_validity_rate']} | hop accuracy {metrics['phase2']['hop_accuracy_within1']} | domain {metrics['phase2']['domain_tag_accuracy']}", "Complete"],
        ["3", "Hybrid retrieval", f"Recall@5 {metrics['phase3']['Recall@5']} | Recall@10 {metrics['phase3']['Recall@10']} | MRR {metrics['phase3']['MRR']}", "Complete"],
        ["4", "NSRAG agent", f"EM {metrics['phase4']['EM']} | F1 {metrics['phase4']['F1']} | avg hops {metrics['phase4']['avg_hops']}", "In progress"],
        ["5", "Formal verifier", "Equation checks, CoVe, and attribution guard implemented; final benchmark report still pending", "Pending eval"],
    ]
    status_colors = {
        "Complete": COLORS["light_green"],
        "In progress": COLORS["light_gold"],
        "Pending eval": COLORS["light_red"],
    }
    for row_idx, row in enumerate(entries, start=1):
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = status_colors[text] if col_idx == 3 else COLORS["white"]
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(14)
                    run.font.color.rgb = COLORS["ink"]
                paragraph.alignment = PP_ALIGN.CENTER if col_idx in (0, 3) else PP_ALIGN.LEFT
    add_textbox(slide, Inches(0.68), Inches(6.35), Inches(11.7), Inches(0.4), "Best headline: phases 2 and 3 are solidly validated; phase 4 is implemented and close to the intended milestone.", 16, COLORS["navy"], True, "Aptos")
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "Current Results", "where the project stands today")
    slide.shapes.add_picture(str(results_path), Inches(0.55), Inches(1.0), width=Inches(7.1))
    add_card(
        slide,
        Inches(8.0),
        Inches(1.1),
        Inches(4.3),
        Inches(1.45),
        COLORS["light_green"],
        "Strongest evidence",
        "The decomposer and retrieval stack are already convincing and give the project real technical depth.",
    )
    add_card(
        slide,
        Inches(8.0),
        Inches(2.8),
        Inches(4.3),
        Inches(1.45),
        COLORS["light_gold"],
        "Current gap",
        f"Phase 4 is narrowly below target: EM {metrics['phase4']['EM']:.1f} vs 10, F1 {metrics['phase4']['F1']:.2f} vs 20.",
    )
    add_card(
        slide,
        Inches(8.0),
        Inches(4.5),
        Inches(4.3),
        Inches(1.55),
        COLORS["light_red"],
        "Honest comparison",
        f"The simple baseline still beats the current agent on HotpotQA answer quality: EM {metrics['phase1']['EM']} / F1 {metrics['phase1']['F1']}.",
    )
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["paper"])
    add_header(slide, "What I Think About The Project", "balanced assessment")
    add_card(
        slide,
        Inches(0.6),
        Inches(1.1),
        Inches(3.85),
        Inches(4.8),
        COLORS["light_teal"],
        "Why it is strong",
        "The project is structurally impressive: it has a phase plan, acceptance gates, reusable evaluation, interpretable traces, and a verifier design.\n\nThat already makes it more than a toy RAG demo.",
    )
    add_card(
        slide,
        Inches(4.74),
        Inches(1.1),
        Inches(3.85),
        Inches(4.8),
        COLORS["light_gold"],
        "What needs work",
        "The agent controller still loses too much quality between retrieval and final synthesis.\n\nA lot of the gap looks like orchestration and exactness, not a lack of project vision.",
    )
    add_card(
        slide,
        Inches(8.88),
        Inches(1.1),
        Inches(3.85),
        Inches(4.8),
        COLORS["light_green"],
        "Best framing for submission",
        "Call it a serious neuro-symbolic reasoning prototype with validated decomposition and retrieval, plus a near-complete agent/verifier stack.\n\nThat claim is ambitious, accurate, and defensible.",
    )
    add_textbox(slide, Inches(0.75), Inches(6.2), Inches(11.5), Inches(0.45), "Bottom line: the idea is good, the codebase is real, and the remaining gap is concentrated in Phase 4 quality tuning.", 18, COLORS["navy"], True, "Aptos Display")
    add_footer_tag(slide)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLORS["navy"])
    add_textbox(slide, Inches(0.8), Inches(0.85), Inches(7.0), Inches(0.8), "How To Present This Today", 27, COLORS["gold"], True, "Aptos Display")
    add_bullets(
        slide,
        Inches(0.9),
        Inches(1.8),
        Inches(7.6),
        Inches(3.4),
        [
            "Lead with the architecture and the phase-wise engineering story.",
            "Report decomposition and retrieval as achieved results, not only planned goals.",
            "Describe Phase 4 as implemented, evaluated, and close to target rather than fully complete.",
            "Describe Phase 5 as the verifier pipeline already coded and ready for final reporting.",
            "Say the project is moving from a strong prototype to a full verified reasoning system.",
        ],
        19,
        COLORS["white"],
    )
    add_card(
        slide,
        Inches(8.4),
        Inches(1.5),
        Inches(4.0),
        Inches(3.6),
        COLORS["paper"],
        "Closing message",
        "NSRAG already demonstrates a real neuro-symbolic multi-hop stack.\n\nThe most credible submission story is not finished SOTA performance, but a well-built and nearly-complete verifiable reasoning framework.",
    )
    add_textbox(slide, Inches(8.55), Inches(5.6), Inches(3.7), Inches(0.55), "Practical reminder: remove or rotate any API keys before sharing the repo or slides.", 12, COLORS["paper"])
    add_textbox(slide, Inches(0.9), Inches(6.85), Inches(4.8), Inches(0.3), "Files generated from current repo artifacts", 11, COLORS["paper"])

    prs.save(PPTX_PATH)


def main() -> None:
    ensure_dirs()
    metrics = load_metrics()
    architecture_path = build_architecture_figure()
    results_path = build_results_figure(metrics)
    build_deck(metrics, architecture_path, results_path)
    print(f"Created: {PPTX_PATH}")
    print(f"Assets: {architecture_path}, {results_path}")


if __name__ == "__main__":
    main()
