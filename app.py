"""Milestone 5 — Gradio web interface for The Unofficial Guide.

Run:  python app.py   then open http://localhost:7860

Assumes the ChromaDB index already exists (run `python index.py` once first).
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "Which Temple CS professor do students most recommend?",
    "How heavy is the workload in Christopher Pascucci's web-dev courses?",
    "What do students say about Data Structures with David Dobor?",
    "How is Richard Beigel's class graded?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Type a question above and press Ask.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(no sources — outside the reviews)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide — Temple CS") as demo:
    gr.Markdown(
        "# The Unofficial Guide — Temple CS Professors\n"
        "Ask about Temple CS professors, course difficulty, and workload. "
        "Answers come **only** from collected student reviews (Rate My Professors); "
        "if the reviews don't cover your question, the guide will say so."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. Is Data Structures hard with Dobor?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    gr.Examples(EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
