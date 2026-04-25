import gradio as gr
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# ---- Load model ----
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- Roles ----
job_roles = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer"
]

# Precompute embeddings
role_embeddings = model.encode(job_roles, convert_to_tensor=True)

# ---- PDF Extract ----
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    return text

# ---- Main Function ----
def analyze(resume_file, resume_text):

    if resume_file is not None:
        text = extract_pdf(resume_file)
    elif resume_text.strip():
        text = resume_text
    else:
        return "Please upload or paste resume."

    emb = model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(emb, role_embeddings)[0]

    top_idx = sims.cpu().numpy().argsort()[::-1][:3]

    result = []
    for i in top_idx:
        score = float(sims[i])
        percent = int(score * 100)
        result.append(f"{job_roles[i]} → {percent}%")

    return "\n".join(result)

# ---- UI ----
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload Resume PDF"),
        gr.Textbox(lines=10, label="Or paste resume text")
    ],
    outputs="text",
    title="💼 AI Career Assistant",
    description="Upload your resume or paste text to get career matches"
)

demo.launch()