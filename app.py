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

    scores = sims.cpu().numpy()
    top_idx = scores.argsort()[::-1][:3]

    max_score = scores[top_idx[0]]

    result = "🎯 TOP CAREER MATCHES\n\n"

    for i in top_idx:
        role = job_roles[i]
        relative = scores[i] / max_score
        percent = int(relative * 100)

        if percent > 75:
            level = "Strong match"
        elif percent > 50:
            level = "Moderate match"
        else:
            level = "Weak match"

        result += f"{role} → {percent}% ({level})\n"

    # ---- Skills ----
    all_skills = set(sum(role_skills.values(), []))
    detected = [s for s in all_skills if s.lower() in text.lower()]

    top_role = job_roles[top_idx[0]]
    required = role_skills[top_role]
    missing = [s for s in required if s not in detected]

    result += "\n🧠 DETECTED SKILLS:\n"
    if detected:
        for s in detected:
            result += f"- {s}\n"
    else:
        result += "None detected\n"

    result += "\n❌ MISSING SKILLS (for top role):\n"
    if missing:
        for s in missing:
            result += f"- {s}\n"
    else:
        result += "None\n"

    result += "\n📌 WHY THIS ROLE?\n"
    matched = [s for s in required if s in detected]

    result += "Matching skills:\n"
    for s in matched:
        result += f"- {s}\n"

    result += "\nRequired skills:\n"
    for s in required:
        result += f"- {s}\n"

    return result

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