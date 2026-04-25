import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# ---- Load Model ----
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---- Roles ----
job_roles = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer"
]

@st.cache_resource
def get_embeddings():
    return model.encode(job_roles, convert_to_tensor=True)

role_embeddings = get_embeddings()

# ---- Skills ----
role_skills = {
    "Data Scientist": ["Python", "Machine Learning", "Statistics"],
    "Machine Learning Engineer": ["Python", "Deep Learning", "TensorFlow"],
    "Backend Developer": ["Python", "APIs", "Databases"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript"]
}

# ---- Functions ----
def predict_roles(text):
    emb = model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(emb, role_embeddings)[0]

    top_idx = sims.cpu().numpy().argsort()[::-1][:3]

    return [
        {"role": job_roles[i], "score": float(sims[i])}
        for i in top_idx
    ]

def extract_skills(text):
    all_skills = set(sum(role_skills.values(), []))
    return [s for s in all_skills if s.lower() in text.lower()]

def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    return text

def interpret(score):
    if score > 0.7:
        return "🔥 Strong match"
    elif score > 0.5:
        return "✅ Moderate match"
    else:
        return "⚠️ Weak match"

# ---- UI ----
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align:center;'>💼 AI Career Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Analyze your resume and discover career paths</p>", unsafe_allow_html=True)

st.divider()

file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
text = st.text_area("Or paste resume text", height=200)

if file:
    text = extract_pdf(file)

if st.button("🚀 Analyze Resume"):
    if not text.strip():
        st.warning("Please enter resume text")
    else:
        roles = predict_roles(text)
        skills = extract_skills(text)

        st.success("Analysis Complete")

        # ---- Top Role Highlight ----
        st.info(f"🏆 Best Match: {roles[0]['role']}")

        # ---- Roles ----
        st.subheader("🎯 Top Career Matches")
        for i, r in enumerate(roles, 1):
            percent = int(r['score'] * 100)
            st.markdown(f"### {i}. {r['role']}")
            st.write(f"Score: {percent}% → {interpret(r['score'])}")
            st.progress(r['score'])
            st.divider()

        # ---- Skills ----
        st.subheader("🧠 Skill Analysis")

        top_role = roles[0]["role"]
        required = role_skills.get(top_role, [])

        skills_lower = [s.lower() for s in skills]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Detected Skills")
            if skills:
                for s in skills:
                    st.write(f"- {s}")
            else:
                st.warning("No strong skills detected")

        with col2:
            st.markdown("### ❌ Missing Skills")
            missing = [s for s in required if s.lower() not in skills_lower]
            if missing:
                for s in missing:
                    st.write(f"- {s}")
            else:
                st.write("No major gaps 🎯")

        # ---- Explanation ----
        st.subheader("📌 Why this role?")
        matched = [s for s in required if s.lower() in skills_lower]

        st.write("Matching skills:")
        for s in matched:
            st.write(f"- {s}")

        st.write("Required skills:")
        for s in required:
            st.write(f"- {s}")