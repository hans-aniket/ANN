import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD MODEL + DATA (LOCAL)
# =========================

model = tf.keras.models.load_model("model/internship_recommender.keras")

students_df = pd.read_csv("data/pm_internship_students_final.csv")
intern_df = pd.read_csv("data/pm_internship_internships_mergedV1.csv")

# Clean skills
def clean_skills(s):
    return [x.strip().lower().replace(" ", "_") for x in s.split(",")]

students_df["Skills_List"] = students_df["Skills"].apply(clean_skills)
intern_df["Required_Skills_List"] = intern_df["Required_Skills"].apply(clean_skills)

# Build skill vocabulary
all_skills = set()
for s in students_df["Skills_List"]:
    all_skills.update(s)
for s in intern_df["Required_Skills_List"]:
    all_skills.update(s)

skill_to_id = {skill: idx+1 for idx, skill in enumerate(sorted(all_skills))}

# Convert to ID sequences
def skills_to_ids(skill_list):
    return [skill_to_id[s] for s in skill_list if s in skill_to_id]

intern_df["ReqSeq"] = intern_df["Required_Skills_List"].apply(skills_to_ids)

max_len = max(intern_df["ReqSeq"].apply(len).max(), 4)

intern_df["ReqSeq_Padded"] = intern_df["ReqSeq"].apply(
    lambda x: pad_sequences([x], maxlen=max_len, padding="post")[0]
)

# =========================
# PDF → TEXT
# =========================

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = PyPDF2.PdfReader(uploaded_file)
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text.lower()

# =========================
# EXTRACT FEATURES FROM RESUME
# =========================

def extract_skills(text):
    found = []
    for sk in skill_to_id.keys():
        if sk.replace("_", " ") in text or sk in text:
            found.append(sk)
    return list(set(found))

def extract_projects(text):
    result = []
    for line in text.split("\n"):
        if "project" in line or "built" in line or "created" in line:
            result.append(line.strip())
    return result

def extract_experience(text):
    result = []
    for line in text.split("\n"):
        if "intern" in line or "experience" in line:
            result.append(line.strip())
    return result

def extract_education(text):
    patterns = ["btech", "b.sc", "b.com", "bca", "mtech", "m.sc", "mba"]
    found = [p for p in patterns if p in text]
    return found

# =========================
# RECOMMENDER
# =========================

def recommend_from_skills(skills, top_k=5):
    seq = skills_to_ids(skills)
    seq_padded = pad_sequences([seq], maxlen=max_len, padding="post")

    intern_seqs = np.array(intern_df["ReqSeq_Padded"].tolist())
    student_seqs = np.array([seq_padded[0]] * len(intern_df))

    preds = model.predict([student_seqs, intern_seqs], verbose=0).flatten()

    top_idx = preds.argsort()[::-1][:top_k]
    results = intern_df.iloc[top_idx].copy()
    results["score"] = preds[top_idx]

    return results[[
        "Internship_ID", "Required_Skills_List", "Sector",
        "Location", "Mode", "Duration", "Stipend", "score"
    ]]

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="AI Internship Recommender", layout="wide")

st.title("🚀 AI Internship Recommendation System")
st.write("Upload your resume (PDF) to get internship recommendations.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📄 Resume Text Preview")
    st.write(text[:1500] + ("..." if len(text) > 1500 else ""))

    skills = extract_skills(text)
    projects = extract_projects(text)
    experience = extract_experience(text)
    education = extract_education(text)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Skills Found")
        st.success(", ".join(skills) if skills else "No skills found")

        st.markdown("### 🎓 Education")
        st.info(", ".join(education) if education else "Not detected")

    with col2:
        st.markdown("### 💼 Experience")
        st.warning("\n".join(experience) if experience else "No experience detected")

        st.markdown("### 🛠 Projects")
        st.write("\n\n".join(projects) if projects else "No projects detected")

    if st.button("🔮 Recommend Internships"):
        if not skills:
            st.error("No skills detected in resume!")
        else:
            with st.spinner("Finding matches..."):
                results = recommend_from_skills(skills, top_k=5)

            st.subheader("⭐ Top Recommendations")

            for _, row in results.iterrows():
                st.markdown(f"""
                ### 🏢 Internship: {row['Internship_ID']}
                **Required Skills:** {", ".join(row['Required_Skills_List'])}  
                **Sector:** {row['Sector']}  
                **Location:** {row['Location']}  
                **Mode:** {row['Mode']}  
                **Duration:** {row['Duration']}  
                **Stipend:** ₹{row['Stipend']}  
                **Compatibility Score:** `{round(row['score'], 3)}`  
                """)
