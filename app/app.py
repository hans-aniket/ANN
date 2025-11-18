import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("model/internship_recommender.keras")

students_df = pd.read_csv("data/pm_internship_students_final.csv")
intern_df = pd.read_csv("data/pm_internship_internships_mergedV1.csv")

def clean_skills(s):
    return [x.strip().lower().replace(" ", "_") for x in str(s).split(",")]

intern_df["Required_Skills_List"] = intern_df["Required_Skills"].apply(clean_skills)

all_skills = set()
for s in intern_df["Required_Skills_List"]:
    all_skills.update(s)

skill_to_id = {skill: idx + 1 for idx, skill in enumerate(sorted(all_skills))}

def skills_to_ids(skill_list):
    return [skill_to_id[s] for s in skill_list if s in skill_to_id]

intern_df["ReqSeq"] = intern_df["Required_Skills_List"].apply(skills_to_ids)

max_len = max(intern_df["ReqSeq"].apply(len).max(), 4)
intern_df["ReqSeq_Padded"] = intern_df["ReqSeq"].apply(
    lambda x: pad_sequences([x], maxlen=max_len, padding="post")[0]
)


def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = PyPDF2.PdfReader(uploaded_file)
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.lower()


def extract_skills(text):
    found = []
    for skill in skill_to_id.keys():
        skill_clean = skill.replace("_", " ")
        if skill in text or skill_clean in text:
            found.append(skill)
    return list(set(found))

def extract_projects(text):
    result = []
    for line in text.split("\n"):
        if "project" in line or "built" in line or "developed" in line:
            result.append(line.strip())
    return result

def extract_experience(text):
    result = []
    for line in text.split("\n"):
        if "intern" in line or "experience" in line:
            result.append(line.strip())
    return result

def extract_education(text):
    keywords = ["btech", "b.sc", "b.com", "bca", "mtech", "m.sc", "mba"]
    return [e for e in keywords if e in text]


def recommend_from_skills(skills, top_k=5):
    seq = skills_to_ids(skills)
    if len(seq) == 0:
        return None

    seq_padded = pad_sequences([seq], maxlen=max_len, padding='post')[0]

    student_batch = np.array([seq_padded] * len(intern_df))
    intern_batch = np.array(intern_df["ReqSeq_Padded"].tolist())

    preds = model.predict([student_batch, intern_batch], verbose=0).flatten()
    top_idx = preds.argsort()[::-1][:top_k]

    results = intern_df.iloc[top_idx].copy()
    results["score"] = preds[top_idx]
    return results


st.set_page_config(page_title="AI Internship Recommender", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; font-size:40px;'>AI Internship Recommendation System</h1>
    <p style='text-align:center; font-size:18px;'>
    Upload your resume to receive internship recommendations powered by a dual-LSTM AI model.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.markdown("### Extracted Resume Text (Preview)")
    st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    skills = extract_skills(text)
    projects = extract_projects(text)
    exp = extract_experience(text)
    edu = extract_education(text)

    st.markdown("---")
    st.subheader("Resume Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Skills Found")
        if skills:
            skill_tags = " ".join([f"<span style='background:#e3f2fd; padding:5px 10px; border-radius:6px; margin-right:5px;'>{s}</span>"
                                   for s in skills])
            st.markdown(skill_tags, unsafe_allow_html=True)
        else:
            st.warning("No known skills detected from vocabulary.")

        st.markdown("### Education Detected")
        st.info(", ".join(edu) if edu else "Not detected")

    with col2:
        st.markdown("### Experience")
        st.write("\n".join(exp) if exp else "No experience detected")

        st.markdown("### Projects")
        st.write("\n".join(projects) if projects else "No projects detected")

    st.markdown("---")

    if st.button("Get Internship Recommendations"):
        if not skills:
            st.error("Cannot recommend without detected skills.")
        else:
            with st.spinner("Calculating compatibility scores..."):
                results = recommend_from_skills(skills)

            if results is None:
                st.error("No matching skills to evaluate.")
            else:
                st.subheader("Top Recommendations")

                for _, row in results.iterrows():
                    st.markdown(
                        f"""
                        <div style='padding:15px; border:1px solid #ddd; border-radius:10px; margin-bottom:18px;'>
                            <h3 style='margin:0;'>{row['Internship_ID']}</h3>
                            <p><strong>Required Skills:</strong> {", ".join(row['Required_Skills_List'])}</p>
                            <p><strong>Sector:</strong> {row['Sector']} | <strong>Location:</strong> {row['Location']}</p>
                            <p><strong>Mode:</strong> {row['Mode']} | <strong>Duration:</strong> {row['Duration']}</p>
                            <p><strong>Stipend:</strong> ₹{row['Stipend']}</p>
                            <p><strong>Compatibility Score:</strong> <span style='font-size:20px;'>{round(row['score'], 3)}</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
