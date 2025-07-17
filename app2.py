import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Page Setup ===
st.set_page_config(page_title="BrainRay - MRI Classifier", layout="wide")

# === Constants ===
BACKGROUND_IMAGE_URL = "https://live.staticflickr.com/65535/54656913546_2e38526305_b.jpg"
MODEL_PATHS = {
    "NeuroVision™ (MobileNetV2)": r"C:\\Users\\jashp\\Downloads\\brain_tumor_app\\models\\model_mobilenetv2_tl.h5",
    "DeepScan™ (Custom CNN)": r"C:\\Users\\jashp\\Downloads\\brain_tumor_app\\models\\model_custom_cnn.h5"
}
BASE_DATASET_PATH = r"C:\\Users\\jashp\\Downloads\\Tumour"
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# === Global Session State ===
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# === CSS Styling ===
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Italiana&display=swap');
.stApp {{
    background-image: url('{BACKGROUND_IMAGE_URL}');
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    color: #f0eaff;
    font-family: 'Italiana', serif !important;
}}
h1, h2, h3, h4, h5, h6 {{
    color: #f0eaff;
    font-family: 'Italiana', serif !important;
}}
.big-title {{
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    animation: fadeIn 2s ease-in-out;
    margin-top: 3rem;
    font-family: 'Italiana', serif !important;
}}
.subtitle {{
    font-size: 20px;
    text-align: center;
    margin-bottom: 2rem;
    font-family: 'Italiana', serif !important;
}}
.animated-heading {{
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    color: #f0eaff;
    animation: fadeInScale 2s ease-in-out;
    margin-top: 2rem;
    margin-bottom: 2rem;
    font-family: 'Italiana', serif !important;
}}
button[kind="primary"],
.stButton > button {{
    background-color: #8e44ec;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.8rem 1.5rem;
    font-size: 18px;
    font-family: 'Italiana', serif !important;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: flex;
    justify-content: center;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 15px;
}}
button[kind="primary"]:hover {{
    background-color: #6c3483;
}}
html, body, [data-testid="stText"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaption"], .stSelectbox label, .stTextInput label,
.stRadio label, .stCheckbox label, .stTextInput > div > label {{
    font-family: 'Italiana', serif !important;
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(-20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeInScale {{
    0% {{ opacity: 0; transform: scale(0.8); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}
</style>
""", unsafe_allow_html=True)

# === Caching ===
@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def load_selected_model(path):
    return load_model(path)

# === Screens ===
def show_welcome_screen():
    st.markdown('<div class="big-title">Welcome to BrainRay</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">BrainRay is your personal diagnostic assistant for brain MRI scans. Upload scans, get predictions from advanced models, and consult with Dr. AI for brain tumor insights.</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🚀 Get Started"):
        st.session_state.page = "model_select"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_model_selection():
    st.markdown('<div class="animated-heading">Select Our Best Models</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 18px; max-width: 800px; margin: auto; margin-bottom: 30px;">
        Choose one of our top-performing AI models, each trained to analyze brain MRI scans with precision.
        Whether you're looking for real-time results or deep analytical power, we've got the right tool for you.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 NeuroVision™")
        st.markdown("**Model:** MobileNetV2  \n**Accuracy:** 92.3%  \n**Speed:** ⚡ Fast  \n**Best for:** Quick assessments on most MRI types.")
        if st.button("Select NeuroVision™"):
            st.session_state.selected_model = "NeuroVision™ (MobileNetV2)"
            st.session_state.page = "dataset_view"
            st.rerun()

    with col2:
        st.subheader("🔬 DeepScan™")
        st.markdown("**Model:** Custom CNN  \n**Accuracy:** 94.1%  \n**Speed:** 🧠 Deeper analysis  \n**Best for:** Detailed review of complex cases.")
        if st.button("Select DeepScan™"):
            st.session_state.selected_model = "DeepScan™ (Custom CNN)"
            st.session_state.page = "dataset_view"
            st.rerun()

    st.divider()
    if st.button("⬅ Back to Welcome"):
        st.session_state.page = "welcome"
        st.rerun()


def show_dataset_viewer():
    st.header("📂 Browse Dataset and Predict")
    model = load_selected_model(MODEL_PATHS[st.session_state.selected_model])
    dataset_split = st.selectbox("Select Dataset Split:", ["train", "valid", "test"])
    tumor_type = st.selectbox("Select Tumor Type:", ["glioma", "meningioma", "no_tumor", "pituitary"])

    folder = os.path.join(BASE_DATASET_PATH, dataset_split, tumor_type)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if files:
        selected_img = st.selectbox("Select an Image:", files)
        full_path = os.path.join(folder, selected_img)

        img = Image.open(full_path).convert('RGB')
        st.image(img, caption=f"Selected MRI: {tumor_type.title()}", use_column_width=True)

        resized = img.resize((224, 224))
        img_array = image.img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        pred_class = np.argmax(preds)
        confidence = preds[pred_class]

        st.markdown(f"### Prediction: **{CLASS_NAMES[pred_class]}**")
        st.markdown(f"Confidence: **{confidence:.2%}**")
        st.bar_chart({CLASS_NAMES[i]: preds[i] for i in range(4)})
    else:
        st.warning("No images found in the selected folder.")

    st.divider()
    if st.button("🧠 Talk with Dr. AI"):
        st.session_state.page = "dr_ai"
        st.rerun()

    if st.button("⬅ Back to Model Selection"):
        st.session_state.page = "model_select"
        st.rerun()

def show_dr_ai_assistant():
    st.header("🤖 Talk with Dr. AI")
    st.caption("Ask about brain tumors, symptoms, diagnosis, or treatments. Dr. AI will help using local medical data.")

    with st.spinner("✨ Initializing Dr. AI... Please wait."):
        vector_store = load_vector_store()
        llm = Llama(
            model_path="models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=20
        )

    user_q = st.text_input("💭 Ask Dr. AI a question")

    if user_q:
        with st.spinner("Dr. AI is thinking..."):
            docs = vector_store.similarity_search(user_q, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""### Instruction:
You are a helpful assistant for brain tumor-related queries using the given context. If the question is unrelated, say: "❌ Sorry, I can only help with brain tumor-related questions."

### Context:
{context}

### Question:
{user_q}

### Response:"""

            res = llm.create_completion(prompt=prompt, max_tokens=400, temperature=0.7, stop=["###"])
            answer = res["choices"][0]["text"].strip()

        st.success(answer)

    if st.button("⬅ Back to Dataset Viewer"):
        st.session_state.page = "dataset_view"
        st.rerun()

# === Page Navigation ===
if st.session_state.page == "welcome":
    show_welcome_screen()
elif st.session_state.page == "model_select":
    show_model_selection()
elif st.session_state.page == "dataset_view":
    show_dataset_viewer()
elif st.session_state.page == "dr_ai":
    show_dr_ai_assistant()

# === Footer ===
st.markdown("""
    <hr style="margin-top: 3rem; border: 1px solid #f0eaff;">
    <div style="text-align: center; color: #f0eaff; padding: 1rem; font-family: 'Italiana', serif;">
        <p style="font-size: 18px;">🚀 Project made by: <strong>Jash Parmar</strong></p>
        <div style="font-size: 24px;">
            <a href="https://www.instagram.com/jashparmar20" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30" height="30">
            </a>
            <a href="https://github.com/jashparmar23" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="30" height="30">
            </a>
            <a href="https://www.linkedin.com/in/jash-parmar-b2a62a283/" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30">
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
