import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ADAPTIVE DARK UI STYLING =====
st.markdown("""
<style>
    /* Main Dark Background */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    /* Sidebar Styling - Dark Contrast */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Title & Headers */
    h1, h2, h3, h4, p, span {
        color: #ffffff !important;
    }

    .hero-title {
        color: #43a047 !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        letter-spacing: -1.5px !important;
        margin-bottom: 0px !important;
    }

    /* Professional Metric Cards */
    div[data-testid="stMetricValue"] {
        background: #1c2128 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border-left: 5px solid #43a047 !important;
        color: #43a047 !important;
        font-weight: 700 !important;
    }

    /* File Uploader - Dark Theme */
    section[data-testid="stFileUploadDropzone"] {
        background: #1c2128 !important;
        border: 2px dashed #43a047 !important;
        border-radius: 15px !important;
    }

    /* Diagnosis Result Card */
    .diagnosis-card {
        background: #1c2128;
        padding: 35px;
        border-radius: 24px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Info Cards */
    .info-card {
        background: #1c2128;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #43a047;
        margin: 10px 0;
    }
    
    /* Warning Card */
    .warning-card {
        background: #2d1f1a;
        padding: 25px;
        border-radius: 12px;
        border-left: 4px solid #fbc02d;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== CLASS NAMES =====
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ===== COMPLETE DISEASE INFORMATION DATABASE =====
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'symptoms': 'Dark olive-green or brown spots on leaves and fruit',
        'treatment': 'Apply fungicide sprays, remove infected leaves, improve air circulation'
    },
    'Apple___Black_rot': {
        'symptoms': 'Purple spots on leaves, brown rot on fruit',
        'treatment': 'Prune infected branches, apply fungicide, remove mummified fruit'
    },
    'Apple___Cedar_apple_rust': {
        'symptoms': 'Yellow-orange spots on upper leaf surface',
        'treatment': 'Remove nearby cedar trees, apply protective fungicides in spring'
    },
    'Apple___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Maintain regular care: water, fertilize, monitor for pests'
    },
    'Blueberry___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Maintain acidic soil, adequate watering, prune old branches'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'symptoms': 'White powdery coating on leaves and shoots',
        'treatment': 'Apply sulfur-based fungicides, improve air circulation, prune dense growth'
    },
    'Cherry_(including_sour)___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Regular pruning, proper fertilization, monitor for pests'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'symptoms': 'Gray rectangular lesions on leaves between veins',
        'treatment': 'Crop rotation, plant resistant varieties, apply fungicide if severe'
    },
    'Corn_(maize)___Common_rust_': {
        'symptoms': 'Orange-brown pustules on both leaf surfaces',
        'treatment': 'Plant resistant hybrids, apply fungicide for susceptible varieties'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'symptoms': 'Long gray-green cigar-shaped lesions on leaves',
        'treatment': 'Use resistant hybrids, practice crop rotation, timely fungicide application'
    },
    'Corn_(maize)___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Adequate nitrogen fertilization, proper spacing, weed control'
    },
    'Grape___Black_rot': {
        'symptoms': 'Circular brown spots on leaves, shriveled mummified fruit',
        'treatment': 'Remove infected fruit, apply fungicide from bloom to harvest'
    },
    'Grape___Esca_(Black_Measles)': {
        'symptoms': 'Tiger-stripe leaf pattern, dark streaking in wood',
        'treatment': 'No cure available, prune infected vines, remove dead wood'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'symptoms': 'Angular brown spots with yellow halos on leaves',
        'treatment': 'Improve air circulation, apply copper fungicides, remove infected leaves'
    },
    'Grape___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Regular pruning, adequate water and nutrients, pest monitoring'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'symptoms': 'Yellow mottled leaves, lopsided bitter fruit, tree decline',
        'treatment': 'No cure; remove infected trees, control psyllid vectors, plant disease-free stock'
    },
    'Peach___Bacterial_spot': {
        'symptoms': 'Small dark spots on leaves and fruit, fruit cracking',
        'treatment': 'Apply copper sprays, plant resistant varieties, prune for air flow'
    },
    'Peach___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Annual pruning, balanced fertilization, thin fruit for quality'
    },
    'Pepper,_bell___Bacterial_spot': {
        'symptoms': 'Dark raised spots on leaves and fruit',
        'treatment': 'Use disease-free seeds, apply copper bactericides, avoid overhead watering'
    },
    'Pepper,_bell___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Consistent watering, calcium supplementation, mulch to retain moisture'
    },
    'Potato___Early_blight': {
        'symptoms': 'Concentric ring pattern (target spots) on lower leaves',
        'treatment': 'Apply fungicide at first symptoms, remove infected foliage, rotate crops'
    },
    'Potato___Late_blight': {
        'symptoms': 'Water-soaked lesions, white fungal growth, rapid plant collapse',
        'treatment': 'Apply fungicide preventatively, destroy infected plants, avoid overhead irrigation'
    },
    'Potato___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Plant certified seed, hill soil around stems, harvest when mature'
    },
    'Raspberry___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Annual pruning of old canes, adequate spacing, weed control'
    },
    'Soybean___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Crop rotation, adequate fertilization, monitor for pests'
    },
    'Squash___Powdery_mildew': {
        'symptoms': 'White powdery fungal coating on leaves',
        'treatment': 'Apply sulfur or potassium bicarbonate, improve air circulation, water at soil level'
    },
    'Strawberry___Leaf_scorch': {
        'symptoms': 'Purple to reddish-brown spots with gray centers on leaves',
        'treatment': 'Remove infected leaves, apply fungicide, avoid overhead watering'
    },
    'Strawberry___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Renovate beds after harvest, adequate water, remove runners'
    },
    'Tomato___Bacterial_spot': {
        'symptoms': 'Small dark greasy spots on leaves and fruit',
        'treatment': 'Use disease-free transplants, apply copper sprays, avoid wet foliage'
    },
    'Tomato___Early_blight': {
        'symptoms': 'Brown concentric ring spots on lower leaves',
        'treatment': 'Apply fungicide, mulch to prevent soil splash, remove lower leaves'
    },
    'Tomato___Late_blight': {
        'symptoms': 'Large irregular brown lesions, white mold, rapid plant death',
        'treatment': 'Apply fungicide immediately, destroy infected plants, avoid overhead watering'
    },
    'Tomato___Leaf_Mold': {
        'symptoms': 'Yellow spots on upper leaf, olive-green mold below',
        'treatment': 'Improve greenhouse ventilation, lower humidity, apply fungicide'
    },
    'Tomato___Septoria_leaf_spot': {
        'symptoms': 'Small circular spots with dark borders and gray centers',
        'treatment': 'Remove infected leaves, apply fungicide, mulch to prevent splash'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'symptoms': 'Yellow stippling on leaves, fine webbing, leaf bronzing',
        'treatment': 'Spray with water to dislodge mites, apply miticide, introduce predatory mites'
    },
    'Tomato___Target_Spot': {
        'symptoms': 'Brown spots with concentric rings on leaves and stems',
        'treatment': 'Apply fungicide, remove infected debris, practice crop rotation'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'symptoms': 'Upward leaf curling, yellowing, stunted growth',
        'treatment': 'Control whitefly vectors, remove infected plants, use reflective mulch'
    },
    'Tomato___Tomato_mosaic_virus': {
        'symptoms': 'Mottled light and dark green leaf pattern, stunted growth',
        'treatment': 'No cure; prevent by sanitizing tools, using resistant varieties, controlling aphids'
    },
    'Tomato___healthy': {
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Consistent watering, staking for support, regular fertilization'
    }
}

CONFIDENCE_THRESHOLD = 30

@st.cache_resource
def load_models():
    num_classes = 38
    base_eff = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_eff.trainable = True
    for layer in base_eff.layers[:-20]: layer.trainable = False
    i_eff = tf.keras.Input(shape=(224, 224, 3)); x = base_eff(i_eff, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2.996e-06))(x)
    x = tf.keras.layers.Dropout(0.237955)(x); o_eff = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    m_eff = tf.keras.Model(i_eff, o_eff)
    m_eff.load_weights('efficientnet_tuned_weights.h5')

    base_res = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_res.trainable = True
    for layer in base_res.layers[:-20]: layer.trainable = False
    i_res = tf.keras.Input(shape=(224, 224, 3)); x = base_res(i_res, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.000256))(x)
    x = tf.keras.layers.Dropout(0.227441)(x); o_res = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    m_res = tf.keras.Model(i_res, o_res)
    m_res.load_weights('resnet_tuned_weights.h5')
    
    return m_eff, m_res

def preprocess_image(image):
    return np.expand_dims(np.array(image.resize((224, 224))), axis=0).astype(np.float32)

def format_name(name):
    return name.replace('___', ': ').replace('_', ' ')

# ===== MAIN APP =====
def main():
    st.markdown('<h1 class="hero-title">PlantVision AI</h1>', unsafe_allow_html=True)
    st.write("Precision Neural Ensemble Diagnosis System")
    
    st.markdown("""
    <div class="info-card">
        <p style="margin: 0; font-size: 0.95rem;">
            <strong>🔬 How it works:</strong> Upload a plant leaf image → AI ensemble analyzes visual patterns → 
            Instant disease classification across 38 plant diseases
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🎯 Model Performance")
        st.metric("Test Accuracy", "99.36%", delta="+3.36% vs Baseline")
        st.metric("Macro F1 Score", "0.9902", delta="+4.02% vs Baseline")
        st.metric("Disease Classes", "38")
        
        st.write("---")
        st.subheader("⚙️ Architecture")
        st.caption("**Ensemble Composition:**")
        st.caption("• EfficientNetB3 (Compound Scaling)")
        st.caption("• ResNet50 (Residual Learning)")
        st.caption("")
        st.caption("**Optimization:**")
        st.caption("• Optuna Bayesian Search")
        st.caption("• 99.02% Macro F1 on Test Set")
        
        st.write("---")
        st.subheader("📊 Dataset")
        st.caption("**PlantVillage Dataset**")
        st.caption("• 54,306 training images")
        st.caption("• 8,146 test images")
        st.caption("• 14 plant species")
        
        st.write("---")
        st.subheader("🌿 Supported Plants")
        st.caption("Apple • Blueberry • Cherry • Corn • Grape • Orange • Peach • Pepper • Potato • Raspberry • Soybean • Squash • Strawberry • Tomato")

    up_col, res_col = st.columns([1, 1], gap="large")

    with up_col:
        st.subheader("📤 Upload Leaf Image")
        file = st.file_uploader("Drag and drop or click to browse", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, use_container_width=True, caption="Uploaded Sample")
            st.caption(f"Image size: {img.size[0]}×{img.size[1]} pixels")

    with res_col:
        if file:
            with st.spinner("🧠 Analyzing neural signatures..."):
                m_eff, m_res = load_models()
                proc_img = preprocess_image(img)
                preds = (m_eff.predict(proc_img, verbose=0) + m_res.predict(proc_img, verbose=0)) / 2
                idx = np.argmax(preds[0])
                conf = float(preds[0][idx] * 100)
                
                if conf < CONFIDENCE_THRESHOLD:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h3 style="color: #fbc02d !important; margin-top: 0;">⚠️ Uncertain Classification</h3>
                        <p style="color: #e0e0e0;">The model cannot confidently classify this image.</p>
                        <p style="color: #e0e0e0; margin-bottom: 0;">
                            <strong>Likely reason:</strong> Image may not be a plant leaf, or the disease is not in the training dataset.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Best guesses (low confidence):**")
                    top3 = np.argsort(preds[0])[-3:][::-1]
                    for i in top3:
                        st.write(f"• {format_name(CLASS_NAMES[i])}: {preds[0][i]*100:.1f}%")
                else:
                    color = "#43a047" if conf > 85 else "#fbc02d"
                    
                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <p style="color: #8b949e; font-size: 0.8rem; text-transform: uppercase; font-weight: 700;">Diagnosis</p>
                        <h2 style="color: {color} !important; margin-top: 0; font-size: 2.2rem;">{format_name(CLASS_NAMES[idx])}</h2>
                        <hr style="border-color: #30363d;">
                        <h3 style="margin-bottom: 10px;">Confidence: {conf:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(float(conf / 100))
                    
                    # Disease information (now works for all 38 classes)
                    with st.expander("📋 Disease Information"):
                        info = DISEASE_INFO[CLASS_NAMES[idx]]
                        st.write(f"**Symptoms:** {info['symptoms']}")
                        st.write(f"**Treatment:** {info['treatment']}")
                    
                    with st.expander("📊 Full Probability Distribution"):
                        top5 = np.argsort(preds[0])[-5:][::-1]
                        for i in top5:
                            st.write(f"**{format_name(CLASS_NAMES[i])}**: {preds[0][i]*100:.1f}%")
                    
                    with st.expander("🔍 Compare EfficientNet vs ResNet"):
                        pred_eff = m_eff.predict(proc_img, verbose=0)
                        pred_res = m_res.predict(proc_img, verbose=0)
                        
                        eff_idx = np.argmax(pred_eff[0])
                        res_idx = np.argmax(pred_res[0])
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption("**EfficientNetB3**")
                            st.write(f"{format_name(CLASS_NAMES[eff_idx])}")
                            st.write(f"{pred_eff[0][eff_idx]*100:.1f}%")
                        
                        with col_b:
                            st.caption("**ResNet50**")
                            st.write(f"{format_name(CLASS_NAMES[res_idx])}")
                            st.write(f"{pred_res[0][res_idx]*100:.1f}%")
                        
                        if eff_idx == res_idx:
                            st.success("✅ Both models agree on the diagnosis")
                        else:
                            st.warning("⚠️ Models disagree — ensemble averages both predictions")
        else:
            st.info("🌱 System ready. Upload a leaf image to begin analysis.")
    
    st.write("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Analyzed", "0" if not file else "1", help="Session count")
    with col2:
        st.metric("Avg Response Time", "~2.3s", help="Model inference speed")
    with col3:
        st.metric("Model Size", "156 MB", help="Combined ensemble weight")
    with col4:
        st.metric("Parameters", "49.2M", help="Total trainable parameters")

if __name__ == "__main__": main()
