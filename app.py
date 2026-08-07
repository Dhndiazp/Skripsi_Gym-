import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# 1. KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Gym Intensity Recommender",
    layout="wide",
    page_icon="🏋️"
)

# ============================================
# 2. CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .sub-header { color: #9CA3AF; font-size: 0.95rem; text-align: center; margin-bottom: 1.5rem; line-height: 1.5; }
    .profile-card {
        background-color: #16171D; border: 1px solid #2D2D2D; border-radius: 16px;
        padding: 2rem 2.5rem; margin: 0 auto 1.5rem auto;
    }
    .profile-title { font-size: 1.15rem; font-weight: 700; text-align: center; margin-bottom: 1.2rem; }
    .result-card { padding: 1.5rem; border-radius: 14px; margin-bottom: 1rem; }
    .card-high { background: linear-gradient(135deg, #3B0D0D 0%, #5A1616 100%); border: 1px solid #7A2222; }
    .card-low { background: linear-gradient(135deg, #0D1F3B 0%, #16305A 100%); border: 1px solid #22497A; }
    .badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.5px; }
    .badge-high { background-color: #EF4444; color: white; }
    .badge-low { background-color: #3B82F6; color: white; }
    .section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.8rem; border-bottom: 2px solid #2D2D2D; padding-bottom: 6px; }
    .tip-item { padding: 10px 14px; background-color: #1A1A1A; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #4B5563; font-size: 0.92rem; color: #E5E7EB; }
    .meta-info { font-size: 0.85rem; color: #9CA3AF; margin-bottom: 12px; }
    .meta-badge {
        display: inline-block; background-color: #1F2937; padding: 3px 10px;
        border-radius: 6px; margin-right: 6px; font-size: 0.8rem; color: #D1D5DB;
    }
    .warning-box {
        padding: 12px 16px; background-color: #3D2E0A; border: 1px solid #8A6D1A;
        border-radius: 10px; color: #F5D061; font-size: 0.9rem; margin-top: 1rem;
    }
    .metric-label { color: #9CA3AF; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button { border-radius: 8px; font-weight: 600; padding: 0.6rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 3. LOAD MODEL & ASSETS
# ============================================
@st.cache_resource
def load_assets():
    data = joblib.load('gym_model_v9.pkl')
    return data['model'], data['scaler'], data['features'], data['cat_names']

model, scaler, features, cat_names = load_assets()

# ============================================
# 4. RULE-BASED / KNOWLEDGE-BASED RECOMMENDATION
# ============================================
def get_age_group(age):
    if age <= 25:
        return 0
    elif age <= 35:
        return 1
    elif age <= 50:
        return 2
    elif age <= 65:
        return 3
    else:
        return 4

AGE_GROUP_LABELS = {
    0: "Remaja/Muda (≤25)",
    1: "Dewasa Muda (26-35)",
    2: "Dewasa Tengah (36-50)",
    3: "Setengah Baya (51-65)",
    4: "Senior (>65)",
}

EXP_LABELS = {1: "Pemula", 2: "Menengah", 3: "Ahli"}

EXERCISE_RULES = {
    0: {
        "Ringan": {
            1: ["Jalan santai", "Peregangan dasar", "Yoga pemula"],
            2: ["Jalan cepat", "Yoga dasar", "Peregangan dinamis", "Bersepeda santai"],
            3: ["Yoga", "Pilates", "Mobility training", "Peregangan aktif"],
        },
        "Berat": {
            1: ["Jogging ringan", "Latihan beban dasar", "Bersepeda", "Renang santai"],
            2: ["Sepak bola", "Futsal", "Basket", "Latihan beban", "Lari Maraton", "Badminton"],
            3: ["HIIT", "CrossFit", "Latihan beban intensif", "Bela diri", "Sprint interval"],
        },
    },
    1: {
        "Ringan": {
            1: ["Jalan cepat", "Yoga dasar", "Pilates pemula"],
            2: ["Yoga", "Pilates", "Bersepeda santai", "Jalan cepat"],
            3: ["Yoga lanjutan", "Pilates intensif", "Mobility training"],
        },
        "Berat": {
            1: ["Jogging", "Latihan beban dasar", "Bersepeda", "Renang"],
            2: ["Lari Maraton", "Gym", "Cross training", "Badminton", "Tenis"],
            3: ["HIIT", "Latihan beban progresif", "CrossFit", "Hiking berat", "Renang intensif"],
        },
    },
    2: {
        "Ringan": {
            1: ["Jalan kaki", "Yoga dasar", "Peregangan"],
            2: ["Jalan cepat", "Yoga", "Pilates", "Badminton rekreasional"],
            3: ["Yoga lanjutan", "Pilates intensif", "Hiking ringan"],
        },
        "Berat": {
            1: ["Jogging ringan", "Bersepeda santai", "Renang santai"],
            2: ["Jogging", "Bersepeda", "Renang", "Gym intensitas sedang"],
            3: ["Gym intensitas tinggi", "Tenis rekreasional", "Hiking", "Latihan beban"],
        },
    },
    3: {
        "Ringan": {
            1: ["Jalan santai", "Tai chi", "Peregangan lembut"],
            2: ["Jalan kaki", "Yoga", "Senam aerobik ringan"],
            3: ["Yoga", "Pilates", "Tai chi lanjutan"],
        },
        "Berat": {
            1: ["Jalan cepat", "Bersepeda santai", "Renang santai"],
            2: ["Bersepeda", "Renang", "Golf", "Senam aerobik"],
            3: ["Latihan kekuatan ringan", "Bersepeda jarak jauh", "Renang intensif"],
        },
    },
    4: {
        "Ringan": {
            1: ["Jalan santai", "Peregangan", "Duduk-berdiri ringan"],
            2: ["Senam lansia", "Tai chi", "Yoga lansia"],
            3: ["Tai chi lanjutan", "Yoga lansia", "Latihan keseimbangan"],
        },
        "Berat": {
            1: ["Jalan cepat ringan", "Renang ringan"],
            2: ["Aqua aerobics", "Bersepeda statis", "Renang ringan"],
            3: ["Latihan kekuatan ringan", "Aqua aerobics intensif", "Bersepeda statis sedang"],
        },
    },
}

# ============================================
# 5. WORKOUT TYPE MAPPING
# ============================================
EXERCISE_TYPE_MAP = {
    "Jalan santai": "Cardio", "Jalan cepat": "Cardio", "Jalan kaki": "Cardio",
    "Jalan cepat ringan": "Cardio", "Bersepeda santai": "Cardio", "Bersepeda": "Cardio",
    "Bersepeda jarak jauh": "Cardio", "Bersepeda statis": "Cardio", "Bersepeda statis sedang": "Cardio",
    "Renang santai": "Cardio", "Renang": "Cardio", "Renang ringan": "Cardio", "Renang intensif": "Cardio",
    "Jogging ringan": "Cardio", "Jogging": "Cardio", "Lari Maraton": "Cardio",
    "Sepak bola": "Cardio", "Futsal": "Cardio", "Basket": "Cardio", "Badminton": "Cardio",
    "Badminton rekreasional": "Cardio", "Tenis": "Cardio", "Tenis rekreasional": "Cardio",
    "Hiking": "Cardio", "Hiking ringan": "Cardio", "Hiking berat": "Cardio",
    "Senam aerobik ringan": "Cardio", "Senam aerobik": "Cardio", "Senam lansia": "Cardio",
    "Golf": "Cardio", "Aqua aerobics": "Cardio", "Aqua aerobics intensif": "Cardio",

    "Latihan beban dasar": "Strength", "Latihan beban": "Strength", "Latihan beban intensif": "Strength",
    "Latihan beban progresif": "Strength", "Gym": "Strength", "Gym intensitas sedang": "Strength",
    "Gym intensitas tinggi": "Strength", "Latihan kekuatan ringan": "Strength",
    "Duduk-berdiri ringan": "Strength",

    "HIIT": "HIIT", "CrossFit": "HIIT", "Bela diri": "HIIT", "Sprint interval": "HIIT",
    "Cross training": "HIIT",

    "Peregangan dasar": "Yoga", "Peregangan dinamis": "Yoga", "Peregangan aktif": "Yoga",
    "Peregangan": "Yoga", "Peregangan lembut": "Yoga", "Yoga pemula": "Yoga", "Yoga dasar": "Yoga",
    "Yoga": "Yoga", "Yoga lanjutan": "Yoga", "Yoga lansia": "Yoga", "Pilates": "Yoga",
    "Pilates pemula": "Yoga", "Pilates intensif": "Yoga", "Mobility training": "Yoga",
    "Tai chi": "Yoga", "Tai chi lanjutan": "Yoga", "Latihan keseimbangan": "Yoga",
}

TYPE_COLORS = {
    "Cardio":   {"bg": "#E3F2FD", "text": "#0D47A1"},
    "Strength": {"bg": "#FFF3E0", "text": "#E65100"},
    "HIIT":     {"bg": "#FCE4EC", "text": "#AD1457"},
    "Yoga":     {"bg": "#E8F5E9", "text": "#2E7D32"},
}

def get_workout_type(exercise_name):
    return EXERCISE_TYPE_MAP.get(exercise_name, "Cardio")

# ============================================
# 6. HEADER
# ============================================
st.markdown('<p class="main-header">🏋️ Gym Intensity & Exercise Recommender</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Interpretasi cerdas untuk profil latihan Anda. '
    'Lengkapi form di bawah untuk mengetahui apakah Anda sebaiknya melakukan '
    'latihan <b>Intensitas Tinggi</b> atau <b>Rendah</b>, lengkap dengan rekomendasi jenis olahraga '
    'berbasis <i>rule-based system</i>.</p>',
    unsafe_allow_html=True
)

# ============================================
# 7. FORM PROFILE - DI TENGAH
# ============================================
left_pad, center, right_pad = st.columns([1, 2.5, 1])

with center:
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<p class="profile-title">👤 User Profile</p>', unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        exp = st.selectbox("Level Pengalaman", [("Pemula", 1), ("Menengah", 2), ("Ahli", 3)], format_func=lambda x: x[0])
        rbpm = st.number_input("Resting BPM (Detak Jantung Istirahat)", 40, 110, 65)
        fat = st.slider("Persentase Lemak Tubuh (%)", 5.0, 50.0, 20.0)
    with row1_c2:
        age = st.slider("Usia", 10, 90, 30)
        freq = st.slider("Frekuensi Latihan (hari/minggu)", 1, 7, 3)
        wgt = st.number_input("Berat Badan (kg)", 40.0, 150.0, 70.0)
        wtr = st.slider("Asupan Air Harian (Liter)", 0.5, 5.0, 2.5)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("💡 Analisis Intensitas Olahraga", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 8. LOGIC PROCESSING (Model Klasifikasi RF)
# ============================================
user_data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "experience_level": exp[1],
    "workout_frequency_days_week": freq,
    "water_intake_liters": wtr,
    "resting_bpm": rbpm,
    "fat_percentage": fat,
    "weight_kg": wgt,
}

inp_df = pd.DataFrame([user_data])
inp_df["age_group"] = pd.cut(inp_df["age"], bins=[0, 25, 35, 50, 65, 120], labels=[0, 1, 2, 3, 4]).astype(int)
inp_df["freq_group"] = pd.cut(inp_df["workout_frequency_days_week"], bins=[0, 2, 4, 7], labels=[0, 1, 2]).astype(int)

# Catatan: pastikan semua kolom fitur terbentuk agar tidak error saat predict.
for col in features:
    if col not in inp_df.columns:
        inp_df[col] = 0

inp_scaled = scaler.transform(inp_df[features])

# ============================================
# 9. PREDIKSI & REKOMENDASI (RULE-BASED)
# ============================================
if analyze_btn:
    prediction = model.predict(inp_scaled)[0]
    probs = model.predict_proba(inp_scaled)[0]
    conf_score = max(probs) * 100

    # --- Ambil rekomendasi dari rule-based system ---
    age_group = get_age_group(age)
    intensity_key = "Berat" if prediction == "High" else "Ringan"
    exp_level = exp[1]
    exercises = EXERCISE_RULES[age_group][intensity_key][exp_level]

    # --- Kelompokkan exercises berdasarkan workout type ---
    grouped = {"Cardio": [], "Strength": [], "HIIT": [], "Yoga": []}
    for ex in exercises:
        wtype = get_workout_type(ex)
        grouped[wtype].append(ex)

    col_pad_l, col1, col2, col_pad_r = st.columns([0.5, 1, 1.2, 0.5])

    with col1:
        st.markdown('<p class="section-title">🎯 Hasil Prediksi Model</p>', unsafe_allow_html=True)

        if prediction == "High":
            st.markdown("""
            <div class="result-card card-high">
                <span class="badge badge-high">🔴 HIGH INTENSITY</span>
                <p style="margin-top:12px; margin-bottom:0;">
                    Model menyarankan latihan dengan beban kerja yang tinggi.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card card-low">
                <span class="badge badge-low">🔵 LOW INTENSITY</span>
                <p style="margin-top:12px; margin-bottom:0;">
                    Model menyarankan latihan yang lebih ringan/moderat.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="metric-label">Confidence Score</p>', unsafe_allow_html=True)
        st.progress(int(conf_score))
        st.markdown(f"**{conf_score:.1f}%**")

    with col2:
        st.markdown('<p class="section-title">💪 Saran Jenis Latihan</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="meta-info">
            <span class="meta-badge">📅 {AGE_GROUP_LABELS[age_group]}</span>
            <span class="meta-badge">🎓 {EXP_LABELS[exp_level]}</span>
        </div>
        """, unsafe_allow_html=True)

        # Render tiap grup dengan warna khusus
        groups_html = ""
        for wtype, ex_list in grouped.items():
            if not ex_list:
                continue
            colors = TYPE_COLORS[wtype]
            items_html = "".join([f"<div class='tip-item'>🏃 {ex}</div>" for ex in ex_list])
            
            groups_html += f"""
            <div style='margin-bottom:14px;'>
                <span style='background-color:{colors["bg"]}; color:{colors["text"]};
                             font-size:0.85em; font-weight:700; padding:4px 12px;
                             border-radius:20px; display:inline-block; margin-bottom:8px;'>
                    {wtype}
                </span>
                <div style='margin-top:4px;'>{items_html}</div>
            </div>
            """
        
        st.markdown(groups_html, unsafe_allow_html=True)

        if age >= 50:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <b>Catatan:</b> Mengingat usia Anda, pastikan melakukan pemanasan sendi lebih lama.
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👆 Lengkapi profil Anda di atas, lalu klik tombol **Analisis Intensitas Olahraga** untuk melihat hasil.")