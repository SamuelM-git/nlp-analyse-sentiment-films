import streamlit as st
import joblib
import re
# import numpy as np

# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse de sentiment 🎬",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# Chargement des modèles
# --------------------------------------------------
model = joblib.load("models/logistic_regression_tfidf.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# --------------------------------------------------
# Initialisation du session state
# --------------------------------------------------
if "text_key" not in st.session_state:
    st.session_state.text_key = 0

# --------------------------------------------------
# Fonction de nettoyage du texte
# (identique au notebook)
# --------------------------------------------------


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------------
# Fonction d'explication de la prédiction
# --------------------------------------------------


def explain_prediction(text, vectorizer, model, top_n=5):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    word_scores = {}
    for idx in vec.nonzero()[1]:
        word_scores[feature_names[idx]] = coefs[idx] * vec[0, idx]

    sorted_words = sorted(
        word_scores.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return sorted_words[:top_n]

# --------------------------------------------------
# Interface utilisateur
# --------------------------------------------------


st.title("🎬 Analyse de sentiment d’avis de films")

st.write(
    "Cette application prédit le **sentiment (positif ou négatif)** "
    "d’un avis de film à l’aide d’un modèle NLP "
    "(TF-IDF + Régression Logistique)."
)

# Zone de saisie
user_input = st.text_area(
    "✍️ Entrez un avis de film :",
    height=150,
    key=f"text_{st.session_state.text_key}"
)

# Boutons
col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyser")

with col2:
    reset = st.button("♻️ Reset")

# Reset
if reset:
    st.session_state.text_key += 1
    st.rerun()

# --------------------------------------------------
# Analyse du sentiment
# --------------------------------------------------
if analyze:
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        clean = clean_text(user_input)
        vectorized = tfidf.transform([clean])

        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        st.markdown("---")

        if prediction == 1:
            st.success(f"✅ **Sentiment POSITIF** "
                       f"(probabilité : {proba[1]:.2f})")
        else:
            st.error(f"❌ **Sentiment NÉGATIF** (probabilité : {proba[0]:.2f})")

        # --------------------------------------------------
        # Mots les plus influents
        # --------------------------------------------------

        important_words = explain_prediction(clean, tfidf, model)

        st.markdown("### 🔎 Mots les plus influents dans la prédiction")
        for word, score in important_words:
            st.write(f"- **{word}** ({score:.3f})")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Projet NLP — TF-IDF + Régression Logistique | "
    "By Samuel M "
)
