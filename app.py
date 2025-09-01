# app.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Perakende İçerik Tabanlı Ürün Önerisi", layout="wide")

# =========================
# Yardımcı Fonksiyonlar
# =========================
@st.cache_data(show_spinner=False)
def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """TR/EN sütun adlarını tek forma getir, temel temizlik yap."""
    col_map = {
        "product_id": ["product_id","ürün_kimliği","urun_kimligi","urun_id","id"],
        "title": ["title","başlık","baslik"],
        "description": ["description","tanım","tanim","aciklama","açıklama"],
        "category": ["category","kategori"]
    }
    resolved = {}
    for k, cands in col_map.items():
        for c in df_raw.columns:
            if c.strip().lower() in [x.strip().lower() for x in cands]:
                resolved[k] = c
                break
    missing = [k for k in col_map if k not in resolved]
    if missing:
        raise ValueError(f"Eksik sütun(lar): {missing}. CSV başlıklarını kontrol edin.")

    df = df_raw.rename(columns={
        resolved["product_id"]: "product_id",
        resolved["title"]: "title",
        resolved["description"]: "description",
        resolved["category"]: "category",
    }).copy()

    df["title"] = df["title"].fillna("Unknown")
    df["description"] = df["description"].fillna(df["title"])
    df["category"] = df["category"].fillna("Unknown")
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_vec_X_sim(df: pd.DataFrame):
    """TF-IDF vectorizer, X matrisi ve tam benzerlik matrisini (cosine) döndür."""
    text = (df["title"].astype(str) + " | "
            + df["description"].astype(str) + " | "
            + df["category"].astype(str))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(text)
    sim = cosine_similarity(X, X)
    return vec, X, sim


def mmr_rerank(sim_row: np.ndarray, X, lambda_mult: float = 0.7, topn: int = 5, exclude_idx: int | None = None):
    """
    MMR (Maximal Marginal Relevance) ile çeşitlilik artıran yeniden sıralama.
    lambda_mult -> 1: alaka ağırlıklı, 0: çeşitlilik ağırlıklı
    """
    n = sim_row.shape[0]
    candidates = [i for i in range(n) if i != exclude_idx] if exclude_idx is not None else list(range(n))
    selected = []
    # öneriler arası benzerlik için matris
    item_sims = cosine_similarity(X, X)

    while candidates and len(selected) < topn:
        if not selected:
            next_idx = int(np.argmax(sim_row[candidates]))
        else:
            best_score, next_idx = -1e18, None
            for i in candidates:
                relevance = sim_row[i]
                diversity = max(item_sims[i, selected]) if selected else 0.0
                score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score, next_idx = score, i
        selected.append(next_idx)
        candidates.remove(next_idx)
    return selected


def recommend(df: pd.DataFrame, sim_matrix, idx: int, topn: int = 5, same_category: bool = True):
    """Klasik (MMR’siz) en yakın komşulara göre öneri."""
    sims = list(enumerate(sim_matrix[idx]))
    sims = sorted((x for x in sims if x[0] != idx), key=lambda x: x[1], reverse=True)

    if same_category:
        base_cat = df.iloc[idx]["category"]
        sims = [x for x in sims if df.iloc[x[0]]["category"] == base_cat]

    sims = sims[:topn]
    rec = pd.DataFrame({
        "rank": np.arange(1, len(sims)+1),
        "product_id": [df.iloc[i]["product_id"] for i,_ in sims],
        "title": [df.iloc[i]["title"] for i,_ in sims],
        "category": [df.iloc[i]["category"] for i,_ in sims],
        "similarity": [float(s) for _, s in sims]
    })
    return rec


def search_products(query: str, vec: TfidfVectorizer, X, df: pd.DataFrame, topn: int = 10):
    """Serbest metin arama → en yakın ürünler."""
    q = vec.transform([query])
    sims = cosine_similarity(q, X)[0]
    order = sims.argsort()[::-1][:topn]
    return pd.DataFrame({
        "rank": range(1, len(order)+1),
        "product_id": df.iloc[order]["product_id"].values,
        "title": df.iloc[order]["title"].values,
        "category": df.iloc[order]["category"].values,
        "score": sims[order]
    })


# =========================
# UI
# =========================
st.title("🛍️ Perakende İçerik Tabanlı Ürün Önerisi")

# Veri girişi
mode = st.radio("Veri girişi yöntemi", ["Dosya yolu", "Dosya yükle"], horizontal=True)
df = None
if mode == "Dosya yolu":
    csv_path = st.text_input("CSV dosya yolu", value="retail_product_recommendation.csv")
    if csv_path:
        try:
            df = normalize_columns(pd.read_csv(csv_path))
        except Exception as e:
            st.error(f"Yükleme hatası: {e}")
else:
    up = st.file_uploader("CSV yükle", type=["csv"])
    if up is not None:
        try:
            df = normalize_columns(pd.read_csv(up))
        except Exception as e:
            st.error(f"Yükleme hatası: {e}")

if df is None:
    st.stop()

st.success(f"{len(df)} satır yüklendi.")
with st.expander("Örnek ilk satırlar"):
    st.dataframe(df.head(10))

# Model hazırla
vec, X, sim = build_vec_X_sim(df)

# --------- Free-text arama ---------
st.subheader("🔎 Serbest metin arama")
q = st.text_input("Bir arama yazın (örn: 'wireless headphones noise cancelling')", "")
if q.strip():
    res = search_products(q, vec, X, df, topn=10)
    st.dataframe(res, use_container_width=True)

st.markdown("---")

# --------- Öneriler ---------
same_category = st.checkbox("Aynı kategoriyle kısıtla", value=True)
topn = st.slider("Kaç öneri gösterilsin?", 3, 20, 5)
use_mmr = st.checkbox("Çeşitlilik için MMR kullan", value=False)
if use_mmr:
    lambda_mult = st.slider("MMR λ (alaka–çeşitlilik dengesi)", 0.10, 0.95, 0.70, 0.05)
else:
    lambda_mult = 0.70  # kullanılmayacak

# Ürün seçimi
options = [f"{row.title} | {row.product_id}" for _, row in df.iterrows()]
selected = st.selectbox("Bir ürün seç:", options, index=0)
sel_id = selected.split("|")[-1].strip()
idx = int(df.index[df["product_id"] == sel_id][0])

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Seçilen Ürün**")
    st.json({
        "product_id": df.iloc[idx]["product_id"],
        "title": df.iloc[idx]["title"],
        "category": df.iloc[idx]["category"],
        "description": df.iloc[idx]["description"]
    }, expanded=False)

with col2:
    st.markdown("**Önerilen Ürünler**")
    if use_mmr:
        # seçilen ürüne göre benzerlik vektörü
        sel_sim_row = cosine_similarity(X[idx], X)[0]
        # kategori filtresi isteniyorsa maskeyle uygula
        if same_category:
            base_cat = df.iloc[idx]["category"]
            mask = (df["category"] == base_cat).values
            sel_sim_row = np.where(mask, sel_sim_row, -1e9)
        chosen = mmr_rerank(sel_sim_row, X, lambda_mult=lambda_mult, topn=topn, exclude_idx=idx)
        rec = pd.DataFrame({
            "rank": range(1, len(chosen)+1),
            "product_id": df.iloc[chosen]["product_id"].values,
            "title": df.iloc[chosen]["title"].values,
            "category": df.iloc[chosen]["category"].values,
            "similarity": cosine_similarity(X[idx], X[chosen]).flatten()
        })
    else:
        rec = recommend(df, sim, idx, topn=topn, same_category=same_category)

    # kısa açıklama kolonu (önizleme)
    rec = rec.merge(df[["product_id","description"]], on="product_id", how="left")
    rec["description"] = rec["description"].str.slice(0, 120) + "..."
    st.dataframe(rec, use_container_width=True)

    st.download_button(
        "Önerileri CSV indir",
        rec.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{df.iloc[idx]['product_id']}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("Kategori dağılımı")
st.bar_chart(df["category"].value_counts())
st.subheader("En sık 10 başlık")
st.bar_chart(df["title"].value_counts().head(10))

st.info("Çalıştır:  `streamlit run app.py`")
