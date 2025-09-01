# 🛍️ Content-Based Retail Recommender (Streamlit)

İçerik tabanlı ürün öneri sistemi. TF-IDF + Cosine Similarity ile benzer ürünleri bulur, **kategori filtresi** ve **MMR** ile çeşitliliği yönetir. Serbest metin arama, öneri listesini CSV indirme ve hızlı EDA grafikleri içerir.

## ✨ Özellikler
- CSV ile veri yükleme (dosya yolu veya upload)
- TF-IDF tabanlı benzerlik hesaplama
- **Aynı kategoriyle kısıtlama**, **aynı başlığı hariç tutma**
- **MMR** (Maximal Marginal Relevance) ile çeşitlilik odaklı yeniden sıralama
- **Serbest metin arama** (free-text query)
- **CSV indirme** (öneri sonuçları)
- Basit grafikler: kategori ve başlık dağılımı

## 🚀 Kurulum
```bash
git clone https://github.com/<kullanıcı-adın>/content-based-retail-recommender.git
cd content-based-retail-recommender
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Çalıştırma
streamlit run app.py
| sütun         | açıklama                               |
| ------------- | -------------------------------------- |
| `product_id`  | benzersiz ürün kimliği                 |
| `title`       | ürün başlığı                           |
| `description` | ürün açıklaması (öneri için ana metin) |
| `category`    | ürün kategorisi                        |
Nasıl Çalışır? (Under the Hood)

TF-IDF vektörleme: title + description + category birleştirilip n-gram (1,2) ile temsil edilir.

Cosine similarity: Her ürünün diğerleriyle benzerliği hesaplanır.

Öneri: Seçilen ürün için en yüksek benzerliğe sahip N ürün getirilir.

MMR (opsiyonel): Alaka–çeşitlilik dengesi için yeniden sıralama yapılır.

Arama: Kullanıcı sorgusu TF-IDF uzayına dönüştürülür; en yakın ürünler listelenir.

Uygulama Ayarları
| Ayar                           | Açıklama                                          |
| ------------------------------ | ------------------------------------------------- |
| **Dosya yolu / Dosya yükle**   | CSV’yi diskten seç veya UI’dan yükle              |
| **Aynı kategori ile kısıtla**  | Öneriler yalnızca seçilen ürünle aynı kategoriden |
| **Kaç öneri gösterilsin?**     | Top-N (3–20)                                      |
| **Çeşitlilik için MMR kullan** | MMR re-ranking’i aç/kapat                         |
| **MMR λ**                      | 1.0 → alaka, 0.1 → çeşitlilik ağırlıklı           |
| **Serbest metin arama**        | Sorgu yaz, benzer ürünler listelensin             |
| **CSV indir**                  | Öneri tablosunu dışa aktar                        |
Referanslar
<p align="center"><img src="screenshots/search.png" width="90%"/></p>
<p align="center"><img src="screenshots/recommendations.png" width="90%"/></p>

...
# 🛍️ Content-Based Retail Recommender (Streamlit)

A **content-based product recommender system** for retail.  
Uses **TF-IDF + Cosine Similarity** on product titles/descriptions, with optional **category filtering**, **MMR re-ranking for diversity**, and **free-text search**.  
Run it with a single command: `streamlit run app.py`.

<p align="center">
  <img src="screenshots/home.png" width="90%" alt="Home screenshot"/>
</p>

---

## ✨ Features
- Load data from CSV (**file path** or **upload**)
- TF-IDF–based similarity matrix
- **Restrict by same category** (optional)
- **MMR re-ranking** for diversity (optional)
- **Free-text search** (e.g., “wireless headphones noise cancelling”)
- Download recommendations as **CSV**
- Quick EDA: **Category distribution** and **Top-10 product titles**

---

## 🚀 Quickstart

```bash
# 1) Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
# or:
# pip install streamlit scikit-learn pandas numpy

# 3) Run the app
streamlit run app.py

| column        | description                                    |
| ------------- | ---------------------------------------------- |
| `product_id`  | unique product identifier                      |
| `title`       | product title                                  |
| `description` | product description (main text for similarity) |
| `category`    | product category                               |

How It Works

TF-IDF vectorization: combine title + description + category and represent with n-grams (1,2).

Cosine similarity: compute similarity across all products.

Recommendation: return top-N most similar items to a selected product.

MMR (optional): re-rank to balance relevance vs. diversity.

APP Settings:
Free-text search: transform query into TF-IDF space and return closest products.
| Setting                       | Description                                  |
| ----------------------------- | -------------------------------------------- |
| **File path / Upload**        | Load CSV from disk or via upload             |
| **Restrict by same category** | Recommendations only from the same category  |
| **Top-N recommendations**     | Choose 3–20 items                            |
| **Use MMR**                   | Toggle Maximal Marginal Relevance re-ranking |
| **MMR λ**                     | 1.0 → relevance, 0.1 → diversity focused     |
| **Free-text search**          | Enter a query; see top matching items        |
| **Download CSV**              | Export current recommendation list           |



