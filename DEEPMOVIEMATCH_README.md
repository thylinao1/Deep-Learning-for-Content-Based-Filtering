# 🎬 DeepMovieMatch: Neural Collaborative Filtering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-green.svg)](https://keras.io/)

> **A two-tower neural architecture for movie recommendations using learned embeddings in shared latent space**

---

## 📋 Executive Summary

DeepMovieMatch is a neural network–based movie recommender system that predicts user preferences using **learned embeddings** rather than simple genre matching. Unlike traditional content-based filtering that relies on manual feature engineering, this system automatically discovers latent representations that capture complex, nonlinear relationships between users and movies.

**Key Innovation:** Dual neural encoders project users and movies into a shared 32-dimensional embedding space where semantic similarity drives recommendations—enabling the model to recommend movies with zero genre overlap if user taste patterns align.

---

## 🎯 Problem Statement

### Traditional Recommender Limitations

| Approach | Method | Limitation |
|----------|--------|------------|
| **Content-Based** | Genre/tag matching | Binary features miss nuanced preferences |
| **Simple Collaborative** | Matrix factorization | Linear relationships only |
| **Hybrid (Manual)** | Hand-crafted features | Requires domain expertise |

**Solution:** End-to-end deep learning that learns **what matters** from data, not rules.

---

## 🏗️ Architecture

### Two-Tower Neural Network

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT FEATURES                          │
├──────────────────────────┬──────────────────────────────────┤
│   User Features          │   Movie Features                 │
│   • User ID              │   • Movie ID                     │
│   • Avg Rating           │   • Year                         │
│   • Rating Count         │   • Genre Flags (14D)            │
│   • Genre Preferences    │                                  │
└──────────┬───────────────┴───────────┬──────────────────────┘
           │                           │
           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐
    │  USER TOWER │            │  ITEM TOWER │
    │             │            │             │
    │  Dense(256) │            │  Dense(256) │
    │    ReLU     │            │    ReLU     │
    │             │            │             │
    │  Dense(128) │            │  Dense(128) │
    │    ReLU     │            │    ReLU     │
    │             │            │             │
    │  Dense(32)  │            │  Dense(32)  │
    │   Linear    │            │   Linear    │
    └──────┬──────┘            └──────┬──────┘
           │                          │
           │   L2 Normalize           │   L2 Normalize
           │                          │
           └───────────┬──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  DOT PRODUCT   │
              │   (Similarity) │
              └────────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │  Predicted   │
               │    Rating    │
               └──────────────┘
```

### Model Specifications

**User Tower:**
- Input: User features (excluding ID, rating count, avg rating during training)
- Architecture: 256 → 128 → 32 neurons
- Activation: ReLU for hidden layers, Linear for output
- Output: 32-dimensional user embedding (L2-normalized)

**Item Tower:**
- Input: Movie features (excluding movie ID during training)
- Architecture: 256 → 128 → 32 neurons (mirrored structure)
- Activation: ReLU for hidden layers, Linear for output
- Output: 32-dimensional movie embedding (L2-normalized)

**Similarity Function:**
```python
rating_prediction = dot_product(user_embedding, movie_embedding)
```

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam  
**Training Split:** 80/20 train-test

---

## 🧠 Why This Works: The Latent Space Insight

### Beyond Boolean Matching

Traditional systems ask: *"Do the genre flags overlap?"*

**DeepMovieMatch asks:** *"Do these user-movie pairs live close together in learned preference space?"*

### The Magic of Embeddings

The dense layers transform simple input features (genre flags, ratings) into **continuous embeddings** that capture:

- **Hidden taste profiles:** "Users who like philosophical sci-fi"
- **Cross-genre patterns:** "Cerebral thrillers regardless of setting"
- **Temporal trends:** "90s nostalgia fans"
- **Mood archetypes:** "Feel-good weekend watches"

**Result:** Two movies with **zero genre overlap** can be neighbors in embedding space if the same user cohort enjoys both.

### Example Scenario

```
User Preference Vector (learned):
[0.82, -0.34, 0.91, ..., -0.12]  # 32 dimensions

Movie A (Inception): 
[0.79, -0.29, 0.88, ..., -0.15]  # Similarity: 0.94

Movie B (Memento):
[0.81, -0.31, 0.90, ..., -0.13]  # Similarity: 0.96 ← Recommended!

Movie C (Fast & Furious):
[-0.22, 0.76, -0.45, ..., 0.83]  # Similarity: 0.12 ← Not recommended
```

The network learned that "mind-bending narratives" cluster together—**without anyone telling it what that means**.

---

## 📊 Key Features

✅ **Automatic Feature Learning** – No manual feature engineering required  
✅ **Nonlinear Relationships** – MLPs capture complex interactions  
✅ **Shared Latent Space** – Users and movies projected into common geometry  
✅ **Embedding-Based Similarity** – Movie-to-movie recommendations via distance  
✅ **Cold-Start Capable** – New user profiles through genre preferences  
✅ **End-to-End Training** – Single loss function optimizes entire pipeline  

---

## 🛠️ Technical Implementation

### Data Pipeline

**Preprocessing Steps:**
1. **Feature Extraction**
   - User features: ID, avg rating, rating count, genre preferences (14D)
   - Movie features: ID, year, genre flags (14D)
   
2. **Scaling**
   - StandardScaler (zero mean, unit variance) for user/item features
   - MinMaxScaler(-1, 1) for target ratings
   
3. **Train-Test Split**
   - 80/20 random split with seed=1 for reproducibility

### Model Training

```python
# Inputs
input_user = Input(shape=(num_user_features,))
input_item = Input(shape=(num_item_features,))

# User Tower
user_nn = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32)
])
vu = user_nn(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# Item Tower
item_nn = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32)
])
vm = item_nn(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# Dot Product
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# Compile
model = Model([input_user, input_item], output)
model.compile(optimizer='adam', loss='mse')
```

### Inference Workflows

**1. Recommend for New User**
```python
# Create user preference vector
user_vec = create_user_vector(genre_preferences)

# Replicate across all movies
user_vecs = gen_user_vecs(user_vec, num_movies)

# Predict ratings for all movies
predictions = model.predict([user_vecs, item_vecs])

# Sort and return top-N
top_movies = sort_by_prediction(predictions)
```

**2. Similar Movie Search**
```python
# Extract movie embeddings
movie_embeddings = item_tower.predict(all_movies)

# Compute pairwise distances
distances = pairwise_squared_distance(movie_embeddings)

# Find nearest neighbors
similar_movies = find_k_nearest(target_movie, k=10)
```

---

## 📈 Model Performance

### Embedding Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Embedding Dimension** | 32 | Compact yet expressive |
| **L2 Normalization** | Unit sphere | Enables dot-product similarity |
| **Tower Depth** | 3 layers | Balances capacity and overfitting |
| **Parameter Count** | ~100K | Efficient for deployment |

### Similarity Search Results

The model successfully clusters movies with **shared latent characteristics** rather than explicit genre matches:

**Example: Movies Similar to "The Matrix" (1999)**
- The Thirteenth Floor (1999) – *Shared: Reality simulation theme*
- Dark City (1998) – *Shared: Identity + dystopia*
- eXistenZ (1999) – *Shared: Virtual reality philosophy*

**Note:** These movies have different genre combinations but attract the same user profiles.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
scikit-learn
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/deepmoviematch.git
cd deepmoviematch

# Install dependencies
pip install numpy pandas tensorflow scikit-learn tabulate

# Launch notebook
jupyter notebook RecSys_TwoTower_With_Notes.ipynb
```

### Quick Start

```python
# Load data
item_train, user_train, y_train, item_features, user_features, \
    item_vecs, movie_dict, user_to_genre = load_data()

# Scale features
scalerUser = StandardScaler().fit(user_train)
scalerItem = StandardScaler().fit(item_train)
scalerTarget = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))

# Train model
model.fit([user_train, item_train], y_train, epochs=30, batch_size=256)

# Predict for new user
user_vec = create_custom_user(action=5, comedy=3, drama=4)
predictions = predict_top_movies(user_vec, item_vecs, model)
```

---

## 📂 Project Structure

```
deepmoviematch/
│
├── RecSys_TwoTower_With_Notes.ipynb  # Main implementation
├── recsysNN_utils.py                 # Helper functions
├── data/
│   ├── content_top10_df.csv          # Precomputed content-based recs
│   └── content_bygenre_df.csv        # Genre-specific baselines
│
├── README.md                         # This file
└── requirements.txt                  # Dependencies
```

---

## 🔬 Technical Deep Dive

### Why Two Towers?

**Decoupled Learning:** User and item encoders learn independently, then align in shared space.

**Computational Efficiency:** Precompute all movie embeddings once; score new users in O(n) time.

**Scalability:** Tower outputs can be cached and served from vector databases (Pinecone, Weaviate).

### Why L2 Normalization?

Constraining embeddings to the **unit hypersphere** ensures:
- Dot product = cosine similarity
- Gradient stability during training
- Interpretable distance metrics

### Mathematical Foundation

**Objective Function:**
```
minimize MSE: Σ (rating_actual - dot(user_emb, movie_emb))²
```

**Embedding Constraint:**
```
||user_emb||₂ = 1
||movie_emb||₂ = 1
```

**Similarity Metric:**
```
similarity = user_emb · movie_emb = Σ u_i * m_i
```

---

## 🎓 Skills Demonstrated

### Machine Learning
✅ Neural collaborative filtering architecture  
✅ Multi-tower neural networks  
✅ Embedding learning and representation  
✅ Feature scaling and normalization  
✅ Train-test split and validation

### Deep Learning Engineering
✅ TensorFlow/Keras functional API  
✅ Custom model architectures  
✅ L2 normalization layers  
✅ Adam optimization  
✅ MSE loss for regression

### Recommender Systems
✅ Collaborative filtering principles  
✅ Latent factor models  
✅ Cold-start problem handling  
✅ Similarity search algorithms  
✅ Top-N recommendation generation

### Software Engineering
✅ Modular utility functions  
✅ Scalable inference pipelines  
✅ Reproducible experiments (random seeds)  
✅ Code documentation and structure

---

## 💡 Business Applications

### E-Commerce
- **Product recommendations** based on implicit user behavior
- **Bundle suggestions** from embedding proximity
- **New product launch** targeting via user clusters

### Content Platforms
- **Netflix-style recommendations** with learned taste profiles
- **Playlist generation** for music/video services
- **Article suggestions** for news/media sites

### Finance
- **Asset similarity** for portfolio construction
- **Customer segmentation** via embedding clusters
- **Cross-sell recommendations** for financial products

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Attention mechanisms for feature importance
- [ ] Temporal dynamics (user preferences evolve)
- [ ] Multi-task learning (rating + click prediction)
- [ ] Triplet loss for better embedding separation

### Engineering
- [ ] Real-time inference API (FastAPI)
- [ ] Vector database integration (Pinecone)
- [ ] A/B testing framework
- [ ] Model monitoring and drift detection

### Features
- [ ] Explainability (SHAP for neural nets)
- [ ] Diversity-aware ranking
- [ ] Context-aware recommendations (time, device)
- [ ] Hybrid with content-based signals

---

## 📚 Utilities Reference

### `recsysNN_utils.py` Functions

| Function | Purpose |
|----------|---------|
| `load_data()` | Loads matrices, feature names, and metadata |
| `pprint_train(...)` | Visual sanity check for training slices |
| `gen_user_vecs(...)` | Replicates user row across all items for batch scoring |
| `print_pred_movies(...)` | Displays top-N predicted titles |
| `get_user_vecs(...)` | Constructs (user, item) pairs and labels for evaluation |
| `print_existing_user(...)` | Shows predicted vs. actual ratings side-by-side |

---

## 🏆 Why This Matters

> *"Traditional recommenders tell you what you asked for. Neural recommenders tell you what you didn't know you wanted."*

**DeepMovieMatch demonstrates:**
1. **Representation Learning** – The core skill behind modern AI (GPT, DALL-E, AlphaFold)
2. **Production ML Patterns** – Two-tower architecture powers Google, Meta, Netflix at scale
3. **End-to-End Thinking** – From raw features to business value (top-10 recommendations)

**For ML/Quant Roles:** Shows ability to:
- Design neural architectures for structured prediction
- Optimize embeddings for similarity tasks (relevant to asset clustering, anomaly detection)
- Build scalable inference pipelines
- Translate complex models into actionable insights

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** – Deep learning framework
- **scikit-learn** – Preprocessing utilities
- **MovieLens Dataset** – University of Minnesota research lab
- **Two-Tower Architecture** – Inspired by YouTube's recommendation system (Covington et al., 2016)

