# 🗣️ Sign Language Analysis – Reconstructive & Contrastive Latent Spaces

This project implements an **Intelligent Agent** designed to analyze and interpret **Sign Language** words through a novel **Reconstructive and Contrastive Deep Learning technique**.

Designed for scenarios with **Limited Data and Resources**, the model leverages a hybrid architecture to learn a continuous **Latent Space**, where signs with the same semantic meaning are clustered together, and different signs are separated. This approach overcomes the limitations of traditional classification in low-resource environments.

> 📄 **Project Presentation:** For a detailed overview of the research, methodology, and results, check the [**Final Project Poster**](RepresentatioHorarum/Poster%20Finalv04.pdf).

---

## 🤖 Introduction to the Intelligent Agent

**Artificial Intelligence (AI)** enables systems to perceive complex patterns and reason about them. In this project, the **Sign Language Analysis System** is modeled as an **Intelligent Agent** that processes visual sequences of signs:

- **Perception** → The agent receives sign language data (video sequences or keypoints) and processes them through a neural encoder.
- **Reasoning** → It projects these inputs into a **Latent Space** optimized by two complementary forces:
    1.  **Reconstruction**: Ensuring the essential features of the sign are preserved.
    2.  **Contrast**: Ensuring signs of the same class are geometrically close, while different signs are pushed apart.
- **Representation** → The agent builds a robust semantic map of signs, allowing for accurate recognition even with few training examples.

This structure solves the problem of high-dimensional video data by reducing it to a meaningful, compact vector representation.

---

## 🚀 Features

- 🤟 **Sign Language Recognition**: Specialized in analyzing isolated words in Sign Language.
- 🧠 **Hybrid Loss Architecture**: Combines **Reconstruction Loss** (Autoencoder style) with **Contrastive/Triplet Loss** for superior embedding quality.
- 📉 **Limited Data Optimized**: Specifically engineered to perform well without massive datasets.
- ⏱️ **Temporal Dynamics Validation**: Robustness tested against temporal distortions (Inverted, Permuted, Shifted).
- 📊 **Semantic Separation Metric**: Implements a custom metric to measure the ratio between inter-class distance and intra-class compactness.
- 🗺️ **Latent Space Visualization**: Uses **UMAP** and **PCA** to visualize how the model clusters signs in 2D/3D space.
- 📄 **Academic Rigor**: Is the undergraduate thesis *"Desarrollo de Técnica Reconstructiva y Contrastiva para el análisis de Palabras en Lenguajes de Señas"* for the program *"Ciencias de la Computacion e Inteligencia Artificial"* from *"Universidad Sergio Arboleda"*.

---

## 💻 Code Workflow

The repository is organized to handle the pipeline from data ingestion to latent space evaluation.

### **Data & Preprocessing**
- **Data Loading**: Scripts to ingest sign language datasets (In this case WLSL_v03, ISL, SLOVO).
- **Feature Engineering**: Normalization and formatting of input sequences for the model.

### **Model Architecture (`Modelo1Idioma.ipynb`)**
- **Encoder-Decoder**: A core structure that learns to compress and reconstruct the input data.
- **Projection Head**: Maps the encoded features to the metric space for contrastive learning.

### **Training & Evaluation**
- **Hybrid Training Loop**: Optimizes weights using a weighted sum of Reconstruction Error and Contrastive/Triplet Loss.
- **Metrics**: Tracks the **Semantic Separation Ratio** to validate the quality of the clusters.
- **Visualization**: Generates UMAP plots to visually confirm the separation of different sign classes.

---

## 🧩 How It Works

1.  **Input**: The system accepts a sequence representing a sign (word).
2.  **Encoding**: The neural network compresses this high-dimensional input into a low-dimensional feature vector (embedding).
3.  **Dual Optimization**:
    - The **Reconstruction** branch tries to recreate the original input from this vector (ensuring data integrity).
    - The **Contrastive** branch compares this vector with others (Anchor, Positive, Negative) to adjust its position in space.
4.  **Latent Space Formation**: Over time, the model organizes the space so that all instances of the word "Hello" are close together, and far from "Goodbye".

### **Model Architecture Summary**
The following diagram illustrates the hybrid architecture and how data flows through the reconstructive and contrastive blocks:

![Model Summary](Representatio_Horarum/Images/Imagenes%20Cap%202/ResumenMod.PNG)

---

## 🧠 Algorithm Summary

### **Triplet & Contrastive Mechanism**
A key component of this project is the use of **Metric Learning**. Instead of just memorizing labels, the model learns the *concept* of similarity.

The training relies on **Triplets**:
- **Anchor (A)**: A reference instance of a specific sign.
- **Positive (P)**: Another instance of the *same* sign (e.g., same word, different signer).
- **Negative (N)**: An instance of a *different* sign.

**The Goal:**
The network minimizes the distance $d(A, P)$ while maximizing $d(A, N)$ by a margin $\alpha$. This creates the semantic clusters shown below:

![Triplet Loss Diagram](Representatio_Horarum/Images/Imagenes%20Cap%202/GrafTriplet.png)

### **Reconstruction Support**
Simultaneously, the model minimizes the **Reconstruction Error** ($L_{recon}$). This acts as a regularizer, preventing the model from "cheating" the contrastive task by collapsing the latent space into meaningless points.

### **Temporal Learning Verification**
To confirm that the model understands the **temporal order** of signs (and doesn't just treat them as static images), the architecture was validated using specific variations of the input sequences:

1.  **Shifted (Corrida)**: A slight temporal displacement of the original video. The model correctly identifies this as **semantically closest** to the original.
2.  **Inverted (Invertida)**: The video sequence played backwards.
3.  **Permuted (Permutada)**: The frames of the video shuffled randomly. The model identifies this as the **farthest** distance, proving that the **order of frames matters** for the learned representation.

This hierarchy of distances ($d_{shifted} < d_{inverted} < d_{permuted}$) confirms the agent effectively learns temporal dynamics.

---

## 🧪 Conclusions

Based on the experimental results documented in the final report:

1.  **Temporal Awareness**: The model successfully encodes temporal dynamics. It differentiates between original sequences and their permuted/inverted variants, confirming it understands the "flow" of a sign, not just individual hand shapes.
2.  **Superiority over Baseline**: The proposed Reconstructive-Contrastive technique consistently outperforms linear dimensionality reduction methods (like PCA) in separating semantic classes.
3.  **Viability for Limited Data**: The hybrid loss approach allows the model to construct structured latent spaces even with small datasets, making it suitable for low-resource environments.
4.  **Data Sensitivity**: The quality of the latent space is heavily dependent on dataset consistency (camera angles, resolution). Future work suggests integrating attention mechanisms to further improve robustness against noise.

---

## 🔧 Hardware & Software Requirements

- **Language**: Python 3.10+
- **Deep Learning Framework**: TensorFlow / Keras
- **Libraries**:
    - `umap-learn` (Dimensionality reduction for visualization)
    - `scikit-learn` (PCA and metrics)
    - `numpy`, `pandas` (Data manipulation)
    - `matplotlib` (Plotting results)
- **Hardware**:
    - GPU support is highly recommended for training the Encoder-Decoder architecture.

---

## 👨‍💻 Author

Developed by **Daniel Camilo Bernal Ternera**.
*Universidad Sergio Arboleda – School of Exact Sciences and Engineering.*

For detailed inquiries or collaboration, please refer to the repository issues section.
