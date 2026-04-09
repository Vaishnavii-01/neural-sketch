# 🎨 Neural Sketch : Real-Time Symbolic Recognition

**Neural Sketch** is a high-performance deep learning application developed as an **MLDL Mini Project (2026)**. It bridges the gap between human creativity and machine intelligence by classifying hand-drawn sketches into 15 distinct categories of fruits and vegetables in real-time.

---

## 🚀 Key Features
* **Deep CNN Architecture :** A custom-tuned Convolutional Neural Network optimized for spatial feature extraction (edges, curves, and junctions) from binary bitmaps.
* **Real-Time Inference :** Achieves near-instantaneous classification ($<200ms$) through a decoupled **Flask-Electron** architecture.
* **Out-of-Distribution (OOD) Handling :** Implements a **45% Confidence Threshold** to distinguish between valid drawings and unrecognizable or invalid inputs.
* **Interactive Desktop UI :** Featuring dynamic stroke-size control and real-time probability visualizations using CanvasJS for an intuitive user experience.

---

## 🛠️ Technical Stack
| Component | Technology |
| :--- | :--- |
| **Deep Learning** | TensorFlow 2.16, Keras |
| **Logic & Inference** | Python 3.12 |
| **Backend/API Bridge** | Flask (RESTful) |
| **Frontend/Desktop** | Electron.js, JavaScript (ES6+), HTML5 Canvas |
| **Data Science** | NumPy, OpenCV, Scikit-Learn, Matplotlib, Seaborn |

---

## 📊 Dataset & Model Details
The system is trained on the **Google "Quick, Draw!" Open Dataset**.
* **Classes :** 15 Fruits & Vegetables (Apple, Banana, Broccoli, Carrot, Pineapple, etc.).
* **Training Samples :** 75,000 unique sketches ($5,000$ samples per class).
* **Preprocessing Pipeline :**
    * Grayscale normalization ($0$ to $1$) for gradient stability.
    * $28 \times 28$ tensor reshaping.
    * Bitwise inversion for white-on-black consistency required by CNN kernels.
* **Regularization :** Dropout (0.4) implemented to mitigate overfitting and improve model generalization on messy user inputs.


---

## 🔮 Future Scope
* **Temporal Sequence Logic :** Integrating **RNN/LSTM** layers to recognize drawings based on the *sequence* of strokes rather than static pixels.
* **Generative AI :** Implementing **GANs** (Generative Adversarial Networks) to transform rough user doodles into high-fidelity, photorealistic assets.
* **Mobile Deployment :** Compiling the inference engine to **TensorFlow Lite** for native Android and iOS applications.

---

## 🎓 Author
* **Vaishnavi Eknath Avhad**
* **Roll No :** 14 | IT Engineering
* **Institute :** Vivekanand Education Society's Institute Of Technology (VESIT)
* **Date :** April 2026
