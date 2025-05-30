# Artificial Intelligence in Next Generation Networks (5G and Beyond)

---

## 📚 About the Book

This repository contains source code, datasets, and simulations that accompany the book:

**Artificial Intelligence in Next Generation Networks (5G and Beyond): Fundamentals, Security and Applications**

The book provides in-depth discussion and practical insights into how AI techniques are applied to secure and optimize NextG (5G and beyond) wireless communication systems, including challenges posed by adversarial machine learning.

---

## 📂 Repository Structure

Each chapter in the book corresponds to a specific directory in the repository:

| Chapter | Title                                                                       | Focus                                                                |
| ------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 01      | Introduction to NextG Networks                                              | Overview of 5G/6G evolution, architecture, and enabling technologies |
| 02      | Understanding Adversarial Attacks and Mitigation Methods                    | Adversarial ML attacks (FGSM, BIM, PGD, etc.) and defense techniques |
| 03      | Mitigation Techniques in MIMO Beamforming                                   | Defense strategies for beamforming in adversarial settings           |
| 04      | Securing Channel Estimation                                                 | CNN-based estimation and robustness against attacks                  |
| 05      | AI-Driven Spectrum Sensing with Robust Adversarial Defenses                 | Semantic segmentation for spectrum detection and its resilience      |
| 06      | Adversarial Resilience in Intelligent Reflecting Surface (IRS) Technologies | IRS vulnerability analysis and mitigation using distillation         |
| 07      | Strengthening AI-Driven Automatic Modulation Recognition (AMR)              | Robust AMR models for MIMO scenarios under adversarial pressure      |
| 08      | Opportunities, Challenges and Future Directions                             | Vision, open problems, and strategic recommendations for NextG       |

Each directory includes:

* Jupyter Notebooks or `.py` scripts
* Explanatory comments and figures
* Sample datasets or links to external datasets
* Pretrained model weights (if available)

---

## 🚀 Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ocatak/NextG-Communications-with-AI.git
   cd NextG-Communications-with-AI
   ```

2. **Set up Python environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Navigate into a chapter and run examples:**

   ```bash
   cd Chapter-02
   jupyter notebook
   ```

---

## 📌 Requirements

* Python 3.8+
* TensorFlow / PyTorch (based on the chapter)
* scikit-learn, matplotlib, seaborn
* Adversarial robustness tools (e.g., `cleverhans`, `foolbox`)
* See each chapter’s `requirements.txt` for specifics

---

## ✨ Highlights

* 🔒 Defense techniques like adversarial training and defensive distillation
* 📶 Applications in MIMO, beamforming, channel estimation, and AMR
* 🎯 Comparison of adversarial robustness across different models and scenarios
* 📊 Experimental evaluation and performance visualization

---

## 👥 Authors

* **Murat Kuzlu** – Old Dominion University, USA
* **Ferhat Ozgur Catak** – University of Stavanger, Norway
* **Yanxiao Zhao** – Virginia Commonwealth University, USA
* **Gokcen Ozdemir** – Erciyes University, Turkey / Old Dominion University, USA

---

## 📄 License

MIT License – see [LICENSE](./LICENSE) for details.

---

## 📬 Contact

For inquiries, contributions, or issues, please contact
[📧 Ferhat Ozgur Catak](mailto:ocatak@gmail.com)
Or open an [issue](https://github.com/ocatak/NextG-Communications-with-AI/issues) in this repository.
