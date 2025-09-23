# STRIDE: A Novel Kernel-Based Framework for Explainable AI

**Submitted to ICLR 2026 (Under Review) & Patent-Pending (App. No. 10-2025-0044029)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

Welcome to the official overview of the STRIDE XAI framework. STRIDE was invented and developed by **Chaeyun Ko** as part of a Master's thesis at Ewha Womans University.

[**My LinkedIn Profile**](https://www.linkedin.com/in/chaeyunko/?locale=en_US) | [**Google Scholar Profile**](https://scholar.google.com/citations?user=z2gnrNUAAAAJ&hl=en&authuser=1) 

---

## 💡 Core Idea & Vision

Confronted with the 2^d complexity wall in XAI, I challenged the established view that enumeration was necessary. STRIDE is the answer to a fundamental question—"Can we decompose a model's logic without brute force?"—and embodies my philosophy of turning deep theoretical inquiry into tangible, real-world impact.

- **New Paradigm**: Moves beyond combinatorial methods by using **Orthogonal Decomposition in a Reproducing Kernel Hilbert Space (RKHS)**.
- **Key Innovation**: Analytically separates a model's output into main effects and high-order interaction effects, revealing **how** features interact, not just **what** features are important.

---

## 🚀 Key Results & Capabilities

- **Performance**: Achieved **up to 9.7x speedup** over TreeSHAP on benchmark datasets. 
- **Fidelity**: Reconstructs original model predictions with **high fidelity (avg. R² > 0.93)**.
- **Validation**: Validated through a **first-author ICLR 2026 submission**, a **pending patent**, and a **Letter of Intent (LOI)** from a leading fintech CTO.

---

## 🛠️ Technical Stack & Source Code

- **Languages**: Python (Proficient), MATLAB
- **Libraries**: Scikit-learn, PyTorch, XGBoost, Pandas, NumPy, Matplotlib

The core algorithms of STRIDE are proprietary and patent-pending. This repository serves as a portfolio demonstrating the project's methodology, results, and the engineering quality of non-proprietary components (e.g., visualization, benchmarking). The full source code is maintained privately, and I would be happy to discuss the architecture and implementation details in an interview.
