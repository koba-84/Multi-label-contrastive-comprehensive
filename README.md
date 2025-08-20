# Multi-Label Contrastive Learning: A Comprehensive Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


Official implementation of the papers:
- **[Exploring Contrastive Learning for Long-Tailed Multi-Label Text Classification](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=W8aMnu8AAAAJ&citation_for_view=W8aMnu8AAAAJ:u5HHmVD_uO8C)**
- **[Multi-Label Contrastive Learning: A Comprehensive Study](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=W8aMnu8AAAAJ&citation_for_view=W8aMnu8AAAAJ:2osOgNQ5qMEC)**


## 🎯 Overview

This repository provides a comprehensive implementation of contrastive learning approaches for multi-label classification tasks. Our work explores the effectiveness of contrastive learning across various domains including computer vision and natural language processing, with a focus on handling long-tailed label distributions.

## ✨ Key Contributions

- **Comprehensive Analysis**: In-depth evaluation of contrastive learning loss functions in multi-label classification scenarios
- **Multi-Domain Evaluation**: Performance analysis across datasets with varying label counts and training data volumes
- **Cross-Modal Insights**: Comparative study between computer vision and natural language processing domains
- **Novel Loss Function**: Development of an improved loss function specifically designed for multi-label classification performance

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for vision experiments)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/audibert-alexandre-fra/Multi-label-contrastive-comprehensive
   cd multi-label-contrastive-comprehensive
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

### Weights & Biases Setup

Our experiments use [Weights & Biases](https://wandb.ai/) for experiment tracking:

1. Create an account at [wandb.ai](https://wandb.ai/)
2. Get your API key from your profile settings
3. Configure credentials in `main.py`:

```python
import os

# Replace with your credentials
os.environ['WANDB_API_KEY'] = 'your_wandb_api_key'
ENTITY_NAME_WANDB = 'your_wandb_username'
```

## 🧪 Running Experiments

### Natural Language Processing

```bash
cd src
python3 main.py --config config/config_nlp.json
```

### Computer Vision

```bash
cd src
python3 main.py --config config/config_vision.json
```

### Non-Contrastive Baseline (BCE Loss)

```bash
cd src
python3 main.py --config config/config_vision_BCE.json
```

## 📊 Results & Reproducibility

Detailed experimental results and configurations are provided in the respective config files. For exact reproduction, specific dataset access may be required.

## 📚 Citation

If you find this work useful in your research, please cite our papers:

```bibtex
@misc{audibert2025multilabelcontrastivelearning,
      title={Multi-Label Contrastive Learning : A Comprehensive Study}, 
      author={Alexandre Audibert and Aurélien Gauffre and Massih-Reza Amini},
      year={2025},
      eprint={2412.00101},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.00101}, 
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research implementation. For production use, additional testing and validation may be required.
