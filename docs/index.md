# Welcome to FairMango

---

**FairMango** is a comprehensive library designed to facilitate **explainable** fairness in machine learning (ML) and artificial intelligence (AI) models. Our goal is to provide tools and methodologies that enable practitioners to measure, assess, and understand biases in their models, ensuring fair and transparent AI systems.

---

## Sections

The documentation follows the best practice for project documentation and consists of four separate parts:

1. [Concepts](concepts.md)
2. [API Documentation](api_documentation/dataset.md)
3. [Tutorials](tutorials.md)
4. [How-To Guides](how-to-guides.md)

---

## Key Features

- **Data Handling 📊:** Separate your data using multiple sensitive features and handle each sensitive group with ease.
- **Performance Metrics 🔍:** Evaluate the model performance on different sensitive groups using many metrics.
- **Fairness Metrics 📏:** Implement and evaluate various state of art fairness metrics.
- **Explainability 💬:** Every result is explainable and can be put into easy to understand words.

---

## Getting Started

### Installation

```bash
pip install fair-mango
```

### Quick Start Guide:

```python
import pandas as pd
from fair_mango.metrics.metrics import DemographicParityDifference

data = {
    'sensitive_1': ['male', 'female', 'female', 'male', 'male', 'male'],
    'sensitive_2': ['white', 'black', 'black', 'black', 'black', 'white'],
    'real_target': [1, 0, 0, 1, 0, 1],
    'predicted_target': [0, 1, 1, 0, 0, 1],
}

df = pd.DataFrame(data)

demographic_parity_diff = DemographicParityDifference(
    data=df,
    sensitive=['sensitive_1', 'sensitive_2'],
    real_target=['real_target'],
    predicted_target=['predicted_target']
)
summary = demographic_parity_diff.summary()
print(summary)

ranking = demographic_parity_diff.rank()
print(ranking)

is_biased = demographic_parity_diff.is_biased(0.2)
print(is_biased)
```

More detailed documentation and tutorials are available to help you integrate FairMango into your workflow. Visit our [reference](reference.md) and [Tutorials](tutorials.md) pages to learn more.

---

## Community and Support

Join our community to stay updated on the latest developments, share your experiences, and get support:

GitHub: [FairMango GitHub Repository](https://github.com/datategy/Fair-Mango)<br>
Contact Us: [Support Email](mailto:contact@datategy.net)

---

## License

FairMango is licensed under the [Apache License](https://github.com/datategy/Fair-Mango/blob/dev/LICENSE)

---

***FairMango:** Striving for fairness and transparency in machine learning and AI.*
