# Deep Learning based surrogate model of severe accidents in nuclear reactors (from the ASTEC simulator)

Simulations of severe accidents in nuclear reactors can take up to months to run, making their use unfeasible to train nuclear operators to react correctly in order to prevent catastrophic events like core melting. In this project we construct a surrogate model of the ASTEC simulator [2], based on the data-driven methodology developed in [3]. More details about the actual ASTEC application of [3] are given in this paper [4].
This project has been developed within the European project ASSAS [1].

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

We frame the problem as a general physical system desribed by a PDE, as we explain in [4]. We use an Auto-Encoder to compress the 1996 degrees of freedom (around 100 variables, both scalars and fields) into a reduced representation. The dynamics of the reduced representation is approximated by a NODE.

- **Task:** Approximation of the dynamics of about 100 physical variables over time at the variation of the time of activation of 10 different operator actions.
- **Dataset:** Simulations from ASTEC [2], generated within the ASSAS project [1]. 
- **Framework:** PyTorch.

---

## Installation

```bash
git clone https://github.com/Aleartulon/ASTEC_surrogate_model/tree/main
cd repo-name
pip install -r requirements.txt
```


---

## Usage

**First dataset generation**
```bash
python -u -m src.dataset_generation.dataset.main
```

**Train**
```bash
python -u -m src.models.AE_NODE.training.main
```

**Test**
```bash
python -u -m src.models.AE_NODE.testing.main
```

---

## Project Structure

```
├── configs/          # Training configuration files
├── data/             # Data loading and preprocessing
├── models/           # Model architectures
├── checkpoints/      # Saved model weights
├── notebooks/        # Exploratory analysis
├── train.py
├── evaluate.py
├── inference.py
└── requirements.txt
```

---

## Results

| Model       | Dataset  | Accuracy | F1 Score |
|-------------|----------|----------|----------|
| Baseline    | CIFAR-10 | 91.2%    | 0.911    |
| Fine-tuned  | CIFAR-10 | 94.7%    | 0.946    |

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

## References

[1] ASSAS Consortium. *ASSAS -- Artificial Intelligence for Simulation of Severe Accidents*. Horizon Europe Project, coordinated by ASNR, 2023–2026. https://assas-horizon-euratom.eu

[2] Chailan, L., Bosland, L., Carénini, L., Chambarel, J., Cousin, F., et al. *Overview of ASTEC Integral Code Status and Perspectives*. 9th European Review Meeting on Severe Accident Research (ERMSAR2019), Prague, Czech Republic, March 2019. DOI: irsn-04106726

[3] Longhi, A., Lathouwers, D., & Perkó, Z. *Latent space modeling of parametric and time-dependent PDEs using neural ODEs*. Computer Methods in Applied Mechanics and Engineering, 448, 118394, January 2026. https://doi.org/10.1016/j.cma.2025.118394