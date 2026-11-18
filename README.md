# BPSN

**A Soft Sensor Net Based on the Symplectic Decomposition–Global Attention Reconstruction Architecture for Biopharmaceutical Industry**

Applied Soft Computing · 2025

[Project page](https://joss0623.github.io/projects/bpsn/) · [Research homepage](https://joss0623.github.io/)

[Paper](https://doi.org/10.1016/j.asoc.2025.113636)

[Publisher / DOI](https://doi.org/10.1016/j.asoc.2025.113636) · [BibTeX](https://joss0623.github.io/citations/bpsn.bib)

## Overview

Combine structured signal decomposition, forecasting, and global-attention reconstruction for soft sensing in biopharmaceutical processes.

## Research focus

- Signal decomposition
- Global attention
- Soft sensing

## Model

<p align="center">
  <img src="./pic/overall_model.png" alt="Overall Model" style="width:80%; max-width:100vw; height:auto;" />
</p>

## Usage

1. **Install Python 3.8**. For convenience, execute the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Decompose Data.**  
   The decomposition code is under the folder `./decompose/`. After running, you will obtain many SGCs.

3. **Train and evaluate model.**  
   Experiment scripts for all benchmarks are provided under the folder `./scripts/`.  
   You can implement univariate prediction based on raw data for a priori knowledge and multivariate sub-sequences prediction based on SGCs.

4. **Series reconstruction.**  
   The reconstruction codes for different scenarios are under the folder `./restructure/`.

5. **Develop your own model:**

    - Add the model file to the folder `./models`. You can follow the example `./models/Transformer.py`.
    - Include the newly added model in the `Exp_Basic.model_dict` of `./exp/exp_basic.py`.
    - Create the corresponding scripts under the folder `./scripts/`.


