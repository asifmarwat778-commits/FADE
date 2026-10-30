<p align="center">
  <img width="732" height="261" alt="Image" src="https://github.com/user-attachments/assets/22d5de75-a283-4791-be50-18f43ecf8df8" />


# FADE

**FADE (First-order AOP Degradation Estimator)** is an open-source machine-learning framework with a graphical user interface (GUI) for predicting observed pseudo-first-order degradation rate constants ($k_{\mathrm{obs}}$) from operational and water-matrix inputs in UV/chlor(am)ine/Fe(III) advanced oxidation processes (AOPs).

FADE supports two treatment systems:

- UV/HOCl/Fe(III)
- UV/NH₂Cl/Fe(III)

## Features

- Individual and batch prediction of $k_{\mathrm{obs}}$
- SHAP-based model interpretation
- One-at-a-time sensitivity analysis
- Experimental and literature-derived training datasets
- Dedicated GUI implementations for both treatment systems

## Using FADE

FADE runs locally in Python. The following GUI implementations are provided:

- `FADE_GUI_DTR_UV_HOCl_Fe(III).py` for UV/HOCl/Fe(III)
- `FADE_GUI_XGBR_UV_NH2Cl_Fe(III).ipynb` for UV/NH₂Cl/Fe(III)

Users can enter individual operational parameters or provide batch inputs to obtain predicted $k_{\mathrm{obs}}$ values, SHAP-based feature interpretations, and sensitivity trajectories for the selected scenario.

## Training Datasets

The repository provides the datasets used for model training and evaluation:

- `Experimental Data of ONP degradation during UV HOCl Fe(III) for ML models training.csv`: 391 experimental observations for ortho-nitrophenol degradation in the UV/HOCl/Fe(III) system.
- `Experimental Data of ONP degradation during UV NH2Cl Fe(III) for ML models training.csv`: experimental observations for ortho-nitrophenol degradation in the UV/NH₂Cl/Fe(III) system.
- `Experimetnal data from Litrature for Machine Learning Training.csv`: 862 literature-derived observations covering 51 micropollutants reported in 11 studies.

The datasets include operational and water-matrix input variables together with experimental or literature-derived $k_{\mathrm{obs}}$ values.

## GUI Demonstrations

Video demonstrations are provided for the UV/HOCl/Fe(III) and UV/NH₂Cl/Fe(III) interfaces, together with a combined presentation illustrating GUI operation for both treatment systems.

## Citation

If you use FADE in your research, please cite the associated manuscript. Full citation details will be added following publication.

## Acknowledgments

**Funding:** This work was supported by the National Key Research and Development Program of China (2023YFC3207104, W.W.) and the National Engineering Research Center for Safe Disposal and Resources Recovery of Sludge (K2024A011).

**AI Use:** OpenAI ChatGPT and Kimi were used as coding aids during the development of FADE.

## Contact

**Muhammad Asif**  
Harbin Institute of Technology  
Email: [20250366@hit.edu.cn](mailto:20250366@hit.edu.cn)
