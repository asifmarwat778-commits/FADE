<p align="center">
  <img width="732" height="261" alt="Image" src="https://github.com/user-attachments/assets/22d5de75-a283-4791-be50-18f43ecf8df8" />


# # FADE
FADE (First-order AOP Degradation Estimator) is a machine learning GUI for predicting the observed pseudo-first-order rate constant (kobs) from operational inputs in UV/chlor(am)ine/Fe(III) advanced oxidation processes.

## Using FADE
The tool is available from the script `gui.py`. It is intended to be run locally via Python. The user can select the target AOP system (UV/HOCl/Fe(III) or UV/NH2Cl/Fe(III)), enter individual operational parameters or batch inputs, and the GUI will output the predicted kobs along with SHAP-based feature importance and one-at-a-time sensitivity trajectories for the entered scenario

## Training Dataset
The complete datasets of experimentally measured kobs values used to train FADE are given in the files `Data2026042605 ONP HOCl.csv` (391 data points) and `Data_NH2Cl.csv` (863 data points). The GUI code is given in the file attached here named FADE_GUI_XGBR_UV_NH2Cl_Fe(III).ipynb and 
FADE_GUI_DTR_UV_HOCl_Fe(III).py of UV/NH2Cl/Fe(III) and UV/HOCl/Fe(III) respectively. The GUI working clips are recorded during UV/NH2Cl/Fe(III), UV/HOCl/Fe(III), and the combined process, and are shown as three video clips. The attachment names of the GUI are; GUI performance during UV-NH2Cl-Fe(III), GUI presentation during UV_HOCL_Fe(III), and GUI combined presentation during UV_HOCL-Fe(III) and UV_NH2Cl_Fe(III). 
## Citation
Muhammad Asif, Wei Wang, Aiwen Wang, Hidayat Ullah Khan. "Pseudo-first-order reaction rate prediction using machine learning: FADE." *journal name* 2026, XX, XXXX-XXXX. DOI: [XX, XXXX-XXXX]

## Acknowledgments
**Funding:** the National Key Research and Development Program of China (2023YFC3207104, W.W.), and the National Engineering Research Center for Safe Disposal and Resources Recovery of Sludge (K2024A011).  
**AI Use:** Artificial intelligence models OpenAI ChatGPT and Kimi were used as a coding aid in the development of RatePred.

## Contact
Muhammad Asif, Harbin Institute of Technology
20250366@hit.edu.cn
