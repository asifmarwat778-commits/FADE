<p align="center">
  <img width="681" height="143" alt="Screenshot 2026-04-14 at 6 03 34 PM" src="https://github.com/user-attachments/assets/c8c4cf39-afb0-4c2b-978f-70b26b47a487" />


# # RatePred
RatePred is a machine learning GUI for predicting the observed pseudo-first-order rate constant (*k*ₒbₛ) from operational inputs in UV/chlor(am)ine/Fe(III) advanced oxidation processes.

## Using RatePred
The tool is available from the script `gui.py`. It is intended to be run locally via Python. The user can select the target AOP system (UV/HOCl/Fe(III) or UV/NH2Cl/Fe(III)), enter individual operational parameters or batch inputs, and the GUI will output the predicted kobs along with SHAP-based feature importance and one-at-a-time sensitivity trajectories for the entered scenario

## Training Dataset
The complete datasets of experimentally measured *k*ₒbₛ values used to train RatePred are given in the files `Data_HOCl.csv` (304 data points) and `Data_NH2Cl.csv` (863 data points).

## Citation
Muhammad Asif, Wei Wang, Aiwen Wang, Hidayat Ullah Khan. "Pseudo-first-order reaction rate prediction using machine learning: RatePred." *Water Research* 2026, XX, XXXX-XXXX. DOI: [10.1021/acs.est.5c16184](https://doi.org/10.1021/acs.est.5c16184)

## Acknowledgments
**Funding:** the National Key Research and Development Program of China (2023YFC3207104, W.W.), and the National Engineering Research Center for Safe Disposal and Resources Recovery of Sludge (K2024A011).  
**AI Use:** Artificial intelligence model Kimi was used as a coding aid in the development of RatePred.

## Contact
Muhammad Asif, Harbin Institute of Technology
0250366@hit.edu.cn
