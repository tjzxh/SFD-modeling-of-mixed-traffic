# 🚗 Stochastic Fundamental Diagram Modeling of Mixed Traffic Flow

[![Paper](https://img.shields.io/badge/Paper-Transportation%20Research%20Part%20C-purple)](https://doi.org/your-doi-link)
[![Code Status](https://img.shields.io/badge/Status-Official%20Code-blue)]()

> 🔍 **Official implementation of the paper:**  
> **Stochastic Fundamental Diagram Modeling of Mixed Traffic Flow: A Data-Driven Approach**  
> Published in *Transportation Research Part C*  

## 🔗 Abstract | 摘要

The integration of automated vehicles (AVs) into existing traffic of human-driven vehicles (HVs) poses significant challenges in modeling and optimizing mixed traffic flow. Existing research often neglects the stochastic nature of traffic flow that is further complicated by AVs, and relies on oversimplified assumptions or specific car-following models. Moreover, the under-utilization of empirical AV datasets undermines realism.

This paper proposes a **novel data-driven framework** to model the **Stochastic Fundamental Diagram (SFD)** of mixed traffic. We:

- Learn CF behavior of all leader-follower pairs (HV-AV, HV-HV, AV-HV, AV-AV) via **Mixture Density Network (MDN)**.
- Model the platoon as a **joint distribution using Markov chains**, allowing stochastic behavior aggregation.
- Validate the model on the **NGSIM I-80 dataset** and apply it to the **Waymo dataset** for real-world AV impact analysis.

Results show that higher AV penetration reduces capacity mean and variance due to conservative but stable AV behavior.

> 本文提出一种**数据驱动建模框架**，用于模拟**混合交通流的随机基本图（SFD）**。核心创新包括：
> - 基于**混合密度网络（MDN）**学习各类跟驰对（HV-AV, HV-HV, AV-HV, AV-AV）的微观行为；
> - 利用**马尔可夫链建模**构建车队联合分布，并推导宏观流量关系；
> - 在**NGSIM I-80**和**混合交通流仿真**上进行验证。
>
> 基于Waymo数据集进行案例研究发现，随着AV渗透率上升，混合交通流的随机性下降（通行能力标准差降低），系统可靠性与运行平稳性提升。然而，交通效率却随之下降（通行能力期望与关键密度降低），印证了已有实证研究中AV稳定但保守的行为特性。


## 📂 Project Structure | 项目结构

```bash
├── data/                     # Processed and trajectory datasets
│   └── waymo/                # Waymo dataset (processed AV-AV, HV-AV and HV-HV car-following data)
├── Micro_MDN/                      # Core codebase
│   ├── trained_model/                # trained MDN model for three types of pair
│   ├── MDN.py/                # Training scripts for MDN
│   ├── inference/            # Platoon simulation and SFD derivation
│   └── utils/                # Utilities and preprocessing
├── notebooks/                # Jupyter notebooks for analysis and visualization
├── figures/                  # Output figures used in the paper
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file
