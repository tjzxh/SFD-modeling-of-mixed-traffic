# 🚗 Stochastic Fundamental Diagram Modeling of Mixed Traffic Flow

[![Paper](https://img.shields.io/badge/Paper-Transportation%20Research%20Part%20C-purple)](https://doi.org/10.1016/j.trc.2025.105279)
[![Code Status](https://img.shields.io/badge/Status-Official%20Code-blue)]()

> 🔍 **Official implementation of the paper:**  
> **Stochastic Fundamental Diagram Modeling of Mixed Traffic Flow: A Data-Driven Approach**  
> Published in *Transportation Research Part C*  

<p align="center">
  <img src="/Mixed_SFD_framework_eng.svg" width="100%" />
</p>

## 🔗 Abstract | 摘要

The integration of automated vehicles (AVs) into existing traffic of human-driven vehicles (HVs) poses significant challenges in modeling and optimizing mixed traffic flow. Existing research often neglects the stochastic nature of traffic flow that is further complicated by AVs, and relies on oversimplified assumptions or specific car-following models. Moreover, the under-utilization of empirical AV datasets undermines realism.

This paper proposes a **novel data-driven framework** to model the **Stochastic Fundamental Diagram (SFD)** of mixed traffic. We:

- Learn CF behavior of all leader-follower pairs (HV-AV, HV-HV, AV-HV, AV-AV) via **Mixture Density Network (MDN)**.
- Model the platoon as a **joint distribution using Markov chains**, allowing stochastic behavior aggregation.
- Validate the model on the **NGSIM I-80 dataset** and apply it to the **Waymo dataset** for real-world AV impact analysis.

Results show that higher AV penetration reduces capacity mean and variance due to conservative but stable AV behavior.

> 本文提出一种**数据驱动建模框架**，用于模拟**混合交通流的随机基本图（SFD）**。核心工作包括：
> - 基于**混合密度网络（MDN）** 学习各类跟驰对（HV-AV, HV-HV, AV-HV, AV-AV）的微观行为；
> - 利用**马尔可夫链建模**构建车队联合分布，并推导宏观流量关系；
> - 在**NGSIM I-80** 和**混合交通流仿真**上进行验证。
>
> 基于**Waymo数据集**进行案例研究发现，随着AV渗透率上升，混合交通流的随机性下降（通行能力标准差降低），系统可靠性与运行平稳性提升。然而，交通效率却随之下降（通行能力期望与关键密度降低），印证了已有实证研究中AV稳定但保守的行为特性。

<p align="center">
  <img src="/Mixed_SFD_framework_zh.svg" width="100%" />
</p>

## 📂 Project Structure | 项目结构

```bash
├── Macro_SFD/
│   ├── equilibrium_state_calculate.py       # Compute equilibrium states
│   ├── leader_follower_conditional_distribution.py # Compute probabilistic leader-follower model
│   ├── platoon_arrangment.py                # Assemble vehicle platoons as Markov chains
│   ├── SFD_plot.py                          # Visualization of the resulting SFD
│   ├── smooth_fd_analysis.py                # Analyze and smooth derived FD/SFD distributions
│
├── Micro_MDN/
│   ├── trained_model/                       # Trained MDN models
│   ├── MDN.py                               # Mixture Density Network architecture and training
│   ├── block_tanh.py                        # Function for MDN
│   ├── MDN_single_validate.py               # Validate MDN on testing trajectories
│   └── data/
│       └── waymo/                           # Processed Waymo trajectories of AV-AV, AV-HV and HV-HV
│
└── README.md
```

## 📄 Citation | 文献引用
Zhang, X., Yang, K., Sun, J., & Sun, J. (2025).
Stochastic fundamental diagram modeling of mixed traffic flow: A data-driven approach.
Transportation Research Part C: Emerging Technologies, 179, 105279.
https://doi.org/10.1016/j.trc.2025.105279
