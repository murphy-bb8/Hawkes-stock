---
layout: default
title: 多维 Hawkes 过程建模
---

# 多维 Hawkes 过程建模：加性基线的 EM 算法实现

## 项目简介

本项目围绕 **A 股限价订单簿中的有毒/无毒订单流**，构建 **4 维多元 Hawkes 过程** 模型，使用 **加性基线** 的 EM 算法进行高效估计。相比传统的 log-link 乘性基线，加性基线具有闭式 M-step 解、计算更高效、参数解释更直观的优势。

## 快速导航

- [完整文档](README.md) - 详细的技术实现、模型公式、实验结果
- [核心代码](hawkes_em_additive.py) - 加性基线 EM 算法实现
- [实验脚本](run_experiment_additive.py) - 45 只股票实盘实验
- [结果展示](results_additive/) - 完整的实验结果和可视化

## 核心特性

- **EM 算法闭式解**：所有参数（μ, α, γ_spread）均有闭式 M-step 解，无需数值优化
- **加性基线模型**：$\lambda_i(t) = \mu_i + \text{(intraday effects)} + \text{(spread effect)} + \text{(excitation)}$
- **三种递进模型**：Model A (常数 μ) → Model B (时变 μ) → Model C (时变 μ + spread 外生项)
- **全量数据拟合**：支持单只股票 78 万+ 事件的大规模数据
- **Cython 加速**：核心递推循环高性能优化

## 主要结果

| 指标 | Model A | Model B | Model C |
|------|---------|---------|---------|
| **AIC最优率** | 0/45 | 0/45 | **45/45 (100%)** |
| **BIC最优率** | 0/45 | 0/45 | **45/45 (100%)** |
| **LL单调性** | - | ✓ (45/45) | ✓ (45/45) |
| **收敛率** | 100% | 100% | 100% |

**关键发现**：
- Model C（时变 μ + spread 外生项）是明确胜出者
- 价格效应稳健：分枝比 High(0.61) < Mid(0.68) < Low(0.76)
- 开盘效应显著：开盘 30 分钟事件到达率为基准的 1.2–1.4 倍

## 数学公式示例

$$\lambda_i(t) = \mu_{i,\text{period}(t)} + \gamma_{\text{spread},i} \cdot x^+(t) + \sum_{j=1}^{4} \sum_{t_k^j < t} \alpha_{ij} \, \omega \, e^{-\omega(t - t_k^j)}$$

其中 $\mu_{i,\text{period}(t)}$ 使用分段常数哑变量刻画日内时段效应。

---

> **注意**：本页面已启用 MathJax 支持，所有数学公式应该正常渲染。如果仍有问题，请刷新页面或检查浏览器控制台。
