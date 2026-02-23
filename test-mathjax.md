---
layout: default
title: MathJax Test
---

# MathJax 公式测试

## 基本测试

行内公式：$\lambda_i(t) = \mu_i + \alpha_{ij} \omega e^{-\omega t}$

块级公式：
$$\lambda_i(t) = \mu_{i,\text{period}(t)} + \gamma_{\text{spread},i} \cdot x^+(t) + \sum_{j=1}^{4} \sum_{t_k^j < t} \alpha_{ij} \, \omega \, e^{-\omega(t - t_k^j)}$$

## \text 命令测试

- $\text{intraday effects}$
- $\text{spread effect}$
- $\text{excitation}$
- $\mu_{i,\text{open}}$
- $\mu_{i,\text{mid}}$
- $\mu_{i,\text{close}}$
- $\mu_{i,\text{normal}}$
- $\gamma_{\text{spread},i}$

## 复杂下标测试

- $p_{n,\text{base}}$
- $p_{n,\text{spread}}$
- $p_{n,\text{excitation}}$
- $\mu_{i,p} = \frac{\sum_{n:u_n=i,\text{period}(n)=p} p_{n,\text{base}}}{T_p}$

## 特殊符号测试

- $\varphi_{ij}(\Delta t)$
- $\int_0^\infty \varphi_{ij}(s)\,ds = \alpha_{ij}$
- $\text{Exp}(1)$
- $S_{\text{mean}}$, $S_{W_1}$, $S_{\text{LB}}$, $S_{\text{ACF}}$

## 集合符号测试

- $p \in \{\text{open}, \text{mid}, \text{close}, \text{normal}\}$
- $\omega \in \{0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0\}$

如果以上公式都能正确显示，说明MathJax配置正常。
