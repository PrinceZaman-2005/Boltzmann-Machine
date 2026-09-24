# ✨ Boltzmann Machine

A simple **Boltzmann Machine built from scratch with NumPy**.

This project is part of my **10 Days of Modeling AI for Beginners** series.

Instead of learning through ordinary forward propagation and backpropagation, we explore a different idea:

> **Learning through energy, probability, and stochastic states.**

---

## Why Boltzmann Machines?

So far in this series:

* **Day 1 — Perceptron:** A neuron makes a decision.
* **Day 2 — Nested Perceptron:** Multiple neurons work together.
* **Day 3 — Backpropagation:** The network learns by propagating error backward.
* **Day 4 — Visualization:** We visualize how neural networks work.
* **Day 5 — Boltzmann Machine:** We explore stochastic, energy-based learning.

This introduces a different way of thinking about neural networks.

---

## Core Idea

A Boltzmann Machine contains units that can take stochastic states.

A simplified view:

```text
        Visible Units
       ●      ●      ●
      ╱│╲    ╱│╲    ╱│╲
     ╱ │ ╲  ╱ │ ╲  ╱ │ ╲
    ●──●──●──●──●──●──●
        Hidden Units
```

Instead of simply asking:

> "What output should this network produce?"

we can ask:

> "Which configuration of these units is more likely?"

The network uses an **energy function** to describe different configurations.

Lower-energy configurations are more probable.

---

## Energy

A simplified Boltzmann energy can be written as:

```text
E = -Σᵢ bᵢsᵢ - Σᵢ<ⱼ wᵢⱼsᵢsⱼ
```

Where:

* `s` = state of a unit
* `b` = bias
* `w` = connection weight
* `E` = energy of the current configuration

The important intuition is:

```text
Lower Energy
     ↓
Higher Probability
     ↓
More likely state
```

---

## Stochastic Units

Unlike the deterministic neurons used earlier in the series, Boltzmann Machine units are stochastic.

A unit can be sampled using a probability such as:

```text
P(s = 1) = sigmoid(input)
```

So the same input does not necessarily produce exactly the same state every time.

The network **samples** possible states.

---

## What This Project Demonstrates

This implementation focuses on understanding the mechanism rather than building a large-scale model.

The experiment explores:

* Stochastic neuron states
* Weights and biases
* Energy calculation
* Sigmoid probability
* Positive and negative phases
* Weight updates
* Energy-based learning

---

## Learning Goal

The goal of this project is **not** to build a state-of-the-art generative model.

The goal is to make the mechanism understandable.

Instead of hiding the mathematics behind a framework, this implementation exposes the basic process:

```text
State
  ↓
Energy
  ↓
Probability
  ↓
Sampling
  ↓
Learning
  ↓
New State
```

---

## Important Note

This is a **small educational implementation**.

A practical Boltzmann Machine can involve much larger networks, more sophisticated sampling methods, and computationally expensive training.

This project intentionally keeps the system small enough to inspect and experiment with.

The implementation may therefore differ substantially from production-scale implementations.

---

## Part of a Series

**10 Days of Modeling AI for Beginners**

- Day 1 → Perceptron
- Day 2 → Nested Perceptron
- Day 3 → Backpropagation
- Day 4 → Neural Network Visualization
- Day 5 → Boltzmann Machine

The goal is simple:

> **Don't just use AI. Understand how it works.** 🐱
