# BERT Model Development Log

## Overview
This document provides a detailed log of various experiments, configurations, and findings during the development and refinement of BERT models with various attention mechanisms.

## Changelog

### Initial Experiments
- **Softmax and Cosine Attention:** Initial trials showed poor performance with both methods. Particularly, cosine attention did not converge.

### Improving Cosine Attention
- **Gradient Clipping:** Stabilized training to an extent but results remained subpar.
- **Increased Weight Decay (0.1):** No noticeable improvement.
- **Reduced Learning Rate (1e-5):** Stabilized training; cosine began outperforming softmax.

### Transition to AdamW
- Post-switch to AdamW, softmax attention with learning rates of 1e-5 and 1e-4 significantly outperformed cosine attention.

### Optimizing Cosine Attention Convergence
- **ReLU on Attention Matrix:** Improved convergence significantly, hypothesizing that model could "ignore" less useful tokens.
- **Activations Other than ReLU:** Did not yield improvements, potentially due to inability to discard tokens.
- **Higher Learning Rates:** Rates above 1e-5 caused divergence; 1e-5 remained optimal.
- **Token Dropout and Angle Distance:** Both modifications led to worse performance compared to baseline ReLU-enhanced cosine attention.

### Addressing Magnitude Issues in Attention Scores
- **Normalizing Attention Matrix:** By dividing attention scores by sequence length, stability at higher learning rates was achieved, surpassing ReLU-based methods.

### Revised Magnitude Normalization
- **Square Root Normalization:** Initially promising, but led to divergence at higher sequence lengths.
- **Learnable Exponent Normalization:** Showed promise but occasionally unstable.
- **Vector Normalization:** Normalizing Q, K, and V vectors provided stable improvements.
- **Loss Penalization for Output Magnitude:** Attempts to penalize high-magnitude outputs did not converge effectively.

### Data Quality Adjustments
- **Punctuation Restoration and Data Cleaning:** Correcting earlier data preprocessing errors improved data quality and model performance.

### Stability with New Data
- **Normalization by Sequence Length:** Reverted to this method to address instability issues caused by long sequences and high vector magnitudes.

## Future Experiments
### Activation Functions
- Explore the impact of different activation functions on the attention matrix:
  - ReLU
  - Sigmoid
  - Softmax (re-evaluation)

### Similarity Scoring
- Evaluate various similarity measures for constructing the attention matrix:
  - Cosine similarity
  - Euclidean distance
  - Manhattan distance

### Learnable Similarity Measures
- Experiment with allowing the model to learn the optimal similarity measure, such as a learnable Chebyshev p-value.
