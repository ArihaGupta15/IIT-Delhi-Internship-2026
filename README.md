# IIT-Delhi-Internship-2026

#abstract

Background: Understanding the complex neural wiring of Autism Spectrum Disorder (ASD) requires moving beyond static brain imaging. Graph Neural Networks (GNNs) offer a powerful, dynamic lens to decode these intricate functional relationships directly from resting-state fMRI data.

Objective: This project leverages the ABIDE dataset to improve ASD classification by treating the brain as a dynamic network, while simultaneously interpreting the hidden neural patterns actually learned by these artificial intelligence models.

Methods: The research progressed from fundamental graph theory to advanced neuro-AI architectures. To establish a baseline in graph convolutions and message passing, a two-layer Graph Convolutional Network (GCN) was first built from scratch and tested on the Cora citation network. This foundation guided a comprehensive review of state-of-the-art methods—including spatiotemporal GNNs, brain transformers, and demographic-aware networks—culminating in the end-to-end implementation of the FBNetGen framework.

Results: By utilizing FBNetGen, the model successfully generated task-aware brain graphs optimized specifically for diagnostics, bypassing the limitations of rigid, pre-calculated Pearson-correlation matrices. A subsequent deep dive into the GCN classifier’s decision-making process mapped its learned "latent space" using PCA and UMAP dimensionality reduction.

Conclusion: The latent space analysis revealed how the AI visually clusters ASD versus healthy control groups, handles variations in hospital acquisition sites, and processes misclassified subjects. This demonstrates that dynamic, graph-based learning captures significantly richer, more actionable functional-connectivity representations than raw baseline data alone.
