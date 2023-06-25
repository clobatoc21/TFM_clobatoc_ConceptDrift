# TFM_clobatoc_ConceptDrift

Repository for final thesis. Ciencia de Datos, UOC.

The constant generation of new data by users, governments and companies has derived, in recent years, in the creation of many analytical instruments to support decision-making.
Most of these tools, including machine learning models, find generalization patterns based on an assumption of staticity in the data; that is, it is understood that the data on which the conclusions are drawn are independent and identically distributed (IID). However, the study of non-stationary environments reflects variations in the distribution of data over time, which are known as concept drift and decrease the predictive capacity of the models if these are not adequately adapted.

This project emerges and is based on the work published by Cloudera Fast Forward Labs in 2021 [1], in which different drift inference methods were evaluated in an unsupervised environment. Through the creation of synthetic datasets and the modification of the experiments to obtain an incremental learning, this project aims to address two of its main limitations.

The generation of synthetic data allows for a controlled introduction of concept drift, which
is key for an accurate analysis of the experiments. The introduction of incremental learning, on
the other side, removes the waiting times associated to batch learning.

[1] Nisha Muktewar Andrew Reed. Inferring concept drift without labeled data, 2021.

**Code for Cloudera's project can be found here: https://github.com/fastforwardlabs/concept-drift
