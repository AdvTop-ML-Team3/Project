-------- Note -----------

This section of the ATML Reproducibility challenge contains the following file directory structure:
```
project
│   README.md
│
└─── GGNN enhanced with WL-Embeddings
       └─── Tox21 dataset
            │ Saved models for easy usage
            │ Training screenshots

       └───  QM9 dataset
            │ Saved models for easy usage
            │ Training screenshots

       └───  BBBP dataset
            │ Saved models for easy usage
            │ Training screenshots

        | ReadMe.md

└───GNN-FiLM on a GGNN
        └─── PPI dataset
            │ Saved models for easy usage
            │ Training screenshots
```


In pursuit of expansion of the paper 'Gated Graph Sequence Neural Networks' [1], this report has studied two extensions:

1. Feature-wise Linear Modulation [2].
2. Weisfeiler-Lehman Embeddings [3]

## 1. Feature-wise Linear Modulation of the GGNN [2]:

### Novelty:
- In Graph Neural Networks, nodes have a representation and this is sent as a message to all neighbours in the graph.
- The weight of a message depends on the representation of the edge between a source and target node.
- Further, all dimensions of the message use this weight.
- This implies that the message passed depends on the learnt weights and the source node representation.
- This computation dominates and doesn't allow for more complex transformation functions.
GNN-FiLM method introduces a hypernetwork and a modulation mechanism to overcome this drawback

### Motivation and Explanation
- The nature of GNN message passing means that complex transformation functions are difficult to implement since the message transformation function is the most dominant computation.
- FiLM uses a hyper network (a neural network to learn the parameters of the actual neural network under consideration).  
- Modulation is similar to the gated mechanism of GGNNs; "forgetting" can be equated to reducing the weight of messages. The difference is that GGNNs apply the gating to all incoming messages whereas in FiLM, application of the gating depends on the edge type.
- This usage of a hyper network and gating mechanism results in the GNN-FiLM performing better than the baseline GGNN.


### Datasets:

#### PPI

The PPI dataset [4] is a database of 24 graphs, each having 2500 nodes that help to understand the interaction between proteins and tissues for Node level classification. We can train GNNs on the multilayer networks to classify nodes as one of 121 classes.

### Results and Discussion:
- The original paper uses Average Micro F1 score to evaluate the performance of the FiLM enhanced GGNN.
- Average Micro F1 is a measure of the machine learning model on multi-label binary problems, by taking into account the contribution of all classes.
- An implementation of this model shows that the Micro F1 score increases with increase in training epochs. We ran 10 epochs and achieved a Micro F1 score of 57.5%.
- Given the rapid improvement, the results of the paper (Micro F1 score of 99% in 500 epochs) seems achievable.

### Code Reference:
For a thorough analysis of the performance of aGGNN  enhanced with FiLM on the PPI dataset we have used the original implementation released by Microsoft Research.  The FiLM extension of the GGNN was proposed and contributed by Marc Brockschmidt [2] (also a co-author of the original GGNN paper). Code reference and thorough documentation of the same is available at https://pypi.org/project/tf2-gnn/

---
## Weisfeiler-Lehman Embeddings [3]

### Novelty:
- Overcoming the decrease in performance associated with an increase in the depth of a Gated Graph Neural Network (and GNNs in general), by using Weisfeiler-Lehman WL-embeddings (explained in this section of the expansion).
- This is especially useful to help capture local structural information and predict chemical properties, in molecular-based prediction.

### Motivation:
- Graph-based neural networks suffer from the 'Curse of depth', meaning that an increase in the number of layers leads to a corresponding decline in performance
- Further analysis of this problem revealed that this was due to the node representations becoming 'indistinguishable' - a phenomenon called 'over-smoothing' [6].
- A similar problem in the non-graph domain of deep learning was overcome with the introduction of the ResNet [5].
- Similarly, for graph neural networks, the use of WL-embeddings has been proposed.

### Explanation
- In GNNs, a node embedding is the first action performed - to encode a node into a feature vector and associate it with a node label. This is called 'atomic embedding'.
- In WLE, we label a node with a combination of the node label and the neighbouring node label. Since every atom in a molecule will have a different set of neighbouring atoms, this embedding will be sufficiently unique for every atom.
- There is a problem that arises here - sparsity. WL-embeddings may create labels that are used less frequently.
- This is overcome in [3] by using two WL embeddings per node (atom):
  1. an atom embedding and
  2. a neighbor embedding,
- There arises a choice while deciding how to combine these two embeddings. The paper proposes two methods:
  1. Concatenation - called concatenated WL (C-WL)
  2. Weighting each embedding - called Gated-sum WL (G-WL)  

### Datasets:

Note: SMILES (simplified molecular-input line-entry system) representation of a molecule refers to a 2-D graph of the molecular structure that helps visualize a molecule. For Tox21 and QM9, I have included pictures of the SMILES diagrams for some molecules.

#### Tox21
Tox21 [7] is a multi-label, graph level classification dataset that studies the toxicity of products in the 21st century, hence the name Tox21. Toxicity tests are performed on many different molecules that are commonly present in food, cleaning agents and the environment in general to study their effects on human beings. The input to the model is the molecule under consideration and the label(s) is the assay name.

For eg., For the molecule SMILES=Cc1ccc([N+](=O)[O-])c2c1O[Hg]2

The image of the corresponding molecule is:

![alt text](https://github.com/AdvTop-ML-Team3/Project/blob/main/Extension%20-%20GGNNs%20with%20other%20Datasets/images/Tox21.png "Tox21 molecule")


There are 12 types of toxicity. The label has twelve entries, 0 = non toxic -1 = unknown and 1 = toxic

For the above molecule, labels [-1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1]



#### BBBP
Binary classification graph level classification
BBBP [8] is the Blood-Brain Barrier Penetration dataset (Binary classification at graph level), which results from a study of barrier membrane permeability between blood and brain fluid. The penetration of this layer is of importance to certain drugs, making this study extremely important to nervous system drug testing. The dataset has details of the permeability of almost 2000 labelled compounds.

#### QM9

QM9 is the Quantum Machine - 9 dataset [10][11] ((Graph level Regression) that has details of the properties of over 134,000 organic molecules. This dataset helps to study the structure and properties of molecules which help in the pharmaceutical industry to develop drugs and the material engineering industry. This dataset is often used in benchmarking existing methods.
For eg.,
For the following molecule,

![alt text](https://github.com/AdvTop-ML-Team3/Project/blob/main/Extension%20-%20GGNNs%20with%20other%20Datasets/images/QM9.png "QM9 molecule")

The output will be a vector of values predicted for 17 chemical properties (like energy of molecular orbitals, rotational constants, etc.) in the order described in [qm9].

### Results and Discussion

- While running the experiments, we obtained the results seen in Table 1 below. Due to resource limitations (extreme size of some datasets, lack of computation resources), the results mentioned in the paper were not obtained, although a sample preview of the possibilities is clearly visible.  

- The overall improvement in performance observed when using CWLE is visible for the BBBP (not documented in their paper, but implemented here) dataset and the Tox21 dataset. This might be attributable to the fact that they are both classification tasks.

- On the other hand, the GWLE offers the biggest advantage to the QM9 dataset, possibly due to the nature of the regression task.

|                          | Plain GGNN | GGNN with WL embeddings | GGNN with concatenated CWLE | GGNN with Gated-sum WLE |
|--------------------------|------------|-------------------------|-----------------------------|-------------------------|
| Accuracy scores reported |            |                         |                             |                         |
| Tox21                    | 91.90 (10) | 90.37 (10)              | 90.27 (10)                  | 90.32 (10)              |
| BBBP                     | 85.20 (10) | 75.29 (10)              | 86.75 (10)                  | 71.11 (10)              |
| MAE scores reported      |            |                         |                             |                         |
| QM9                      | 9.50 (2)   | 6.41 (2)                | 5.89 (2)                    | 5.65 (2)                |

Table 1: Performance of various WL-Embeddings on a GNN for three datasets

![alt text](https://github.com/AdvTop-ML-Team3/Project/blob/main/Extension%20-%20GGNNs%20with%20other%20Datasets/images/WLE%20Graph.png "Results graph")


### Code Reference:

In order to thoroughly test the performance of a GGNN on these datasets, we have used the off-the-shelf implementation of the GGNN and the datasets from Chainer Chemistry. The WLE extensions on the GGNN have been trained with datasets from Stanford University's MoleculeNet [8] Code reference and documentation of the same is maintained by Chainer Chemistry, at [9].

### Reference:

[1] Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493.


[2] Brockschmidt, M., 2020, November. Gnn-film: Graph neural networks with feature-wise linear modulation. In International Conference on Machine Learning (pp. 1144-1152). PMLR.

[3] Ishiguro, K., Oono, K. and Hayashi, K., 2020. Weisfeiler-Lehman Embedding for Molecular Graph Neural Networks. arXiv preprint arXiv:2006.06909.

[4] Zitnik, M. and Leskovec, J., 2017. Predicting multicellular function through multi-layer tissue networks. Bioinformatics, 33(14), pp.i190-i198.

[5] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[6] Qimai Li, Zhichao Han, and Xiao-ming Wu. Deeper Insights into Graph Convolutional Networks for Semi-supervised Learning. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI-18), 2018.

[7] [tox21] National toxicology Program, 2018. Toxicology in the 21st Century. Tox21. Available at: https://tox21.gov/

[8] Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande, MoleculeNet: A Benchmark for Molecular Machine Learning, arXiv preprint, arXiv: 1703.00564, 2017.

[9] Chemistry, C., Chainer Chemistry 0.5.0 documentation. Available at: https://chainer-chemistry.readthedocs.io/en/latest/install.html


[10] L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.

[11] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1, 140022, 2014. [bibtex]
