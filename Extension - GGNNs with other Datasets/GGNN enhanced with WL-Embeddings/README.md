-------- Note -----------

This section contains the [Weisfeiler-Lehman Embedding (WLE) ](https://arxiv.org/abs/2006.06909)  applied to the Gated Graph Neural Network and tested on the following datasets:

- [PPI (Protein Protein Interaction)](https://arxiv.org/abs/1707.04638) dataset.
- [BBBP (Blood-Brain Barrier Penetration)](https://arxiv.org/abs/1703.00564) dataset
- [Tox21 (Toxicity of products in the 21st century)](https://tox21.gov) dataset

Please find a pre-trained model and a screenshot of the corresponding training for each of these datasets, for each modification of the WLE (plain WLE, concatenated CWLE and gated-sum GWLE), for easy replication.

Code Reference:

We have used the official implementation of the GGNN and the datasets from [Chainer Chemistry]( https://chainer-chemistry.readthedocs.io/en/latest/install.html ). The WLE extensions on the GGNN have been trained with datasets from [Stanford University's MoleculeNet](https://arxiv.org/abs/1703.00564). Code reference and documentation of the same is maintained by Chainer Chemistry, at [ https://chainer-chemistry.readthedocs.io/en/latest/install.html ]( https://chainer-chemistry.readthedocs.io/en/latest/install.html ).
