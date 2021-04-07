# Repository for Team 3's implementation and verification of the paper [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)

(This project has been carried out by **Team 3, a single member team**. Due credit has been given in the individual report to the minimal code contributed by the only other member before they dropped the course.)

I have implemented the paper in Python, using Pytorch. This implementation is used to verify the results of the paper, in particular the claims of perfect (or near-perfect) accuracy on [bAbI](https://research.fb.com/downloads/babi/) tasks 4, 15, 16, 18 and 19, and learning graph algorithms like shortest path and Eulerian circuit.

In implementing the code, I have referred to the [original paper](https://arxiv.org/abs/1511.05493), translating the equations directly into Pytorch code and closely following the description of the proposed model. In certain cases where the paper's description was insufficient to translate to code (like defining [Graph Level Features](https://github.com/AdvTop-ML-Team3/Project/blob/main/model.py)), I have referred the [original author's code in Lua](https://github.com/yujiali/ggnn), for an idea as to how to proceed.

Further, since the paper does not deal with the variation of results depending on the format of input data, I have used the [preprocessing code](https://github.com/yujiali/ggnn/tree/master/babi/data) provided by the original author, Dr. Yujia Li. I am grateful to the authors, for sharing these online.

---

### Requirements

This code has been tested in an environment, with:
- Python 3.8
- Pytorch 1.7

---
### Execution

To run this code, please run the following command from the command prompt:
```
python main.py --task_id TASK_ID
```

#### Possible command line arguments:

```
  --num_train:          Size of the training dataset, 1 to 1000
  --task_id:            Which bAbI task to perform. Valid values = 4,15,16,18,19,-4,-5 (Please see section below)
  --hidden_layer_dim:   Dimension of the hidden layer
  --num_steps:          Number of steps to unroll the propagation process (paper suggests 5 steps)
  --epochs:             Number of training epochs
  --batch_size:         Mini batch size
  --model:              The neural network used to solve the task ('ggnn','rnn','lstm') <- to compare the performance
  --question_id:        Task 4 has 4 questions. Please enter the question id (0-3), for other tasks please leave it at the default 0
  --dataset_id:         There are 10 folds of datasets per task, 1-10. Enter 0 to calculate the average across all datasets
```

---

### Task ID to Task mapping:

Tasks -4 and -5 are the simple graph algorihtm learning tasks. Tasks 4, 15, 16, 18 19 are [bAbI](https://research.fb.com/downloads/babi/) tasks.

- -4:   "Shortest Path"
- -5:   "Eulerian Circuit"
- 4:    "Two Argument Relations"
- 15:   "Basic Deduction"
- 16:   "Basic Induction"
- 18:   "Size Reasoning"
- 19:   "Path Finding"

### Reference:

- Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493.


---
<!--
Possible extensions

- [ ] Reproduce the code for the original paper from existing Github resources and make enough refactoring
- [ ] Try to choose and determine the directions of possible expansion from the following list

Possible directions for expansion:

- [ ] New experimental setup
  - Perform experiment on other datasets - citation datasets are a good example
  - Perform experiment with other tasks (no need to do all 20 tasks, just the most relevant - task 19 is important)

- [ ] Compare the architecture with other GNNs
  - Maybe there are more recent models that work better

- [ ] Add New layers to the proposed architecture and check if if improves/reduces the performance

- [ ] Ablation Studies
  - If one part of the neural network appears to be the reason behind good performance, remove that part and repeat the experiment to confirm/deny our hypothesis

- [ ] Different Evaluation protocol
  - instead of accuracy try a different metric
  - on the pathfinding task analyse the types of errors encountered and why they might be occurring.

- [ ] Microstudy (?) - he said this will be very difficult
  - Study one specific case analyse the model behaviour on fixed examples and tweak/modify the model accordingly. -->
