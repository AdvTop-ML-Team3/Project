VERBOSE = True
# Node classification tasks need 1D annotations, Graph level classification 2D annotations
ANNOT_DIMENSION = {'4': 1, '15': 1, '16': 1, '18': 2, '19': 2, '-4': 2, '-5': 2}
SUPPORTED_TASKS = [4, 15, 16, 18, 19, -4, -5]
SUPPORTED_NETS = ['ggnn', 'rnn', 'lstm']
TASK_NAMES = {
    -4: "Shortest Path",
    -5: "Eulerian Circuit",
    4: "Two Argument Relations",
    15: "Basic Deduction",
    16: "Basic Induction",
    18: "Size Reasoning",
    19: "Path Finding"
}
