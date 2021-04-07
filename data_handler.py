import os

import numpy as np
import torch
import torch.utils.data as data

import specifications


def get_data_loader(args, mode):
    if args.model != 'ggnn':
        file_path = '/'.join(
            ['babi_data', 'processed_' + str(args.dataset_id), 'rnn', mode, str(args.task_id) + '_rnn.txt']
            if args.task_id > 0 else
            ['babi_data', 'extra_seq_tasks', 'fold_' + str(args.dataset_id), 'noisy_rnn', mode,
             str(abs(args.task_id)) + '_rnn.txt'])
        with open(file_path, 'r') as f:
            file_content = [[int(c.strip()) for c in line.split() if c] for line in f]
            raw_x = torch.nn.utils.rnn.pad_sequence([torch.Tensor([[j == i for j in range(args.hidden_layer_dim)]
                                                                   for i in l[:-1]]) for l in file_content],
                                                    batch_first=True)
            raw_y = torch.Tensor([l[-1] for l in file_content])
            data = torch.utils.data.TensorDataset(raw_x, raw_y)
    else:
        types = dict()
        for info in ['edge_types', 'node_ids', 'question_types', 'labels', 'graphs']:
            file_path = '/'.join(
                ['babi_data', 'processed_' + str(args.dataset_id), mode, str(args.task_id) + '_' + info + '.txt']
                if args.task_id > 0 else
                ['babi_data', 'extra_seq_tasks', 'fold_' + str(args.dataset_id), 'noisy_parsed', mode,
                 str(abs(args.task_id)) + '_' + info + '.txt']
            )
            types[info] = file_path if info == 'graphs' else get_dtype(file_path) if os.path.exists(file_path) else None
            types['n_' + info] = len(types[info]) if types[info] else -1
            vars(args)['n_' + info] = len(types[info]) if types[info] else -1
        types['labels'] = types['labels'] or types['node_ids']
        data = BABI(args, types)
    return data, torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=(mode == 'train'),
                                             pin_memory=True, num_workers=0)


def graphs_from_file(file_name):
    data_list, edge_list, target_list = [], [], []
    max_number_in_graph = -float("inf")
    with open(file_name, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                data_list.append([edge_list, target_list])
                edge_list, target_list = [], []
            else:
                line_tokens = [c.strip() for c in line.split(" ") if c]
                int_tokens = list(map(int, line_tokens[line_tokens[0] == "?":]))
                if line_tokens[0] == "?":
                    target_list.append(int_tokens)
                else:
                    edge_list.append(int_tokens)
                max_number_in_graph = max(max_number_in_graph, max(int_tokens))
    return data_list, max_number_in_graph


def preprocess(data_list, n_annotation_dim, num_nodes, n_questions, task_id):
    task_data_list = [[] for _ in range(n_questions)]
    for item in data_list:
        fact_list = item[0]
        ques_list = item[1]
        for question in ques_list:
            question_type = question[0]
            if task_id == 19:
                question_output = np.zeros([1 + (task_id == 19)])
                assert (len(question) == 5)
                question_output[0] = question[-2]
                question_output[1] = question[-1]
            else:
                question_output = np.array(question[-1])
            annotation = np.zeros([num_nodes, n_annotation_dim])
            for anno_index in range(n_annotation_dim):
                annotation[question[anno_index + 1] - 1][anno_index] = 1
            task_data_list[question_type - 1].append([fact_list, annotation, question_output])
    return task_data_list


def generate_adj_matrix(edges, num_nodes, n_edge_types):
    a = np.zeros([num_nodes, num_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx - 1][(e_type - 1) * num_nodes + src_idx - 1] = 1
        a[src_idx - 1][(e_type - 1 + n_edge_types) * num_nodes + tgt_idx - 1] = 1
    return a


def get_dtype(data_path):
    data_type = {}
    with open(data_path, 'r') as f:
        for line in f:
            if len(line.strip()) != 0:
                line_tokens = line.strip('\n').split("=")
                assert (len(line_tokens) == 2)
                data_type[line_tokens[0]] = line_tokens[1]
    return data_type


class BABI(data.Dataset):
    def __init__(self, args, types):
        super(BABI, self).__init__()
        self.__dict__.update(types)
        self.all_data, max_number_in_graph = graphs_from_file(self.graphs)
        for k in ['n_node_ids', 'n_labels']:
            if self.__dict__[k] < 0:
                self.__dict__[k] = max_number_in_graph
                vars(args)[k] = max_number_in_graph
        self.data = preprocess(self.all_data[args.num_train:], specifications.ANNOT_DIMENSION[str(args.task_id)],
                                 self.n_node_ids, self.n_question_types, args.task_id)[args.question_id]

    def __getitem__(self, item):
        adj_matrix = generate_adj_matrix(self.data[item][0], self.n_node_ids, self.n_edge_types)
        annotation = self.data[item][1]
        answer = self.data[item][2] - 1
        return adj_matrix, annotation, answer

    def __len__(self):
        return len(self.data)
