'''
This file contains my implementation of the GGNN and GGSNN models described in the paper [1]. 
Code contains equations from Sections 3 and 4 in the paper [1] and some inspiration from the author's Lua implementation at: https://github.com/yujiali/ggnn/blob/master/ggnn
No code has been reproduced from the original repository - I've only used the core ideas for direction. 
'''

import torch
import torch.nn as nn

import specifications


class GGNN(nn.Module):

    def __init__(self, args):
        super(GGNN, self).__init__()
        # Initialise values from main args
        self.task_id = args.task_id
        self.hidden_layer_dim = args.hidden_layer_dim
        self.annotation_dim = specifications.ANNOT_DIMENSION[str(args.task_id)]

        # Initialise values from command line arguments
        self.num_nodes = args.n_node_ids
        self.num_edges = args.n_edge_types
        self.num_labels  = args.n_labels
        self.num_steps = args.num_steps

        # The first layer of the GGNN
        self.linear = nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim * self.num_edges)

        # Basic Recurrence propagation layer
        self.propagate = Propagation(self.hidden_layer_dim, self.num_nodes, self.num_edges)

        # GGSNNs need graph level features and different output layer
        self.graph_level_feature = GraphLevelFeature(self.hidden_layer_dim, self.annotation_dim)
        self.graph_linear_output = nn.Linear(self.hidden_layer_dim,
                                             self.num_labels)  # Output layer for graph level classification

        # For node level classification by GGNN
        self.linear_hidden = nn.Linear(self.hidden_layer_dim + self.annotation_dim, self.hidden_layer_dim)
        self.node_linear_output = nn.Linear(self.hidden_layer_dim, 1)  # Output layer for node level classification
        self.tanh = nn.Tanh()

    def forward(self, initial_state, annotation, adj_matrix):

        track_path = []

        for i in range(self.num_steps):
            # Pass incoming and outgoing states through two separate linear layers
            incoming_edge = self.linear(initial_state)
            incoming_edge = incoming_edge.view(-1, self.num_nodes, self.hidden_layer_dim, self.num_edges).transpose(2, 3).transpose(1, 2).contiguous()
            incoming_edge = incoming_edge.view(-1, self.num_nodes * self.num_edges, self.hidden_layer_dim)

            outgoing_edge = self.linear(initial_state)
            outgoing_edge = outgoing_edge.view(-1, self.num_nodes, self.hidden_layer_dim, self.num_edges).transpose(2, 3).transpose(1, 2).contiguous()
            outgoing_edge = outgoing_edge.view(-1, self.num_nodes * self.num_edges, self.hidden_layer_dim)

            initial_state = self.propagate(incoming_edge, outgoing_edge, initial_state, adj_matrix)

            track_path.append(initial_state)

            # Authors describe tasks 18 and 19 separately as graph level classification
        if self.task_id == 18:
            output = self.graph_level_feature(torch.cat((initial_state, annotation), 2))
            output = self.graph_linear_output(output)

        elif self.task_id == 19:
            # Additionally, task 19 needs to keep track of the path, to determine if there exists a path at all
            layer1 = self.graph_level_feature(torch.cat((track_path[0], annotation), 2))
            layer1 = self.graph_linear_output(layer1).view(-1, 1, self.num_labels )
            layer2 = self.graph_level_feature(torch.cat((track_path[1], annotation), 2))
            layer2 = self.graph_linear_output(layer2).view(-1, 1, self.num_labels )
            output = torch.cat((layer1, layer2), 1)

        else:
            # Node selection tasks or node-level classification tasks
            output = self.linear_hidden(torch.cat((initial_state, annotation), 2))
            output = self.tanh(output)
            output = self.node_linear_output(output).sum(2)

        return output


class GraphLevelFeature(nn.Module):
    '''
    The paper handles Task 18 and 19 differently, where the task is graph-level classification
    Base Reference for code (in Lua): https://github.com/yujiali/ggnn/blob/master/ggnn/GraphLevelGGNN.lua
    '''

    def __init__(self, hidden_layer_dim, num_annotations):
        super(GraphLevelFeature, self).__init__()

        self.hidden_layer_dim = hidden_layer_dim
        self.num_annotations = num_annotations

        self.linear = nn.Linear(self.hidden_layer_dim + self.num_annotations, self.hidden_layer_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, initial_state):
        # Equation 7 in the original paper.
        gate = self.sigmoid(self.linear(initial_state))
        h = self.tanh(self.linear(initial_state))
        gated_h = (gate * h).sum(1)
        act_gated_h = self.tanh(gated_h)
        return act_gated_h


class Propagation(nn.Module):
    '''
    Basic recurrence of the propagation model (Section 3.2)
    '''

    def __init__(self, state_dim, num_nodes, num_edge_types):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.gate = nn.Linear(state_dim * 3, state_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, state_in, state_out, state_cur, adj_matrix):
        '''
        I've directly coded the equations from Section 3.2 of the paper (Li, Yujia, et al.).
        '''

        # From fig 1(c), I've separated the adjacency matrix into A_in and A_out.
        boundary = self.num_nodes * self.num_edge_types
        A_in = adj_matrix[:, :, :boundary]
        A_out = adj_matrix[:, :, boundary:]

        # Multiply the incoming edge matrix with the corresponding node state
        a_in = torch.matmul(A_in, state_in)
        a_out = torch.matmul(A_out, state_out)

        # Message passing over edges - Eqn 2 in the paper
        # Concatenation allows for hidden states with larger than the annotation size [ref]
        a = torch.cat((a_in, a_out, state_cur), 2)

        # The update gate computation - Eqn 3
        update = self.sigmoid(self.gate(a))
        # The reset gate computation - Eqn 4
        reset = self.sigmoid(self.gate(a))

        # Activated node state vector - Eqn 5
        h_tilde = self.tanh(self.gate((torch.cat((a_in, a_out, reset * state_cur), 2))))

        # Output of the node - Eqn 6
        output = (1 - update) * state_cur + update * h_tilde

        return output
'''
Ref:
- [1] Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).
'''
