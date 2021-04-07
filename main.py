import argparse

import numpy as np

import specifications
from wrapped import train, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', "-train", type=int, default=100, help='Size of the training dataset, 1 to 1000')
    # -4, -5 denotes extra tasks "shortest path" and "eulerian circuit" respectively
    parser.add_argument('--task_id', "-task", type=int, default=18, help='Which bAbI task to perform. Valid values = 4,15,16,18,19,-4,-5')
    parser.add_argument('--hidden_layer_dim', "-hidden", type=int, default=None, help='Dimension of the hidden layer')
    parser.add_argument('--num_steps', "-step", type=int, default=5, help='Number of steps to unroll the propagation process') #Paper uses 5 steps (Ref: Section 5.1)
    parser.add_argument('--epochs', "-e", type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', "-b", type=int, default=10, help='Mini batch size')
    parser.add_argument('--model', '-m', type=str, default='ggnn', help='The neural network used to solve the task')

    '''
        For bAbI task 4, there are 4 types of questions: n,w,e,s - encoded as 0,1,2,3. (Ref: Weston, Jason, et al).
        In the GGSNN paper (Ref: Li, Yujia, et al), the authors propose "training a separate GG-NN for each task".
        Moreover, "When evaluating model performance, for all bAbI tasks that contain more than one questions in
        one example [here, Task 4], the predictions for different questions were evaluated independently."
        (section titled: More Training Details)
        Hence, task 4's individual questions were evaluated separately.
        Here, question id has default value 0 for other tasks.
    '''

    parser.add_argument('--question_id', "-q", type=int, default=0, help='Task 4 has 4 questions. Please enter the question id (0-3)')

    '''
        From the section titled "More training details" in the assigned paper,
        "For all tasks in this section, we generate 1000 training examples and 1000 test examples,
        50 of the training examples are used for validation. As there is randomness in the dataset
        generation process, we generated 10 such datasets for each task, and report the mean and
        standard deviation of the evaluation performance across the 10 datasets."
        Here, the input can be any number from 1 to 10.
        The default value is 0, indicating the calculation of average accuracy across all datasets.
    '''

    parser.add_argument('--dataset_id', "-data" , type=int, default=0, help='There are 10 datasets per task, 1-10. Enter 0 to calculate the average across all datasets')

    args = parser.parse_args()
    if not check_valid(args):
        return

    print_intro_message(args)

    accuracy = []
    data_ids = [args.dataset_id] if args.dataset_id else range(1, 11)
    for i in data_ids:
        args.dataset_id = i
        train_output = train(args)
        test_output = predict(args, train_output['net'])
        accuracy.append(np.mean(test_output['accuracy']))

    print("Results obtained for task ", specifications.TASK_NAMES[args.task_id],
          (args.task_id == 4) * ("for question: " + str(args.question_id)),
          "using the ",args.model, " architecture with", args.num_train, "training examples is", "{:.2f}%".format(np.mean(accuracy) * 100))


def check_valid(args):
    args.model = args.model.lower()
    args.hidden_layer_dim = args.hidden_layer_dim or (8 if args.model == 'ggnn' else 50)
    checks = [
        (args.question_id == 0 or (args.task_id == 4 and args.question_id in range(4)),
         "Please enter a valid question id, tasks other than task 4 can only have question id 0"),
        (args.dataset_id in range(11), "Please enter a valid dataset id (0-10)"),
        (args.task_id in specifications.SUPPORTED_TASKS, "Task id is not supported."),
        (args.model in specifications.SUPPORTED_NETS, "Model is not supported.")
    ]
    print('\n'.join([msg for check, msg in checks if not check]))
    return all([check for check, msg in checks])


def print_intro_message(options):
    print_msg("An implementation of the paper \"Gated Graph Sequence Neural Networks\" by Yujia Li et al.")
    print_msg("In this implementation, we deal with", len(specifications.TASK_NAMES), "tasks: " +
              ', '.join('Task ' + str(tid) + ':(' + name + ')' for tid, name in specifications.TASK_NAMES.items()) + '.')
    print("\n")
    print_msg("Task selected:", options.task_id, specifications.TASK_NAMES[options.task_id])
    print_msg("Architecture selected:", options.model)

def print_msg(*args):
    if specifications.VERBOSE:
        print(*args)



if __name__ == "__main__":
    main()


'''
Reference:

- Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).
- Weston, Jason, et al. "Towards ai-complete question answering: A set of prerequisite toy tasks." arXiv preprint arXiv:1502.05698 (2015).

'''
