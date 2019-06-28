import pickle


class DisplayForm:
    def __init__(self, root_node):
        self.root_node = root_node
        self.open_nodes = {root_node}
        self.node_indices = [root_node]
        self.node_keys = {}

        self.add_nodes(self.root_node)
        for i, node in enumerate(self.node_indices):
            self.node_keys[node] = i

    def add_nodes(self, node):
        self.node_indices.append(node)
        for child in node.children.values():
            self.add_nodes(child)

    def node_display(self, node, move):
        if node in self.open_nodes:
            if node.has_children():
                status_sign = '+'
            else:
                status_sign = 'x'
        elif node.has_children():
            status_sign = '-'
        else:
            status_sign = ' '
        tabbing = '\t' * len(node.full_move_list) + '|' + status_sign + ' ' + str(move) + ' '
        if node.principal_variation:
            measure_str = " Q: {0:.4f} P: {1:.4f}".format(node.q, node.log_total_p)
        else:
            measure_str = " P: {0:.4f}".format(node.log_total_p)

        return tabbing + measure_str + "  (" + str(self.node_keys[node]) + ") Count: " + str(node.num_nodes) + "\n"

    def recursive_display(self, node, open_nodes, move):
        display = self.node_display(node, move)
        if node in open_nodes:
            for move, child in node.children.items():
                display += self.recursive_display(child, open_nodes, move)
        return display

    def __str__(self):
        return self.recursive_display(self.root_node, self.open_nodes, ())

    def open_node(self, node_index):
        node = self.node_indices[node_index]
        if node not in self.open_nodes:
            self.open_nodes.add(node)

        curr_parents = node.parents
        while len(curr_parents) > 0:
            self.open_nodes.add(curr_parents[0])
            curr_parents = curr_parents[0].parents

    def open_pv(self, node):
        best_child = node.best_child
        self.open_nodes.add(best_child)
        if best_child != self and best_child and  best_child.best_child:
            self.open_pv(best_child)

def reconstruction(matrix):
    from ..core.board import Board
    instructions = ""
    xs = []
    ys = []
    for i in range(0, SIZE):
        for j in range(SIZE):
            if matrix[i, j, Board.FIRST_PLAYER] == 1:
                xs.append((i, j))
            elif matrix[i, j, Board.SECOND_PLAYER] == 1:
                ys.append((i, j))
    for i in range(len(xs)):
        instructions += 'board.move(' + str(xs[i][0]) + ", " + str(xs[i][1]) + ")" + '\n'
        if i < len(ys):
            instructions += 'board.move(' + str(ys[i][0]) + ", " + str(ys[i][1]) + ")" + '\n'
    return instructions

SIZE = 9
def print_board(matrix):
    board_string = ""
    for i in range(0, SIZE):
        board_string += "\n"
        for j in range(SIZE):
            if matrix[j, i, 1] == 1:
                char = 'x'
            elif matrix[j, i, 2] == 1:
                char = 'o'
            else:
                char = ' '
            board_string += "|" + char
        board_string += "|"
    return board_string

node_num = -4442216286206565317
with open(str(node_num) + '.pkl', 'rb') as f:
    root_node = pickle.load(f)

df = DisplayForm(root_node)
print(print_board(root_node.get_matrix()))
print(df)
#print(reconstruction(root_node.get_matrix()))
while True:
    inp = input("_:")
    if inp.startswith('open'):
        command, x = inp.split(' ')
        try:
            x = int(x)
        except:
            print('Parsing coordinate error')
            continue
        df.open_node(x)
        print(df)

    if inp.startswith('openpv'):
        command, x = inp.split(' ')
        try:
            x = int(x)
        except:
            print('Parsing coordinate error')
            continue
        df.open_pv(df.node_indices[x])
        print(df)

    if inp.startswith('board'):
        command, x = inp.split(' ')
        try:
            x = int(x)
        except:
            print('Parsing coordinate error')
            continue
        if df.node_indices[x].has_matrix():
            print(print_board(df.node_indices[x].get_matrix()))
        else:
            print('No Matrix')

    if inp.startswith('reconstruct'):
        command, x = inp.split(' ')
        try:
            x = int(x)
        except:
            print('Parsing coordinate error')
            continue
        if df.node_indices[x].has_matrix():
            print(reconstruction(df.node_indices[x].get_matrix()))
        else:
            print('No Matrix')