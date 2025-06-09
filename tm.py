import numpy as np
from utils import *

class TM:
    """Basic class for a multidimensional Turing machine"""

    def __init__(self, trans, tape=None, state='start'):
        self.state = state
        self.tape = tape or {}
        self.head_shape = np.array(trans[0][1]).shape
        self.N = len(trans[0]) - 4
        self.head = (0,) * self.N
        # self.trans = {(t[0],t[1]): tuple(t[2:]) for t in trans}
        self.trans = {}

        for transition in trans:

            symbol_blocks = np.array(transition[1])
            symbol_block_shape = symbol_blocks.shape
            symbol_blocks = symbol_blocks.ravel()
            symbol_blocks = [[i] if i != -1 else [0, 1] for i in symbol_blocks]
            symbol_blocks = cartesian_prod(*symbol_blocks)
            symbol_blocks = [i.reshape(symbol_block_shape) for i in symbol_blocks]

            for symbol_block in symbol_blocks:
                new_symbol_block = np.array(transition[3])
                new_symbol_block[new_symbol_block == -1] = symbol_block[new_symbol_block == -1]

                new_symbol_block = to_tuple(new_symbol_block)
                symbol_blocks = to_tuple(symbol_block)

                self.trans[(transition[0], symbol_blocks)] = (transition[2], new_symbol_block, *transition[4:])


    def read_tape(self):
        """Returns the symbol block at the current head"""
        symbol = np.empty(self.head_shape)
        points = cartesian_prod(*[list(range(i)) for i in self.head_shape])
        for point in points:
            point = tuple(point)
            tape_pos = tuple(np.array(self.head) + point)
            if tape_pos not in self.tape:
                symbol[point] = 0
            else:
                symbol[point] = self.tape[tape_pos]
        return symbol


    def write_tape(self, v):
        # symbol = np.empty(self.head_shape)
        v = np.array(v)
        points = cartesian_prod(*[list(range(i)) for i in v.shape])
        for point in points:
            point = tuple(point)
            tape_pos = tuple(np.array(self.head) + point)
            self.tape[tape_pos] = v[point]


    def step(self):
        """Iterate the Turing Machine by one full step"""

        # Use 0 as the default value if the tape does not contain a symbol at the head
        # if self.head not in self.tape:
        #     self.tape[self.head] = 0

        # Current state and symbol
        state_and_symbol = (self.state, to_tuple(self.read_tape()))

        # Determine next state, symbol, and how the head should move
        # Halt the machine if a transition is not defined
        if state_and_symbol in self.trans:
            new_state, new_symbol, *move = self.trans[state_and_symbol]
        else:
            new_state = 'halt'
            new_symbol = state_and_symbol[1]
            move = (0,) * len(self.head)

        # Swap state, modify the tape, move the head
        self.state = new_state
        # self.tape[self.head] = new_symbol
        self.write_tape(new_symbol)
        self.head = tuple(self.head[i] + move[i] for i in range(len(move)))


    def get_tape_as_array(self):
        """Returns the current Turing tape as an array"""
        keys = np.array(list(self.tape.keys()))
        mins = np.min(keys, axis=0)
        maxs = np.max(keys, axis=0)
        tape = np.zeros(maxs-mins+1, int)
        for pos in self.tape.keys():
            symbol = self.tape[pos]
            i = tuple(pos - mins)
            tape[i] = symbol
        return tape


    def pprint(self):
        for r in self.get_tape_as_array():
            print(''.join(map(lambda x: str(x), r)))


    def __call__(self, steps):
        """Runs the machine until the halt state or the given number of steps is reached. Returns the tape as an array."""
        for _ in range(steps):
            self.step()
            if self.state == 'halt':
                break
        return self.get_tape_as_array()






if __name__ == '__main__':

    # trans = [
    #     ['U', 0, 'R', 1,  1,  0],
    #     ['R', 0, 'D', 1,  0, -1],
    #     ['D', 0, 'L', 1, -1,  0],
    #     ['L', 0, 'U', 1,  0,  1],
    #     ['U', 1, 'L', 0, -1,  0],
    #     ['R', 1, 'U', 0,  0,  1],
    #     ['D', 1, 'R', 0,  1,  0],
    #     ['L', 1, 'D', 0,  0, -1],
    # ]
    # tm = TM(trans, state='U', head_shape=(2,2))
    #
    # tm.tape[(0,1)] = 3
    #
    # print(tm.tape)
    #
    # t = tm.read_tape()
    #
    # print(t)

    # trans = [
    #     ['a', [[0,1],[1,1]], 'b', [[0,0],[0,0]], +1, 0],
    # ]

    X=-1

    trans = [
        ['start', [[X,X,X],[X,0,X],[X,X,X]], 'start', [[X,X,X],[X,0,X],[X,X,X]], +1, 0],
    ]

    tm = TM(trans)
    print(tm.trans)

    tape = tm(100)

    print(tape)