import numpy as np

class TM:
    """Basic class for a multidimensional Turing machine"""

    def __init__(self, trans, tape=None, state='start'):
        self.state = state
        self.tape = tape or {}
        self.trans = {(t[0],t[1]): tuple(t[2:]) for t in trans}
        self.N = len(trans[0]) - 4
        self.head = (0,) * self.N

    def step(self):
        """Iterate the Turing Machine by one full step"""

        # Use _ as the default value if the tape does not contain a symbol at the head
        if self.head not in self.tape:
            self.tape[self.head] = 0

        # Current state and symbol
        state_symbol = (self.state, self.tape[self.head])

        # Determine next state, symbol, and how the head should move
        # Halt the machine if a transition is not defined
        if state_symbol in self.trans:
            new_state, new_symbol, *move = self.trans[state_symbol]
        else:
            new_state = 'halt'
            new_symbol = state_symbol[1]
            move = (0,) * len(self.head)

        # Swap state, modify the tape, move the head
        self.state = new_state
        self.tape[self.head] = new_symbol
        self.head = tuple(self.head[i] + move[i] for i in range(len(move)))

        # print(self.tape)

    def __call__(self, steps):
        """Runs the machine until the halt state or the given number of steps is reached. Returns the tape as an array."""
        for _ in range(steps):
            self.step()
            if self.state == 'halt':
                break
        return self.get_tape_as_array()

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



if __name__ == '__main__':

    trans = [
        ['U', 0, 'R', 1,  1,  0],
        ['R', 0, 'D', 1,  0, -1],
        ['D', 0, 'L', 1, -1,  0],
        ['L', 0, 'U', 1,  0,  1],
        ['U', 1, 'L', 0, -1,  0],
        ['R', 1, 'U', 0,  0,  1],
        ['D', 1, 'R', 0,  1,  0],
        ['L', 1, 'D', 0,  0, -1],
    ]

    tm = TM(trans, state='U')
    tm.__call__(11000)
    tape = tm.get_tape_as_array()
    print(tape)