from random import randint
import re
import time

import numpy as np
from matplotlib import pyplot as plt


def egcd(a, b):  # a < b
    if b < 0:
        d, x, y = egcd(a, -b)
        return d, x, -y
    if a == 0:
        # Base case
        return b, 0, 1
    q, r = b // a, b % a
    d, x, y = egcd(r, a)
    # d = rx + ay
    #   = (b-qa)x + ay = a(y - qx) + bx
    return d, y - q*x, x


def get_viewport(pts):
    xs, ys = [xy[0] for xy in pts], [xy[1] for xy in pts]
    return [min(xs) - 5, max(xs) + 5, min(ys) - 5, max(ys) + 5]


def draw_arrow(ax, fr, to, label):
    ax.annotate(label, xy=[to[0]+.5, to[1]+.5], color='blue', textcoords='data')
    ax.annotate('', xy=to, color='blue', xytext=fr, arrowprops=dict(arrowstyle="->"))


def sample_random_unitary():
    MAXC = 20
    d, x, y = 0, 0, 0
    while d != 1:
        a, b = randint(-MAXC, MAXC), randint(-MAXC, MAXC)
        d, x, y = egcd(a, b)
    # d = ax + by
    return np.array([[a, b], [-y, x]])


def parse_update(s, dim=2):
    U = np.eye(dim, dtype=np.int64)

    if s == "swap":
        return np.array([[0, 1], [1, 0]], dtype=np.int64)
    parts = s.strip().split("=")
    if len(parts) != 2:
        print("INPUT ERROR: missing '='")
        return None
    lhs, rhs = parts

    # Check LHS
    if lhs[0] != 'b':
        print("INPUT ERROR: Write e.g. 'b1 =' or 'b2 ='")
        return None
    try:
        idx_out = int(lhs[1:]) - 1
        if idx_out < 0 or idx_out >= dim:
            print(f"INPUT ERROR: Only basis vectors in range b1 ... b{dim} can be used")
            return None
        U[idx_out, idx_out] = 0  # Reset this position, it should be filled.

        # Check RHS
        bits = [x.strip() for x in re.split('([-+])', rhs)]
        sgn_pos = True
        for bit in bits:
            if bit == '-':
                sgn_pos = not sgn_pos
            elif bit != '+':
                parts = [x.strip() for x in bit.split('b')]
                if len(parts) != 2:
                    print(f"INPUT ERROR: part '{bit}' I could not parse as e.g. '5 b1'")
                    return None
                if len(parts[0]) > 1 and parts[0][-1] == '*':
                    # strip away '*' sign.
                    parts[0] = parts[0][:-1].strip()
                elif parts[0] == '':
                    parts[0] = '1'
                mult, idx_in = int(parts[0]), int(parts[1]) - 1
                mult *= (1 if sgn_pos else -1)
                sgn_pos = True
                if idx_in < 0 or idx_in >= dim:
                    print(f"INPUT ERROR: Only basis vectors in range b1 ... b{dim} can be used")
                    return None
                U[idx_in, idx_out] += mult
    except ValueError:
        print("INPUT ERROR: integer expected at some point...")
        return None
    if abs(U[idx_out, idx_out]) != 1:
        print("!!! WARNING !!! You are moving to a sublattice. "
              "The old basis cannot be expressed (with integers) in terms of the new basis!")
        # return None
    return U


def read_update(b0, b1):
    U = None
    while U is None:
        try:
            line = input("Update basis (e.g.: b1 = b1 - 2*b2) ")
        except EOFError:
            return b0, b1, False
        if line == "" or line == 'quit':
            return b0, b1, False
        U = parse_update(line)

    # Update basis.
    b0p = b0 * U[0, 0] + b1 * U[1, 0]
    b1p = b0 * U[0, 1] + b1 * U[1, 1]
    return b0p, b1p, True


def basis_is_reduced(g0, b0, b1, dim=2):
    sqn = sum(x**2 for x in b0)
    dot = sum(b0[i] * b1[i] for i in range(dim))

    if sqn > sum(x**2 for x in g0):
        return False  # b0 could have been reduced more...
    # Check size-reducedness of b1.
    return 2 * abs(dot) <= sqn


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.ion()
    plt.show()

    MAXC = 5
    g0, g1 = [0, 0], [0, 0]
    while g0[0] * g1[1] - g0[1] * g1[0] == 0:
        g0 = np.array([randint(-MAXC, MAXC) for _ in range(2)])
        g1 = np.array([randint(-MAXC, MAXC) for _ in range(2)])
    if sum(x**2 for x in g0) > sum(x**2 for x in g1):
        g0, g1 = g1, g0
    print(f"Secret basis: b1 = {g0}, b2 = {g1}")

    U = sample_random_unitary()
    # Update basis.
    b0 = g0 * U[0, 0] + g1 * U[1, 0]
    b1 = g0 * U[0, 1] + g1 * U[1, 1]

    print(f"Initial basis: b1 = {b0}, b2 = {b1}")

    cont = True
    while cont:
        plt.cla()  # clear axis

        plt.axis(get_viewport([(0, 0), b0, b1]))

        r = range(-50, 51)
        pts = np.concatenate([[g0 * i + g1 * j for i in r] for j in r])
        x, y = zip(*pts)

        draw_arrow(ax, (0, 0), b0, f'b1 = [{b0[0]}, {b0[1]}]')
        draw_arrow(ax, (0, 0), b1, f'b2 = [{b1[0]}, {b1[1]}]')

        plt.plot([int(xx) for xx in x], [int(yy) for yy in y], 'ro')
        plt.plot([0], [0], 'bo')
        plt.draw()
        plt.pause(0.001)

        if basis_is_reduced(g0, b0, b1):
            print("=== YOU HAVE WON! ===\nClose plot to play again :)?")
            plt.ioff()
            plt.show()
            break

        b0, b1, cont = read_update(b0, b1)

    print(f"Final basis: b1 = {b0}, b2 = {b1}")


if __name__ == '__main__':
    main()
