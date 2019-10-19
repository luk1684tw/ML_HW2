import argparse
import numpy as np

from scipy.special import comb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--win', type=int)
    parser.add_argument('-b', '--lose', type=int)
    args = parser.parse_args()

    prior = np.array([args.win, args.lose], dtype=np.int)
    posterior = np.zeros((2))
    testcase = open('./testcase.txt', 'r')
    trails = list()
    for line in testcase:
        trails.append(line.strip())

    for i, trail in enumerate(trails):
        m, l = 0, 0
        for case in trail:
            if case == '0':
                l += 1
            else:
                m += 1
        p = m / (m + l)
        likelihood = comb(m+l, m) * p**m * (1-p)**l
        
        posterior = prior + np.array([m, l])
        print (f'case {i + 1}: {trail}')
        print (f'Likelihood: {likelihood}')
        print (f'Beta prior:     a = {prior[0]} b = {prior[1]}')
        print (f'Beta posterior: a = {posterior[0]} b = {posterior[1]}\n')

        prior = posterior


    