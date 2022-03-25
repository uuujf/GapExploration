# import gym
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

class TreeWorld(object):
    def __init__(self, H=2, A=10, S=100, rho=0.2):
        super(TreeWorld, self).__init__()

        self.init = 0
        self.H = H
        self.A = A
        self.S = S
        self.rho=rho

        self._transit()

    def _transit(self):
        self.P = np.zeros([self.S, self.A, self.S])

        subP = np.random.rand(self.S, self.A, self.S-1)
        subP[:,:,0] = 0.0 # prevent MDP from returning to initial state
        subP /= np.sum(subP, axis=(2), keepdims=True)
        self.P[:,:,:self.S-1] = subP

        # one action that has gap
        self.P[0, self.A-1, :] *= 0.0
        self.P[0, self.A-1, self.S-1] = self.rho
        self.P[0, self.A-1, 1:self.S-1] = (1.0 - self.rho)/(self.S-2)

        self.P[self.S-1, :, :] = 0.0
        self.P[self.S-1, self.A-1, self.S-1] = 1.0
        self.P[self.S-1, :self.A-1, 1:self.S-1] = 1/(self.S-2)

        return True

    def _reward(self, hard="gap"):
        # (S, A)
        reward = np.zeros([self.S, self.A])
        if hard == "gap":
            reward[self.S-1] += 1.0
        elif hard == "hard":
            reward[self.S-5] += 1.0
        else:
            raise Exception("Not Defined")
        return reward

    def play(self, reward, policy):
        # reward: (S, A)
        # policy: (H, S, A)
        trajectory = []
        x = 0
        V = 0
        for h in range(self.H):
            a = np.random.choice(self.A, p=policy[h,x,:])
            y = np.random.choice(self.S, p=self.P[x,a,:])
            V += reward[x, a]
            trajectory.append( (x,a,y) )
            x = y
        return V, trajectory

    def _V_pi_0(self, reward, policy):
        # reward: (S, A)
        # policy: (H, S, A)
        p_x = np.zeros(self.S)
        p_x[0] = 1
        V = 0
        for h in range(self.H):
            p_xa = policy[h] * p_x.reshape((self.S, 1)) # (S, A)
            p_y = np.sum(p_xa.reshape((self.S, self.A, 1)) * self.P, axis=(0,1))
            V += np.sum(p_xa * reward)
            p_x = np.copy(p_y)
            # print(p_y.shape, p_y.sum())
        return V

    def _Q_opt(self, reward):
        # reward: (S, A)
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros(self.S)
        for h in range(self.H-1, -1, -1):
            Q[h] = reward + np.sum(self.P * V.reshape((1, 1, self.S)), axis=2)
            Q[h] = np.minimum(Q[h], self.H)
            V = np.copy(np.amax(Q[h], axis=1))
            # pi[h] = np.eye(self.A)[np.argmax(Q[h], axis=1)]
        return Q

    def _gap(self, reward):
        # reward: (S, A)
        Q = self._Q_opt(reward)
        V = np.amax(Q, axis=2, keepdims=True)
        gap_table = V - Q
        gap_table[0, 1:, :] *= 0.0 # initial state is 0
        gap = np.min(gap_table[np.nonzero(gap_table)])
        return gap

    def _V_opt_0(self, reward):
        Q = self._Q_opt(reward)
        return np.max(Q[0,0])

    def regret(self, reward, policy_list):
        # reward: (S, A)
        # policy: (H, S, A)
        V_opt_0 = self._V_opt_0(reward)
        regret = 0
        regrets = []
        for policy in policy_list:
            V_pi_0 = self._V_pi_0(reward, policy)
            regret += V_opt_0 - V_pi_0
            regrets.append(regret)
        return regrets

    def planning_error(self, reward, policy_list):
        # reward: (S, A)
        # policy: (H, S, A)
        regrets = self.regret(reward, policy_list)
        errors = [regrets[i] / (i+1) for i in range(len(regrets))]
        return errors


class UCBVI(object):
    def __init__(self, ENV, clip=0.0):
        super(UCBVI, self).__init__()

        self.ENV = ENV
        self.A = self.ENV.A
        self.S = self.ENV.S
        self.H = self.ENV.H

        self.clip = clip


        self.histogram = np.zeros([self.S, self.A, self.S])

    def _update_history(self, episode_trajectory):
        for (x,a,y) in episode_trajectory:
            self.histogram[x,a,y] += 1

    def _empi_transit(self):
        # (S, A, S)
        N_SA = np.tile( self.histogram.sum(axis=2, keepdims=True), (1,1,self.S)) # (S, A, S)
        NonZeros = (N_SA != 0)
        empi_transit = np.ones_like(self.histogram) / self.S
        empi_transit[NonZeros] = self.histogram[NonZeros] / N_SA[NonZeros]
        return np.copy(empi_transit)

    def _planning_bonus(self):
        # logarithmic factors are ignored
        # (S, A)
        bonus = 1.0*np.sqrt( self.H**2 / 2.0 / np.maximum(1, np.sum(self.histogram, axis=2)) )
        return bonus

    def _exploration_bonus(self):
        # logarithmic factors are ignored
        # lower order terms are ignored
        # (S, A)
        # bonus1 = np.sqrt( 8.0 * self.H**2 / np.sum(self.histogram, axis=2) )
        # bonus1 = bonus1 * (bonus1 > self.clip)
        # bonus2 = 120.0 * (self.S + self.H) * self.H **3  / np.sum(self.histogram, axis=2)
        # bonus3 = 240.0 * (self.H ** 6 * self.S**2) / ( np.sum(self.histogram, axis=2)**2 )
        bonus1 = np.sqrt( 1.0 * self.H**2 / np.maximum(1, np.sum(self.histogram, axis=2)) )
        # if np.sum(bonus1 <= self.clip) > 0:
        #     print('is clipping')
        bonus1 = bonus1 * (bonus1 > self.clip)
        bonus2 = 1.0 * (self.S + self.H)  / np.maximum(1, np.sum(self.histogram, axis=2))
        # bonus3 = 1.0 * (self.H ** 1 * self.S) / (np.maximum(1, np.sum(self.histogram, axis=2) )**2 )
        bonus3 = 0
        # bonus2 = (self.S + self.H) * self.H **3  / np.sum(self.histogram, axis=2)
        # bonus3 = 240.0 * (self.H ** 6 * self.S**2) / ( np.sum(self.histogram, axis=2)**2 )
        return bonus1 + bonus2 + bonus3

    def _Q_ucb(self, reward, bonus):
        # reward: (S, A)
        # bonus: (S, A)
        trans = self._empi_transit() # (S, A, S)

        pi = np.zeros((self.H, self.S, self.A))
        Q = np.zeros((self.H, self.S, self.A))
        V = np.zeros(self.S)
        for h in np.arange(self.H-1, -1, -1):
            Q[h] = reward + bonus + np.sum(trans * V.reshape((1, 1, self.S)), axis=2)
            Q[h] = np.minimum(Q[h], self.H)
            V = np.amax(Q[h], axis=1)
            pi[h] = np.eye(self.A)[np.argmax(Q[h], axis=1)]
        return Q, pi

    def unsupervised_exploration(self, K, reward):
        # reward: (S, A)
        BB = []
        Pis = []
        r0 = np.zeros_like(reward) # zero reward
        for _ in range(K):
            # exploration
            bonus = self._exploration_bonus()
            BB.append(bonus)
            _, policy = self._Q_ucb(r0, bonus)
            _, trajectory = self.ENV.play(r0, policy)
            self._update_history(trajectory)

            # planning
            bonus = self._planning_bonus()
            _, policy = self._Q_ucb(reward, bonus)
            Pis.append(np.copy(policy))
        return Pis, BB

if __name__ == '__main__':
    np.random.seed(1)

    H = 5
    A = 10
    S = 10
    rho=0.4
    env = TreeWorld(H, A, S, rho)
    r = env._reward(hard="gap")
    print(env._gap(r))
    # print(env._gap(env._reward(hard="hard")))


    K = 50000
    algo_gap = UCBVI(env, clip=rho/H)
    # algo_0 = UCBVI(env, clip=0.0)
    # algo_1 = UCBVI(env, clip=1.0)

    pis_gap, BB_gap = algo_gap.unsupervised_exploration(K, r)
    regrets_gap = env.regret(r, pis_gap)
    errors_gap = env.planning_error(r, pis_gap)
    # print(errors_gap)


    plt.plot(range(K), errors_gap, "-r")
    # plt.plot(range(K), errors_0, "-b")
    # plt.plot(range(K), errors_1, "-g")

    baseline = np.sqrt( 4 / np.arange(1,K+1)) * 10
    plt.plot(range(K), baseline, "--g")

    plt.yscale("log", base=2)
    plt.ylim(top=2.0)
    # plt.xlim(left=0.0)
    plt.xlabel(r"numer of episodes", fontsize=15)
    plt.ylabel(r"planning error", fontsize=15)
    plt.legend([r"UCB-Clip, $\widetilde{\mathcal{O}}(1/k)$", r"A minimax rate, $\Theta(1 / \sqrt{k})$"], fontsize=15)

    plt.savefig("error.pdf")
