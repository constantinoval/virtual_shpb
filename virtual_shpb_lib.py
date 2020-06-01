import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.style.use('bmh')

def runningMeanFast(x, N):
    rez = np.roll(np.convolve(x, np.ones((N,))/N)[(N-1):], N//2)
    rez[:N//2] = rez[N//2]
    return rez

class virtual_SHPB:
    def __init__(self, striker={}, input_bar={},
                 output_bar={}, sample={}):
        self.striker = striker
        self.input_bar = input_bar
        self.output_bar = output_bar
        self.sample = sample
        self.check_bars()
        self.check_striker()
        self.check_sample()

    def check_striker(self):
        self.striker['V'] = self.striker.get('V', 10)
        self.striker['L'] = self.striker.get('L', 200e-3)
        self.striker['c'] = self.input_bar['c']
        self.striker['D'] = self.input_bar['D']

    def check_bars(self):
        for bar in [self.input_bar, self.output_bar]:
            bar['L'] = bar.get('L', 1)
            bar['gauge_pos'] = bar.get('gauge_pos', bar['L']/2.)
            if bar['gauge_pos'] < 0 or bar['gauge_pos'] > bar['L']:
                bar['gauge_pos'] = bar['L']/2.
            bar['E'] = bar.get('E', 200e9)
            bar['rho'] = bar.get('rho', 7850.)
            bar['c'] = np.sqrt(bar['E']/bar['rho'])
            bar['D'] = bar.get('D', 20e-3)
            bar['S'] = bar['D']**2/4.*np.pi
            # if self.striker['L'] > bar['L']:
            #     self.striker['L'] = bar['']

    def check_sample(self):
        self.sample['L'] = self.sample.get('L', 10e-3)
        self.sample['D'] = self.sample.get('D', 20e-3)
        self.sample['s0'] = self.sample.get('s0', 200e6)
        self.sample['Et'] = self.sample.get('Et', 1000e6)
        self.sample['S'] = self.sample['D']**2/4.*np.pi
        self.sample['s'] = lambda e: self.sample['s0']+self.sample['Et']*e

    def calc_pulses(self, numpoints=100):
        eI0 = self.striker['V']/2.0/self.striker['c']
        duration = 2.0 * self.striker['L'] / self.striker['c']
        self.pulse_duration = duration
        def eIf(x): return eI0
        t = [0]
        deps = [0]
        eps = [0]
        sg = [0]
        eI = []
        eR = []
        eT = []
        L = self.sample['L']
        S = self.sample['S']
        dt = duration / numpoints
        self.dt = dt
        while t[-1] < duration:
            L *= (1+deps[-1]*dt)
            S = self.sample['S']*self.sample['L']/L
            eI.append(eIf(t[-1]))
            eet = min(eI[-1] * self.input_bar['S'] * self.input_bar['E'] /
                      self.output_bar['S'] / self.output_bar['E'],
                      self.sample['s'](eps[-1])
                      * S/self.output_bar['S']/self.output_bar['E'])
            eT.append(eet)
            eR.append(eI[-1] - eet * self.output_bar['E'] *
                      self.input_bar['S'] / self.input_bar['E']
                      / self.input_bar['S'])
            deps.append((self.input_bar['c'] * (eI[-1] + eR[-1])
                         - self.output_bar['c']*eT[-1])/L)
            eps.append(eps[-1]+deps[-1]*dt)
            sg.append(self.sample['s'](eps[-1]))
            t.append(t[-1] + dt)
        eI.append(eI[-1])
        eR.append(eR[-1])
        eT.append(eT[-1])
        self.t = t
        self.eps = eps
        self.deps = deps
        self.s = sg
        self.e_i = eI
        self.e_r = eR
        self.e_t = eT

    def calc_bars_signals(self):
        total_duration = self.input_bar['L'] / self.input_bar['c'] +\
            max(self.input_bar['L'] / self.input_bar['c'],
                self.output_bar['L'] / self.output_bar['c']) +\
            self.pulse_duration*1.1
        N = np.int(total_duration / self.dt)
        t = np.linspace(0, total_duration, N)
        b1 = np.zeros(N)
        b2 = np.zeros(N)
        L1 = self.input_bar['L']-self.input_bar['gauge_pos']
        L2 = self.input_bar['gauge_pos']
        L3 = self.output_bar['gauge_pos']
        L4 = self.output_bar['L']-self.output_bar['gauge_pos']
        b1 += np.interp(t, np.array(self.t) + L1 /
                        self.input_bar['c'], self.e_i, left=0, right=0)
        b1 -= np.interp(t, np.array(self.t) + (L1+2*L2) /
                        self.input_bar['c'], self.e_r, left=0, right=0)
        b2 -= np.interp(t, np.array(self.t) + (L1+L2) /
                        self.input_bar['c'] + L3 / self.output_bar['c'],
                        self.e_t, left=0, right=0)
        b2 += np.interp(t, np.array(self.t) + (L1+L2) /
                        self.input_bar['c'] + (L3+2*L4) / self.output_bar['c'],
                        self.e_t, left=0, right=0)
        self.tot_t = t
        self.b1 = b1
        self.b2 = b2

    def plot_setup(self):
        f = plt.figure()
        L1 = self.input_bar['L']-self.input_bar['gauge_pos']
        L2 = self.input_bar['gauge_pos']
        L3 = self.output_bar['gauge_pos']
        L4 = self.output_bar['L'] - self.output_bar['gauge_pos']
        Ls = self.sample['L']
        striker = plt.Rectangle((-self.striker['L'], -self.striker['D'] / 2),
                                self.striker['L'], self.striker['D'])
        b1 = plt.Rectangle((0, -self.input_bar['D'] / 2),
                           L1+L2, self.input_bar['D'], facecolor='r')
        b2 = plt.Rectangle((L1+L2+Ls, -self.output_bar['D'] / 2),
                           L3+L4, self.output_bar['D'], facecolor='m')
        s = plt.Rectangle((L1+L2, -self.sample['D'] / 2),
                          Ls, self.sample['D'], facecolor='g')
        plt.gca().add_artist(striker)
        plt.gca().add_artist(b1)
        plt.gca().add_artist(b2)
        plt.gca().add_artist(s)
        plt.axvline(L1, color='k', lw=1)
        plt.axvline(L1+L2+Ls+L3, color='k', lw=1)
        plt.xlim(-self.striker['L']*1.1, (self.input_bar['L'] +
                                          self.output_bar['L']+self.sample['L'])*1.1)
        D = max(self.striker['D'], self.input_bar['D'], self.output_bar['D'])
        plt.ylim(-D, D)
        return plt.gcf(), plt.gca()

    def add_noise(self, a=1e-4):
        self.b1 += np.random.random_sample(len(self.b1))*2*a-a
        self.b2 += np.random.random_sample(len(self.b2))*2*a-a


if __name__ == '__main__':
    shpb = virtual_SHPB()
    shpb.striker['V'] = 30
    shpb.output_bar['E'] = 70e9
    shpb.striker['L'] = 0.3
    shpb.output_bar['rho'] = 2600
    shpb.output_bar['gauge_pos'] = 0.1
    shpb.check_bars()
    shpb.calc_pulses()
    shpb.calc_bars_signals()
    shpb.add_noise(3e-4)
    plt.plot(shpb.tot_t*1e6, shpb.b1)
    plt.plot(shpb.tot_t * 1e6, shpb.b2)
    f, ax = shpb.plot_setup()
    plt.show()
