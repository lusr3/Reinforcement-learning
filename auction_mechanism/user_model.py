import math
import random
import numpy as np
from auction_model import AuctionModel

'''
Definition for constant.
SIGMA: The data size of the local model.
W: Channel bandwith.
CAP: Effective capacitance parameter of computing chipset.
TMAX: Maximum communication delay.
N0: Background noise.
GAMMA: Global accuracy.
VARPI: Parameter in computing global iterations.
'''
SIGMA = 1e3
W = 15e3
CAP = 1e-28
TMAX = 400
N0 = -174 * 1e-3
GAMMA = 0.5
VARPI = 10

def generate_global_iterations(accuracy):
    '''
    Generate global iteration times base on local accuracy.
    '''
    return math.ceil((VARPI * math.log(1 / GAMMA)) / (1 - accuracy))

def param_init():
    '''
    Init the param that user needs for deciding a bid.
    '''
    param = {
        'channel gain' : random.uniform(-95, -90),
        'cycle' : random.uniform(10, 50) * 1e3,
        'sample' : 8e7,
        'antenna max' : np.random.randint(10, 51),
        'subchannel max' : np.random.randint(10, 51),
        'transmission power max' : random.uniform(6, 10) * 1e-3,
        'transmission power min' : random.uniform(0, 2) * 1e-3,
        'computation frequency max' : random.uniform(3, 5) * 1e9,
        'computation frequency min' : random.uniform(0.1, 0.2) * 1e9,
        'RHO' : 1
    }
    return param

class UserModel(object):
    '''
    User model for deciding a bid.
    '''
    def __init__(self, param : dict) -> None:
        self.hn = param['channel gain']
        self.cn = param['cycle']
        self.sn = param['sample']
        self.an_max = param['antenna max']
        self.bn_max = param['subchannel max']
        self.pn_max = param['transmission power max']
        self.pn_min = param['transmission power min']
        self.fn_max = param['computation frequency max']
        self.fn_min = param['computation frequency min']
        self.rho = param['RHO']
        # accuracy for algorithm 1 and 2
        self.e1 = 1e-10
        self.e2 = 1e-20
        self.set()
    
    def set(self):
        '''
        Set the init communication power, computing frequency,
        antenna num, sub-channel num and local accuracy.
        '''
        self.pn = random.uniform(self.pn_min, self.pn_max)
        self.fn = random.uniform(self.fn_min, self.fn_max)
        self.an = self.an_max
        self.bn = self.bn_max
        self.accuracy = random.random()
    
    def generate_optimal_pn(self, gi):
        '''
        Given the global iterations gi, computing frequency fn,
        and local accuracy, get the optimal communication power
        to minimize the time and energy cost.
        Corresponding to Algorithm 1.
        '''
        phi = lambda x, y : (1 + x) * (math.log(1 + x)) - x - y
        theta = (self.an - 1) / (self.bn * N0 * W)
        A = math.log2(1 / self.accuracy) * self.cn * self.sn / self.fn
        B = SIGMA / ((TMAX / gi - A) * self.bn * W)
        pn_min = (2 ** B - 1) / (self.hn * theta)
        auxi = self.rho * theta * self.hn

        phi_pnmax = phi(theta * self.pn_max * self.hn, auxi)
        phi_pnmin = phi(theta * pn_min * self.hn, auxi)

        if phi_pnmax < 0:
            return self.pn_max
        elif phi_pnmin >= 0:
            return max(pn_min, self.pn_min)
        # the bisection method
        p1 = max(0, pn_min)
        p2 = self.pn_max
        while p2 - p1 >= self.e1:
            pu = (p1 + p2) / 2
            if phi(theta * pu * self.hn, auxi) <= 0:
                p1 = pu
            else:
                p2 = pu
        return (p1 + p2) / 2

    def generate_optimal_fn(self, gi):
        '''
        Given the global iterations gi, communication power pn,
        and local accuracy, get the optimal computing frequency
        to minimize the time and energy cost.
        '''
        # set the first derivative to 0 to get the optimal fn
        op_fn = (self.rho / (2 * CAP)) ** (1 / 3)
        
        rn = self.bn * W * math.log2(1 + ((self.an - 1) * self.pn * self.hn) / (self.bn * W * N0))
        fn_min = (self.sn * self.cn * math.log2(1 / self.accuracy)) / (TMAX / gi - SIGMA / rn)
        fn_min = min(fn_min, self.fn_min)

        if op_fn > self.fn_max:
            op_fn = self.fn_max
        elif op_fn < fn_min:
            op_fn = self.fn_min
        return op_fn
    
    def generate_optimal_accuracy(self, gamma1, gamma2):
        '''
        Given the global iterations gi, computing frequency fn,
        and communication power pn, get the optimal accuracy
        to minimize the time and energy cost.
        Corresponding to Algorithm 2.
        '''
        H = lambda x, y : gamma1 * math.log2(1 / y) + gamma2 - x * (1 - y)

        cur_acc = self.accuracy
        xi = random.random() * 1e7

        # loop to get optimal accuracy
        while True:
            pre_acc = cur_acc
            cur_acc = gamma1 / (math.log(2) * xi)
            # until meet the break condition or convergence
            if H(xi, cur_acc) >= 0 or abs(cur_acc - pre_acc) < self.e2:
                break
            xi = (gamma1 * math.log2(1 / cur_acc) + gamma2) / (1 - cur_acc)
        return cur_acc
    
    def get_energy_and_time(self):
        '''
        Get computing cost, communication cost,
        computing time and communication time.
        '''
        e_comp = CAP * self.cn * self.sn * self.fn * self.fn
        t_comp = self.cn * self.sn / self.fn
        rn = self.bn * W * math.log2(1 + ((self.an - 1) * self.pn * self.hn) / (self.bn * W * N0))
        e_com = (SIGMA * self.pn) / rn
        t_com = SIGMA / rn
        return e_comp, e_com, t_comp, t_com

    def bid_decide(self):
        '''
        User n decide one bid aim at minimizing
        the energy and time cost.
        Corresponding to Algorithm 3.
        '''
        # simplify the convergence condition to itreration times
        iterations = 10
        for _ in range(iterations):
            # first step
            gi = generate_global_iterations(self.accuracy)
            self.pn = self.generate_optimal_pn(gi)
            self.fn = self.generate_optimal_fn(gi)

            # second step
            a = VARPI * math.log2(1 / GAMMA)
            e_comp, e_com, t_comp, t_com = self.get_energy_and_time()
            gamma1 = a * (e_comp + t_comp)
            gamma2 = a * (e_com + t_com)
            self.accuracy = self.generate_optimal_accuracy(gamma1, gamma2)

    def send_bid(self, bs:AuctionModel, fo=0.01, yitaa=1, yitab=1):
        '''
        Send bid to the BS.
        '''
        cost = self.bn * fo * yitab + self.an * fo * yitaa
        bid = (self.accuracy, self.bn, self.an, cost)
        bs.add(bid)