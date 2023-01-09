'''
To simplify the implementation,
we assume that every user just gives one bid.
'''

from user_model import *

'''
Definition for constant.
AMAX: Max num for antenna num.
BMAX: Max num for sub-channel num.
TAO: Parameter in computing the satisfaction for bs.
'''
AMAX = 100
BMAX = 100
TAO = 8

def optimal_solution(data):
    '''
    Optimal solution using dynamic programming.
    '''

    normalized_vals, channels, antennas = data
    n = len(channels)
    
    dp = [[0 for j in range(BMAX)] for i in range(AMAX)]
    suma = [[0 for j in range(BMAX)] for i in range(AMAX)]
    sumb = [[0 for j in range(BMAX)] for i in range(AMAX)]
    users = [[0 for j in range(BMAX)] for i in range(AMAX)]

    for k in range(n):
        for antenna_num in range(AMAX):
            if antenna_num - antennas[k] < 0:
                continue
            for channel_num in range(BMAX):
                if channel_num - channels[k] < 0:
                    continue
                pre_val = dp[antenna_num][channel_num]
                cur_val = dp[antenna_num - antennas[k]][channel_num - channels[k]] + normalized_vals[k]
                if cur_val > pre_val:
                    dp[antenna_num][channel_num] = cur_val
                    suma[antenna_num][channel_num] = suma[antenna_num - antennas[k]][channel_num - channels[k]] + antenna_num
                    sumb[antenna_num][channel_num] = sumb[antenna_num - antennas[k]][channel_num - channels[k]] + channel_num
                    users[antenna_num][channel_num] = users[antenna_num - antennas[k]][channel_num - channels[k]] + 1
    return dp[AMAX - 1][BMAX - 1], suma[AMAX - 1][BMAX - 1], sumb[AMAX - 1][BMAX - 1], users[AMAX - 1][BMAX - 1]

def fix_price_solution(data, exp, fo, yitaa, yitab):
    '''
    Fix price scheme for three types:
    linear, sub-linear and super linear.
    The difference reflects in the exp.
    '''

    fa = fo * yitaa ** exp
    fb = fo * yitab ** exp
    bid_satisfactions, channels, antennas = data
    Fn = channels * fb + antennas * fa
    normalized_vals = bid_satisfactions - Fn
    n = len(channels)
    social_welfare = 0
    suma, sumb = 0, 0
    num = 0

    # first come, first served
    for idx in range(n):
        if normalized_vals[idx] >= Fn[idx] and suma + antennas[idx] <= AMAX \
            and sumb + channels[idx] <= BMAX:
            suma += antennas[idx]
            sumb += channels[idx]
            social_welfare += normalized_vals[idx]
            num += 1
    return social_welfare, suma, sumb, num

def bs_max_utility_solution(data):
    '''
    Solution for maximum the bs utility.
    '''

    bid_satisfactions, costs, channels, antennas = data
    normalized_vals = bid_satisfactions - costs
    social_welfare = 0
    suma, sumb = 0, 0
    index = np.flip(np.argsort(bid_satisfactions))
    num = 0

    for idx in index:
        if suma + antennas[idx] <= AMAX and sumb + channels[idx] <= BMAX:
            suma += antennas[idx]
            sumb += channels[idx]
            social_welfare += normalized_vals[idx]
            num += 1
    return social_welfare, suma, sumb, num

class AuctionModel(object):
    '''
    Auction moedel for BS to decide winners.
    '''
    def __init__(self) -> None:
        self.user_accuracy = []
        self.user_subchannel = []
        self.user_antenna = []
        self.user_cost = []
        self.user_winner = []
    
    def add(self, bid):
        '''
        Add a bid to BS.
        '''
        acc, channel, antenna, cost = bid
        self.user_accuracy.append(acc)
        self.user_subchannel.append(channel)
        self.user_antenna.append(antenna)
        self.user_cost.append(cost)
        self.user_winner.append(False)
    
    def set(self, yitaa, yitab):
        '''
        Transform the type to numpy array
        and get useful information.
        '''
        transform = lambda x : np.array(x)
        self.user_accuracy = transform(self.user_accuracy)
        self.user_subchannel = transform(self.user_subchannel)
        self.user_antenna = transform(self.user_antenna)
        self.user_cost = transform(self.user_cost)
        self.user_winner = transform(self.user_winner)
        self.bid_satisfaction = TAO / self.user_accuracy
        self.source = self.user_subchannel * yitab + self.user_antenna * yitaa
        self.normalized_value = self.bid_satisfaction - self.user_cost

    def get(self):
        '''
        Get basic information of bid from BS.
        '''
        return self.bid_satisfaction, self.user_cost, self.user_subchannel, self.user_antenna

    def greedy_algorithm(self):
        '''
        Greedy algorithm for deciding the winning bids.
        Corresponding to Algorithm 4.
        '''
        
        winner = set()
        N = set(np.arange(0, len(self.user_winner), 1))
        social_welfare, suma, sumb = 0, 0, 0
        index = np.flip(np.argsort(self.normalized_value / self.source))
        i = 0

        while len(N) != 0:
            idx = index[i]
            if self.user_subchannel[idx] + sumb <= BMAX and self.user_antenna[idx] + suma <= AMAX:
                suma += self.user_subchannel[idx]
                sumb += self.user_antenna[idx]
                self.user_winner[idx] = True
                social_welfare += self.normalized_value[idx]
                winner.add(idx)
                N.remove(idx)
            else:
                break
            i += 1
        return social_welfare, suma, sumb, len(winner)
