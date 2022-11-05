import random as rd
# 定义学习率、折扣因子和迭代次数
ALPHA = 0.5
GAMMA = 1
EPISODES = 200
# 定义基本信息
start = (0, 0)
goal = (0, 11)
cliff = set([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)])
dx = [1, -1, 0, 0]
dy = [0, 0, -1, 1]
dire = {0 : '↑', 1 : '↓', 2 : '←', 3 : '→'}

# 根据 e-greedy 得到 s 状态下的动作 a
def get_action(q, s, e):
    p = rd.random()
    if p < e:
        a = rd.randint(0, 3)
    else:
        a = q[s].index(max(q[s]))
    return a

def get_state_reward(s, a):
    nx, ny = s[0] + dx[a], s[1] + dy[a]
    # cliff
    if (nx, ny) in cliff:
        new_state = start
        reward = -100
    elif nx < 0 or nx > 3 or ny < 0 or ny > 11:
        new_state = s
        reward = -1
    elif (nx, ny) == goal:
        new_state = (nx, ny)
        reward = 0
    else:
        new_state = (nx, ny)
        reward = -1
    return new_state, reward

def sarsa(q):
    episode = 0
    e = 1
    while episode < EPISODES:
        s = start
        a = get_action(q, s, e)
        while s != goal:
            ns, r = get_state_reward(s, a)
            na = get_action(q, ns, e)
            # 更新 Q table
            q[s][a] += ALPHA * (r + GAMMA * q[ns][na] - q[s][a])
            s = ns
            a = na
        episode += 1
        e = 1 / episode


def plot(q):
    graph = [[0 for i in range(12)] for j in range(4)]
    graph[goal[0]][goal[1]] = 'G'
    for pos in cliff:
        graph[pos[0]][pos[1]] = '*'
    s = start
    while s != goal:
        a = q[s].index(max(q[s]))
        graph[s[0]][s[1]] = dire[a]
        ns, r = get_state_reward(s, a)
        s = ns
    for i in range(3, -1, -1):
        for j in range(12):
            print(graph[i][j], end=' ')
        print('\n', end='')

def main():
    # 初始化 Q table
    q = dict()
    for i in range(0, 4):
        for j in range(0, 12):
            q[(i, j)] = [0, 0, 0, 0]
    sarsa(q)
    plot(q)

if __name__ == '__main__':
    main()