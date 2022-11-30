import matplotlib.pyplot as plt

x = []
y1 = []
y2 = []
with open(r'rewards.txt.bak', encoding ="utf-8" ) as file:
        for line in file:
            temp = line.replace('\n', '').split()
            x.append(int(temp[0]))
            y1.append(float(temp[2]))
with open(r'rewards_ddqn.txt', encoding ="utf-8" ) as file:
        for line in file:
            temp = line.replace('\n', '').split()
            y2.append(float(temp[2]))
plt.plot(x, y1, color='r', label="original")
plt.plot(x, y2, color='b', label="double-dqn")
plt.xlabel('step')
plt.ylabel('average_reward')
plt.legend()
plt.show()