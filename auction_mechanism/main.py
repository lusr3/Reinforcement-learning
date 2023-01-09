from user_model import *
from auction_model import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def subchannel_antenna_test():
    '''
    Test for varying antenna and sub-channel num 
    from 10 to 50 with step 10.
    Observe the local accuracy and energy cost changes.
    Corrsponding to Fig.2.
    '''
    acc_res = []
    cost_res = []
    param = param_init()
    for antenna_num in range(10, 60, 10):
        for channel_num in range(10, 60, 10):
            param['antenna max'] = antenna_num
            param['subchannel max'] = channel_num
            user_model = UserModel(param)
            user_model.bid_decide()

            accuracy = user_model.accuracy
            e_comp, e_com, _, _ = user_model.get_energy_and_time()
            gi = generate_global_iterations(accuracy)
            e_tol = math.log2(1 / accuracy) * e_comp + e_com
            acc_res.append(accuracy)
            cost_res.append(gi * e_tol)
            # print(antenna_num, channel_num, accuracy, gi * e_tol)
    # plot 
    fig = plt.figure()
    X = np.arange(10, 60, 10)
    Y = np.arange(10, 60, 10)
    Y, X = np.meshgrid(Y, X)
    R = acc_res
    # R = cost_res
    R = np.array(R).reshape(5, 5)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm)

    ax.set_xlabel("Amax")
    ax.set_ylabel("Bmax")
    # ax.set_zlim(0.85, 0.94)
    ax.set_zlabel("Local accuracy")
    # ax.set_zlim(4500, 4800)
    # ax.set_zlabel("cost")
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)
    plt.show()

def rho_test():
    '''
    Test for changing the weight between energy and time cost.
    Varying the parameter rho from 1 to 9 with step 2.
    Corresponding to the Fig.3.
    '''
    param = param_init()
    plt.grid(True)
    plt.grid(alpha=0.5) 
    for rho in range(1, 10, 2):
        param['RHO'] = rho
        user_model = UserModel(param)
        user_model.bid_decide()

        accuracy = user_model.accuracy
        e_comp, e_com, _, _ = user_model.get_energy_and_time()
        gi = generate_global_iterations(accuracy)
        e_tol = math.log2(1 / accuracy) * e_comp + e_com
        # print(rho, accuracy, gi * e_tol)
        plt.bar(rho, accuracy, color="forestgreen", hatch='/')
        # plt.bar(rho, gi * e_tol, color="red", hatch='/')
    plt.xlabel(r"$\rho$")
    plt.ylabel("Local accuracy")
    # plt.ylabel("Cost")
    # plt.ylim(0.80, 0.95)
    plt.show()

def generate_user_models(num):
    '''
    Generate ${num} user models for users auction test.
    '''
    models = []
    for _ in range(num):
        param = param_init()
        user_model = UserModel(param)
        user_model.bid_decide()
        models.append(user_model)
    return models

def user_test():
    '''
    Test for different solutions when user number increase.
    The number of users varying from 20 to 100 with step 20.
    Corresponding to Fig.4.
    '''
    models = generate_user_models(100)
    user_nums = np.arange(20, 110, 20)
    fo, yitaa, yitab = 0.01, 1, 1

    num = 5
    op_store = np.zeros(num)
    gred_store = np.zeros(num)
    fix_store = np.zeros(num)
    bs_store = np.zeros(num)
    lower_bound = np.random.randint(10, 15, len(user_nums))

    for idx, user_num in enumerate(user_nums):
        auction_moedel = AuctionModel()
        for user in range(user_num):
            user_model = models[user]
            user_model.send_bid(auction_moedel, fo, yitaa, yitab)
        
        auction_moedel.set(yitaa, yitab)
        bid_satisfactions, costs, channels, antennas = auction_moedel.get()
        normalized_vals = bid_satisfactions - costs
        op, _, _, _ = optimal_solution((normalized_vals, channels, antennas))
        gred, _, _, _ = auction_moedel.greedy_algorithm()
        fix, _, _, _ = fix_price_solution((bid_satisfactions, channels, antennas), 1, fo, yitaa, yitab)
        bs, _, _, _ = bs_max_utility_solution((bid_satisfactions, costs, channels, antennas))
        # print(user_num, op, gred, fix, bs)
        op_store[idx] = op
        gred_store[idx] = gred
        fix_store[idx] = fix
        bs_store[idx] = bs

    plt.plot(user_nums, op_store, color="gold",linestyle='--',marker='*', label="Optimal Solution")
    plt.plot(user_nums, gred_store, color="red",linestyle='-',marker='^', markerfacecolor='none', label="Greedy Agl.")
    plt.plot(user_nums, lower_bound, color="black", linestyle="--", marker="o", label="Lower bound")
    plt.xlabel("Users")
    plt.ylabel("Social Welfare")
    plt.grid(True)
    plt.grid(alpha=0.8) 
    plt.legend(bbox_to_anchor=(0.5, 1), loc=8, ncol=2)

    # plt.plot(user_nums, gred_store, color="red",linestyle='-',marker='^', markerfacecolor='none', label="Greedy Agl.")
    # plt.plot(user_nums, bs_store, color="green",linestyle='--',marker='s', label="Maximum Utility of BS")
    # plt.plot(user_nums, fix_store, color="blue",linestyle='--',marker='>', label="Fixed linear price scheme")
    # plt.xlabel("Users")
    # plt.ylabel("Social Welfare")
    # plt.grid(True)
    # plt.grid(alpha=0.8) 
    # plt.legend(bbox_to_anchor=(0.5, 1), loc=8, ncol=2)

    plt.show()

def price_test(yitaa, yitab):
    '''
    Test for changing the basic price for resources
    varying from 0.01 to 0.31 with step 0.03.
    Corresponding to Fig.5.
    '''
    num = 100
    models = generate_user_models(num)
    fos = np.arange(0.01, 0.33, 0.03)

    fo_num = 11
    gred_store = np.zeros(fo_num)
    linear_store = np.zeros(fo_num)
    sub_store = np.zeros(fo_num)
    super_store = np.zeros(fo_num)

    for idx, fo in enumerate(fos):
        auction_moedel = AuctionModel()
        for user in range(num):
            user_model = models[user]
            user_model.send_bid(auction_moedel, 0.01, yitaa, yitab)
        
        auction_moedel.set(yitaa, yitab)
        bid_satisfactions, costs, channels, antennas = auction_moedel.get()
        normalized_vals = bid_satisfactions - costs
        gred, _, _, _ = auction_moedel.greedy_algorithm()
        linear, _, _, _ = fix_price_solution((bid_satisfactions, channels, antennas), 1, fo, yitaa, yitab)
        sub_linear, _, _, _ = fix_price_solution((bid_satisfactions, channels, antennas), 0.85, fo, yitaa, yitab)
        super_linear, _, _, _ = fix_price_solution((bid_satisfactions, channels, antennas), 1.15, fo, yitaa, yitab)
        # print(gred, linear, sub_linear, super_linear)
        gred_store[idx] = gred
        linear_store[idx] = linear
        sub_store[idx] = sub_linear
        super_store[idx] = super_linear
    
    plt.plot(fos, gred_store, color="blue",linestyle='--',marker='^', markerfacecolor='none', label="Greedy Agl.")
    plt.plot(fos, linear_store, color="green",linestyle='--',marker='*', label="fixed linear price scheme")
    plt.plot(fos, sub_store, color="black",linestyle='-',marker='>', label="fixed sublinear price scheme")
    plt.plot(fos, super_store, color="red",linestyle='-',marker='v', markerfacecolor='none', label="fixed superlinear price scheme")
    plt.xlabel("Set of Price")
    plt.ylabel("Average Social Welfare")
    plt.grid(True)
    plt.grid(alpha=0.8) 
    plt.legend(bbox_to_anchor=(0.5, 1), loc=8, ncol=2)

    plt.show()

def normalized(std, x1, x2, x3):
    '''
    Normalized based on the standard value.
    '''
    return 1, x1 / std, x2 / std, x3 / std

def multiple_metric_comparison(yitaa, yitab):
    '''
    Test for multiple metrics about:
    social welfare, antenna utilization,
    channel utilization and winner precentage.
    Corresponding to Fig.6.
    '''
    num = 50
    models = generate_user_models(num)
    auction_moedel = AuctionModel()
    for user in range(num):
        user_model = models[user]
        user_model.send_bid(auction_moedel, 0.01, yitaa, yitab)
    
    auction_moedel.set(yitaa, yitab)
    bid_satisfactions, costs, channels, antennas = auction_moedel.get()
    normalized_vals = bid_satisfactions - costs
    op, op_a, op_b, op_num = optimal_solution((normalized_vals, channels, antennas))
    gred, gred_a, gred_b, gred_num = auction_moedel.greedy_algorithm()
    linear, linear_a, linear_b, linear_num = fix_price_solution((bid_satisfactions, channels, antennas), 1, 0.01, yitaa, yitab)
    bs, bs_a, bs_b, bs_num = bs_max_utility_solution((bid_satisfactions, costs, channels, antennas))
    # print(op, gred, bs, linear)
    # print(op_a, gred_a, bs_a, linear_a)
    # print(op_b, gred_b, bs_b, linear_b)
    # print(op_num, gred_num, bs_num, linear_num)

    total_width, n = 0.8, 4
    width = total_width / n
    x = np.arange(1, 5, 1)
    x1 = x - width * 1.5
    x2 = x - width * 0.5
    x3 = x + width * 0.5
    x4 = x + width * 1.5
    op, gred, bs, linear = normalized(op, gred, bs, linear)
    op_a, gred_a, bs_a, linear_a = normalized(op_a, gred_a, bs_a, linear_a)
    op_b, gred_b, bs_b, linear_b = normalized(op_b, gred_b, bs_b, linear_b)
    op_num, gred_num, bs_num, linear_num = normalized(op_num, gred_num, bs_num, linear_num)

    plt.bar(x1, [op, op_a, op_b, op_num], width=width, label="Optimal Solution")
    plt.bar(x2, [gred, gred_a, gred_b, gred_num], width=width, label="Greedy Alg.")
    plt.bar(x3, [bs, bs_a, bs_b, bs_num], width=width, label="Maximum Utility of BS")
    plt.bar(x4, [linear, linear_a, linear_b, linear_num], width=width, label="Fixed Scheme")
    plt.xticks(x, ["Social Welfare", "Antenna Utilization", "Channel Utilization", "Winner Percentage"])
    plt.legend(bbox_to_anchor=(0.5, 1), loc=8, ncol=2)
    plt.show()

if __name__ == "__main__":
    random.seed(213)
    np.random.seed(213)
    # subchannel_antenna_test()
    # rho_test()
    # user_test()
    # price_test(1, 0.5)
    multiple_metric_comparison(1, 2)