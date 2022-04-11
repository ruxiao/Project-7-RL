#Testing the Data
#agent is already defined in the training set above.
test_data = x_test
l_test = len(test_data) - 1
state = getState(test_data, 0)
total_profit = 0
is_eval = True
done = False
states_sell_test = []
states_buy_test = []
#Get the trained model
model_name = r'C:\Users\Administrator\LehighProject\修改实验\model4'
agent = Agent(x_test.shape[1], is_eval, model_name)
state = getState(test_data, 0)
total_profit = 0
agent.inventory = []

for t in range(l_test):
    action = agent.act(state)
    #print(action)
    #set_trace()
    next_state = getState(test_data, t + 1)
    reward = 0

    if action == 1:
        agent.inventory.append(y_test.iloc[t,:])
        states_buy_test.append(t)
        print("Buy: " + formatPrice(y_test.iloc[t]))

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        #reward = max(test_data[3][t] - bought_price, 0)
        reward = max(pd.DataFrame(y_test.iloc[t,:]).values - bought_price,0)
        total_profit += y_test.iloc[t,:] - bought_price
        states_sell_test.append(t)
        print("Sell: " + formatPrice(y_test.iloc[t,:]) + " | profit: " + formatPrice(y_test.iloc[t,:] - bought_price))

    if t == l_test - 1:
        done = True
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("------------------------------------------")

plot_behavior(y_test,states_buy_test, states_sell_test, total_profit)