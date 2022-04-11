import numpy as np
feature_size=x_train.shape[1]
agent = Agent(feature_size)
data = x_train
l = len(data)
batch_size = 32
episode_count = 1

for i in range(episode_count +1):
    print("Running episode "+str(i) + "/" + str(episode_count))
    #Initializating
    state = getState(x_train,0)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    #each episode
    for t in range(len(x_train)-1):
        #initializating
        reward = 0
        action = agent.act(state)
        next_state = getState(x_train,t+1)
        
        if action == 1:
            agent.inventory.append(y_train.iloc[t,:])
            states_buy.append(t)
            
        elif action == 2 and len(agent.inventory)>0:
            bought_price = agent.inventory.pop(0)
            reward = max(pd.DataFrame(y_train.iloc[t]-bought_price).values,0)
            total_profit += y_train.iloc[t]-bought_price
            states_sell.append(t)
        
        done = True if t == l-1 else False
        agent.memory.append((state,action,reward,next_state,done))
        state = next_state
        
        if done:
            print("-------------------------------")
            print('Total Profit: ' + formatPrice(total_profit))
            print("-------------------------------")
            
            plot_behavior(y_train,states_buy,states_sell,total_profit)
        if len(agent.memory)>batch_size:
            agent.expReplay(batch_size)
        
    if i%2 == 0:
        agent.model.save(r"C:\Users\Administrator\py\lehigh project\修改实验\model" + str(i))
    


print(agent.model.summary())