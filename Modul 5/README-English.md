# Module 5: Reinforcement Learning

## Table of Contents
- [Module 5: Reinforcement Learning](#module-5-reinforcement-learning)
    - [Table of Contents](#table-of-contents)
    - [Terminology](#terminology)
    - [Introduction](#introduction)
    - [Reinforcement Learning Approaches](#reinforcement-learning-approaches)
    - [Algorithms](#algorithms)
        - [Q-Learning](#q-learning)
        - [NEAT Algorithm](#neat-algorithm)


## Terminology
- `Agent`: has the task of achieving the goal
- `Environment`: provides feedback on the actions taken by the Agent
- `Current State` ($s$): the current condition or situation from the Agent's perspective
- `Next State` ($s'$): the next condition or situation after the Agent takes an action
- `Goal`: the objective that the Agent wants to achieve
- `Action` (a): the action that the Agent will choose to achieve the goal
- `Policy` ($\pi$): the strategy/policy used by the Agent to choose actions
- `Reward` ($R$): a value to measure the success of the Agent's action
- `Penalty`: a value to measure the failure of the Agent's action

## Introduction
Reinforcement Learning (RL) is a technique in *machine learning* that studies how an agent should act in an environment to maximize the received reward. RL works based on the concept of **trial and error**, where the agent explores various actions, receives feedback in the form of rewards or penalties, and adjusts its strategy to achieve optimal goals.

In fact, RL is widely used in various applications such as *games*, *robotics*, *recommendation systems*, *search engines*, and others.

> For this module, we will implement RL in a simple *game* 🎮

## Reinforcement Learning Approaches
There are several approaches that can be used in RL, including:
1. **Value-Based**: Determining the policy indirectly by **learning the action value function** $Q(s, a)$ and choosing the action with the highest value.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

- **$Q(s, a)$** = How well the Agent takes action ($a$) in a state ($s$).  
- **$\alpha$** = Learning rate of the value function.
- **$r$** = Reward received after taking action ($a$).  
- **$\gamma$** = Discount factor that determines how far the Agent considers future rewards.  
- **$\max_{a'} Q(s', a')$** = The best value that can be obtained from the next state ($s'$).  

2. **Policy-Based**: **Learning the policy $\pi(a | s)$** without the need for a value function, using optimization methods such as *gradient ascent*.  

$$
\nabla J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta} (a | s) R \right]
$$

- **$J(\theta)$** = Objective function in the form of reward to be maximized.  
- **$\theta$** = Parameters of the policy ($\pi$)  
- **$\mathbb{E} [\cdot]$** = Average expectation of samples obtained during exploration.
- **$\nabla_{\theta} \log \pi_{\theta} (a | s)$** = Gradient of the policy logarithm.
- **$\pi_{\theta} (a | s)$** = Probability of choosing action ($a$) in state ($s$).  
- **$R$** = Total reward obtained after taking action ($a$). 

> Reference: [REINFORCE Algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

3. **Model-Based** builds an **environment transition model** $P(s' | s, a)$ to predict the next state before the agent makes a decision.

$$
s' \sim P(s' | s, a)
$$

- **$s'$** = Next state after the Agent takes action ($a$) in state ($s$).  
- **$\sim$** = ($s'$) is randomly taken from the probability distribution $P(s' | s, a)$.  
- **$P(s' | s, a)$** = Transition probability from state ($s$) to state ($s'$) after taking action ($a$).  

## Algorithms
### Q-Learning
Q-Learning is one of the algorithms in RL that falls under the **Value-Based** category. This algorithm learns the action value function $Q(s, a)$ to maximize the reward obtained by the Agent.

**Implementation Example**
For the implementation, you can check the code [taxi.py](/Modul%205/Q_Learning/taxi.py). This code is an implementation of Q-Learning in the Taxi-v3 *game* from OpenAI Gym. The goal of this code is to train the Agent to pick up passengers and deliver them to their destinations efficiently.

![Taxi-v3](/Modul%205/assets/taxi_image.png)

```python
for i in range(training_episodes):
    state, _ = env.reset()
    done = False
    penalties, reward = 0, 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = numpy.max(q_table[next_state])

        # Update q-value
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # According to Q-Learning formula
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        
    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
```

### NEAT Algorithm
NEAT (NeuroEvolution of Augmenting Topologies) is an algorithm that combines *genetic algorithm* and *neural network* to find the optimal neural network architecture. Well, actually NEAT is not an RL algorithm, but this algorithm can train Agents in an RL environment.

> Reference: [NEAT Algorithm](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

**Implementation Example**
For the implementation, you can check the code [neat_car.py](/Modul%205/NEAT/neat_car.py). The goal of this code is to train a car to navigate a track without hitting the walls.

![NEAT](/Modul%205/assets/neat_car_image.png)

```python
def run_car(genomes, config):
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    # Init game
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load(gambar_peta)

    # Main loop
    global generation
    generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)


        # Input data and get result from network
        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10

        # Update car and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                genomes[i][1].fitness += car.get_reward()

        # check
        if remain_cars == 0:
            break

        # Drawing
        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        screen.blit(text, text_rect)

        text = font.render("remain cars : " + str(remain_cars), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)

        pygame.display.update()
        clock.tick(60)
```
