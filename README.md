# 2025_BRP_RL

|                 Developer                |       
| :--------------------------------------: |  
| [Seung Woo Han](https://github.com/SeungwooHH11) | 
|         🧑‍💻 AI-Development               |         
<br>

## Project Overview
- **Project**
    - Retrieval and placement optimization in shipyard hull-block storage yard
- **Superviser**
    - Prof. Jong Hun Woo (SNU, KOREA)
- **Paper**
    - Optimization of Hull-block Placement and Retrieval Path Planning for Congested Planar Storage in Shipyard
    - Transportation part E (Preparing submitting)
<br>

## Project Introduction
We propose retrieval path planning and placement decision algorithm for minimizing the relocation of interference block
<br>
Effective operation of storage yard is major points in shipyard logistics . <br>
Planar transportation constraint of hull-block incurs the interference block relocation in retrieval path. <br>
Dynamic programming based retrieval path plannig algorithm and graph reinforcement learning based placement algorithm. <br>

<p align="center">
<img src="figure/Problem_description.png" width="80%">
<br>


## Main

### 1️⃣ Retrieval process
#### 1.1 Block retrieval process with blocking item relocation
<p align="center">
<img src="figure/Retrieval_process.png">
<br>

### 2️⃣ Retrieval algorithm
#### 2.1 Minimum blocking path plannig algorithm
<p align="center">
<img src="figure/MBP.png" width="65%">

<br>

#### 2.2 Buffer-aware path planning algorithm
<p align="center">
<img src="figure/BAP.png" width="65%">
<br>


### 3️⃣ Placement algorithm

#### 3.1 Simulated Greedy Placement
- State composed of unscheduled blocks and transporters
    - **edge attributed graph**: compact and effective representation of block transportation statue
        - nodes representing location which contain current transporter information
        - edges representing blocks with origin and destination by disjunctive edge
    - **Crystal graph convolutional neural network**: graph neural network that suitable for encoding edge attributed graph

<br>

#### 3.2  Graph reinforcement learning

##### 3.2.1 State
- State composed of unscheduled blocks and transporters
    - **edge attributed graph**: compact and effective representation of block transportation statue
        - nodes representing location which contain current transporter information
        - edges representing blocks with origin and destination by disjunctive edge
    - **Crystal graph convolutional neural network**: graph neural network that suitable for encoding edge attributed graph

##### 3.2.2 Action
- a combination of the vessel and quay-wall (machine assignment and job sequencing)
    - **candidate vessels**
        - newly launched vessels from the docks
        - vessels returning from sea trials
        - vessels waiting at sea owing to the shortage of quay-walls
        - vessels that need to be reallocated due to interruption
    - **candidate quay walls**
        - empty quay walls
        - occupied quay walls with preemption allowed

##### 3.2.3 Reward
- minimization of the total cost in the post-stage outfitting process
- a sum of three cost-related rewards
    - **penalty cost**: the penalty cost for the delay in the delivery of vessels
    - **moving cost**: the cost of moving the vessels
    - **loss cost**: the additional processing cost

<br>

##### 3.2.4 Network Structure
<img src="figure/Placement network.png"/>


- **Representation module**
    - Two types of latent representation are extracted from the heterogeneous graphs and auxiliary matrix, respectively
    - For heterogeneous graphs, the embedding vectors of nodes are generated using the relational information between nodes
    - For an auxiliary matrix, the embedding vectors for combinations of quay-walls and vessels are generated using the MLP layers 
- **Aggregation module**
    - Input vectors for the output model are generated based on the embedding vectors from the representation module
- **Output module**
    - The actor layers calculate the probability distribution over actions $\pi_{\theta} (\cdot|s_t)$
    - The critic layers calculate a approximate state-value function $V_{\pi_{\theta}} (s_t)$, respectively

#### 4.2 Reinforcement Learning Algorithm
- **PPO(proximal policy optimization)**
    - Policy-based reinforcement learning algorithm
