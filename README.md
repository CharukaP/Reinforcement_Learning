**Part 1**

Libraries used in this exercise

	Numpy
	Matplotlib
Parameters

There are three main parameters used for governing the entire excersis and rest of the parameters are function driven

	Number of arms (k)
	Steps of an episodes (steps)
	Number of episodes considered to get average performance (episodes)

Part 1 file can be access by using Assignment1.1.ipynb file

**Part 2**

In real world fixed values for rewards are difficult to observe there is always possibility to change in values with time. In this section we will explore such changes and how the algorithm manages to perform with those changes.

Change in Parameter

We are exploring three different changes to our mean parameters in order to mimic real world scenarios and introduce those functions to our original code.

	Drift change
	Reverting change
	Abrupt change
 
First change we can introduce is drift change here we consider drift in mean values with each step of the actions. Change in mean value is defined as below formula and according to that drift_change function is added to the code.
μ_t = μ_(t-1) + ϵ_t  | ϵ_t ∈ N(0,(0.001)^2)

Second changing behavior is reverting change here we consider slightly bigger change in mean values than first instance. Change in mean is explained well in below formula. Similarly reverting_change function is introduced for the code to accommodate change

μ_t = K * μ_(t-1) + ϵ_t  | ϵ_t ∈ N(0,(0.001)^2 ), K=0.5

The last change was abrupt change. Here we permute the element in mean values of k-arms with probability of 0.005. Although its small number change in value have a huge impact on reward collection.

With all these changes new parameter introduced to all the four algorithms with mean changing parameter where input of 0 to 3.

	0 = No mean change
	1 = Drift change
	2 = Reverting change
	3 = Abrupt change
 
Evaluation of non-stationary adaption

Adaptation to non-stationary problem of different algorithm is a paramount requirement in this study. Such comparison requires different approach due non-stationary behavior of reward distribution. Hence, we compare algorithm performance with the distribution of last reward of each episode keeping the step size of each episode to 10,000 iterations. Similarly, the total episodes count for this comparison kept as 1000 as previously. The main three algorithms evaluated in study are.

	Optimistic greedy (decreasing step size)
	ϵ - greedy with fixed step size (step size = 0.1)
	ϵ - greedy with decreasing step size

Furthermore, two different initiation steps carried out in order to analyze results. This will let the reader a better understanding of the performance of algorithm.

	Same starting mean for all the episodes (Randomly generated mean with seed=100)
	Same mean for each bandit different value for each episode (randomly generated ui where u1=u2=u3=u4……..=u10)
 

Analysis of part 2 can be accessed with Assignment1.2.ipynb file. Similarly part 2 can be use to recreate part 1 as well selecting aprpriate parameters in the section.


Detail explanation on analysis is uploaded in the "Assignment_1.pdf" file for reference.
