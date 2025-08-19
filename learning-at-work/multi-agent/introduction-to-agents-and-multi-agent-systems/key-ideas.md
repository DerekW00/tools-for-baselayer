# Key Ideas

## Key Ideas in Agent Design

### Tasks, Goals, and Rewards

* **Task:** The objective or problem the agent is meant to accomplish.
* **Goal:** Desired outcomes or states the agent aims to achieve.
* **Reward:** A scalar measure of success that evaluates the quality of the agent’s actions relative to its goals.

### State, Observations, and Beliefs

* **State:** The complete situation of the environment.
* **Observation (Perception):** Information the agent gathers via sensors; may be partial or noisy.
* **Belief:** Internal estimate of the state, often probabilistic, used in partially observable environments.

### Actions and Policies

* **Action:** A step the agent can perform to influence the environment.
* **Policy:** A mapping from states (or perceptions/beliefs) to actions, guiding the agent’s behavior.

### Planning and Learning

* **Planning:** Using models of transitions and rewards to predict future outcomes and select actions.
* **Learning:** Improving behavior based on experience (e.g., reinforcement learning).
* **Exploration vs. Exploitation:** Balancing the discovery of new strategies with leveraging known ones.

### Time Horizons

* **Episodic:** Tasks with clear start and end (e.g., chess).
* **Continuing:** Ongoing tasks with no fixed end (e.g., stock trading).
* **Discounting:** Rewards may be weighted to prefer immediate outcomes over distant ones.

### Formal Models

* **Markov Decision Process (MDP):** ⟨S, A, T, R, γ⟩ with
  * Transition model **T(s, a, s′)**,
  * Reward model **R(s, a, s′)**,
  * Discount factor γ.
* **Partially Observable MDP (POMDP):** Extends MDP with observation space Ω and observation model Z(o | s′, a); the agent must plan in belief space.

### Types of Environments

* **Fully vs. Partially Observable:** Complete vs. incomplete information about the state.
* **Deterministic vs. Stochastic:** Predictable vs. probabilistic outcomes of actions.
* **Static vs. Dynamic:** Whether the environment changes while the agent deliberates.
* **Single-Agent vs. Multi-Agent:** Acting alone vs. interacting with other agents.
* **Known vs. Unknown Physics:** Transition model given vs. must be learned.
* **Discrete vs. Continuous:** Finite vs. infinite state/action spaces.

> **What’s next:** Explore **Multi-Agent Systems**, where multiple agents interact in shared environments.
