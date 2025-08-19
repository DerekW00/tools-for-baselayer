# MAS Systems

## Architectures of Multi-Agent Systems

### Centralized Architectures

* **Description:** A single central agent controls and coordinates all others.
* **Advantages:** Easy to design, optimal solutions possible with full knowledge.
* **Disadvantages:** Single point of failure, poor scalability, limited adaptability.
* **Examples:** Client–server systems, factory automation with a central controller.

### Decentralized Architectures

* **Description:** Each agent makes its own decisions and coordinates with others locally.
* **Advantages:** Robustness, scalability, flexibility.
* **Disadvantages:** Complex design, coordination challenges, no guaranteed global optimality.
* **Examples:** Internet routing, bird flocking.

### Hybrid Architectures

* **Description:** Combine centralized planning with decentralized execution.
* **Advantages:** Balance between global coordination and local adaptability.
* **Disadvantages:** Complex to implement; requires careful coordination between layers.
* **Examples:** Smart cities (central oversight + local autonomy), supply chain management systems.

### Patterns and Protocols

* **Contract Net Protocol:** Tasks are announced, agents bid, and winners execute.
* **Market-Based Control:** Decisions mediated by auctions and prices.
* **Consensus Protocols:** Agents converge on shared values or decisions.
* **CTDE (Centralized Training, Decentralized Execution):** Widely used in multi-agent reinforcement learning.

### Design Trade-Offs

* Optimality vs. scalability.
* Latency vs. global visibility.
* Robustness vs. simplicity.
* Communication cost vs. decision quality.

