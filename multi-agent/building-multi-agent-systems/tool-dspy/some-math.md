# Some Math

## DSPy Optimization: Inputs, Outputs, and Ground Truth

### 1) Optimization Equation

DSPy frames prompt optimization as an expected-reward maximization:

$$
\theta^* \;=\; \arg\max_{\theta \in \Theta} \; \mathbb{E}_{(x,y)\sim \mathcal{D}}\!\left[ R\!\big(f_{\theta}(x),\, y\big) \right].
$$

* $$\theta$$: prompt parameters (instructions, few-shot examples, etc.)
* $$f_{\theta}$$: the LLM when run with prompt $$\theta$$
* $$(x,y)$$: input and desired output
* $$R(\cdot,\cdot)$$: reward/metric comparing model output vs. ground truth
* $$\mathcal{D}$$: data distribution

With a finite labeled dataset $$D_{\text{train}}=\{(x_i,y_i)\}_{i=1}^N$$, DSPy optimizes the empirical objective:

$$
\hat{\theta}
\;=\; \arg\max_{\theta \in \Theta} \;
\frac{1}{N}\sum_{i=1}^{N} R\!\big(f_{\theta}(x_i),\, y_i\big).
$$

***

### 2) What Are $$x$$ and $$y$$?

* **Inputs** $$x$$ (what you care about)
  * QA system $$\rightarrow$$ questions
  * Summarizer $$\rightarrow$$ documents
  * Retriever $$\rightarrow$$ queries
* **Ground truth** $$y$$ (what you want)
  * QA $$\rightarrow$$ the correct answer
  * Summarization $$\rightarrow$$ a reference summary
  * Retrieval $$\rightarrow$$ a set of relevant documents $$Y$$

A labeled dataset is:

$$
D \;=\; \{(x_i, y_i)\}_{i=1}^{N}.
$$

***

### 3) How to Get Ground Truth $$y$$

* **Supervised datasets:** SQuAD, CNN/DailyMail, Natural Questions, BoolQ, etc.
* **Human annotations:** e.g., ticket classification for custom tasks.
* **Teacher LLM distillation:** use a stronger model to produce pseudo-labels $$\tilde{y}$$; then optimize prompts so a cheaper model mimics them. Validate with clean dev/test splits.

***

### 4) Reward Function $$R(\cdot,\cdot)$$

Choose task-appropriate metrics (larger is better).

*   **Exact Match (classification, short-answer QA):**

    $$
    R(\hat{y}, y) \;=\; \mathbf{1}\!\big[\hat{y} = y\big].
    $$
*   **Span/token overlap (NER, span QA):** F1 score

    $$
    \text{F1} \;=\; \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.
    $$
* **Text generation (summarization, translation):** BLEU, ROUGE, BERTScore.
*   **Retrieval:** For ground-truth set $$Y$$ and top-$$k$$ predicted set $$\hat{Y}_{1\!:\!k}$$,

    $$
    \operatorname{Recall@}k \;=\; \frac{\lvert Y \cap \hat{Y}_{1\!:\!k}\rvert}{\lvert Y\rvert}.
    $$

    If $$r_i$$ is the rank of the first relevant document for query $$i$$ (use $$\tfrac{1}{r_i}=0$$ if none found),

    $$
    \operatorname{MRR} \;=\; \frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}.
    $$
* **Task-specific checks:** e.g., “does the SQL query execute successfully?”

***

### 5) Toy Example

**Task:** Arithmetic word problems.

* **Dataset example:**\
  $$x$$ = "John has 3 apples, Mary gives him 2 more. How many apples does he have?"\
  $$y = 5$$
* **Candidate prompts** $$\theta$$**:**
  1. "Solve this math problem carefully."
  2. "You are a math tutor. Reason step by step before giving the final answer."
*   **Model runs:** let $$\hat{y}=f_{\theta}(x)$$. Suppose

    $$
    f_{\theta_1}(x)=4, \quad f_{\theta_2}(x)=5.
    $$
*   **Rewards (Exact Match):**

    $$
    R(4,5)=0, \quad R(5,5)=1.
    $$

    DSPy prefers $$\theta_2$$ and continues searching variations. Over many examples, it converges toward prompts with higher average reward.

***

### ✅ Summary

* $$x$$: task input (question, document, query, etc.)
* $$y$$: ground truth label (answer, summary, relevant docs, etc.)
* DSPy maximizes expected/empirical reward over labeled or pseudo-labeled data.
* Pick $$R$$ that truly reflects success for your task.

***

### ⚠️ Sanity Checks & Pitfalls

* The theory uses $$\mathcal{D}$$; in practice we optimize over $$D_{\text{train}}$$. Keep held-out dev/test sets.
* Pseudo-labels $$\tilde{y}$$ can be noisy; monitor for overfitting to teacher idiosyncrasies.
* Align $$R$$ with downstream goals (e.g., ROUGE $$\not\Rightarrow$$ factuality).
* For retrieval, $$y$$ is a set $$Y$$; prefer ranking/set metrics (Recall@$$k$$, MRR, nDCG) over exact match.
* Avoid test leakage when searching/selecting prompts.
