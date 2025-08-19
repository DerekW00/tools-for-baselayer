# Declarative Approach

## DSPy: A Declarative Approach

### Introduction to DSPy's Declarative Programming Model

DSPy’s declarative model shifts focus from **how** to solve a task to **what** the task is.\
You specify the desired inputs/outputs and (optionally) high-level instructions; DSPy generates the prompts, wires modules, and can optimize them against a metric. This enables:

* Complex reasoning chains without hand-crafting every prompt
* Automatic few-shot selection and prompt tuning
* Reuse across models/datasets with minimal code changes

> Core idea: declare **Signatures** and compose **Modules**; let teleprompters optimize the _how_.

***

### Declarative vs. Imperative (Quick Intuition)

* **Imperative**: you script every step (“clean data → craft prompt variant A → insert demos → call LLM → post-process”).
* **Declarative**: you state the _contract_ (inputs/outputs + brief instructions). The system manages prompt formatting, example injection, and evaluation/optimization.

| Feature       | Imperative Programming      | Declarative Programming (DSPy)         |
| ------------- | --------------------------- | -------------------------------------- |
| Focus         | _How_ to achieve a result   | _What_ result is desired               |
| Control       | You micromanage steps       | System abstracts execution             |
| Code          | Step-by-step instructions   | High-level specifications (Signatures) |
| Abstraction   | Lower level                 | Higher level                           |
| Example langs | C, Java (procedural Python) | SQL, Prolog, **DSPy**                  |

***

### Signatures: Declaring Inputs/Outputs (+ optional instructions)

A **Signature** is a typed contract for an LM step: field names, which are inputs vs. outputs, and a short docstring as instructions. Fields use `dspy.InputField()` / `dspy.OutputField()` with optional `desc=` text that becomes part of the prompt.

```python
import dspy

class QuestionAnswering(dspy.Signature):
    """Answer the question using only the given context, concisely."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often 1–5 words")
```

You can also build Signatures programmatically, e.g.:

```python
qa_sig = dspy.Signature("context, question -> answer", "Answer the question using the context.")
```

***

### Modules: Composable Building Blocks

A **Module** wires one or more predictors/tools into a pipeline. Inside, you typically instantiate `dspy.Predict(...)` or `dspy.ChainOfThought(...)` with your Signature. The module’s `forward` method defines how the components are connected. Modules return `dspy.Prediction` objects.

```python
import dspy

class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict(QuestionAnswering)  # trainable LM call

    def forward(self, context: str, question: str):
        return self.solve(context=context, question=question)

# Example usage
qa = SimpleQA()
pred = qa(context="France's capital is Paris.", question="What is the capital of France?")
print(pred.answer)  # -> "Paris"
```

***

### Predict vs. Chain-of-Thought (CoT)

* **`dspy.Predict(Signature)`**: formats a prompt (with optional demos) and returns only the declared outputs—ideal for classification, extraction, and factoid QA.
* **`dspy.ChainOfThought(Signature)`**: same contract, but the module elicits a hidden _reasoning_ scratchpad before producing the outputs. The reasoning can be logged for training or analysis, but in production you usually return only the final answer.

```python
reason_then_answer = dspy.ChainOfThought(QuestionAnswering)
pred = reason_then_answer(context="2+3=5; 5+4=9", question="What is 5+4?")
print(pred.answer)      # "9"
# Optional (for debugging/distillation):
# print(pred.reasoning)
```

***

### From Declarative Spec to Optimization (Teleprompters)

Once your pipeline is declared, DSPy can _optimize_ instructions and few-shot demos via **teleprompters** (gradient-free outer-loop search). You supply a metric (e.g., exact match); DSPy explores candidate prompts/examples and installs the best ones into your predictors.

```python
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match

# Toy trainset: list of dspy.Example with ground-truth answers
trainset = [
    dspy.Example(
        context="Paris is the capital of France.",
        question="Capital of France?",
        answer="Paris"
    ).with_inputs("context", "question"),
]

# Optimize the module's prompt & few-shots
tele = BootstrapFewShot(metric=answer_exact_match, max_labeled_demos=8)
qa_optimized = tele.compile(SimpleQA(), trainset)
```

`BootstrapFewShot` composes demos from labeled data (and optionally bootstrapped outputs). Other teleprompters (e.g. `MIPROv2`) exist for different search strategies.

***

### Why this is “Declarative” in Practice

* **Contract-first**: the Signature (and docstring/field descriptions) defines the task intent. You can swap implementations (`Predict` ↔ `ChainOfThought`, model A ↔ model B) without breaking downstream code.
* **Pluggable optimization**: teleprompters operate on declared steps to maximize your metric—no imperative prompt hacking in application code.
* **Structured returns**: modules return `Prediction` objects keyed by your Signature (e.g., `answer`), with optional auxiliary fields (e.g., `reasoning` for CoT).

***

### Gotchas & Best Practices

1. **Always give the Signature a useful docstring and field descriptions** — they directly shape the generated prompt.
2. **Keep outputs concise & schema-like** — short answers or JSON schemas improve stability and scoring.
3. **Start with `Predict`, add CoT only if metrics improve** — CoT helps for math/multi-hop reasoning, but adds cost.
4. **Pick the right metric** — e.g., `answer_exact_match` for factoid QA, span-level F1 for extraction.
5. **Use small train/dev splits first** — avoid overfitting and wasted tokens; widen the search space later.
6. **Remember: modules return `Prediction` objects** — access via declared fields (e.g., `pred.answer`).

***

### Takeaway

With DSPy you describe **what** a step should consume and produce, compose those steps as **Modules**, and let the system optimize the **how** (prompt wording, demos, reasoning mode).

This makes your LLM pipelines more:

* **Modular** (easy to compose/reuse),
* **Optimizable** (teleprompters tune prompts/examples),
* **Portable** (swap models or datasets without rewriting prompts).

### By shifting from _imperative prompt engineering_ to _declarative specification_, you unlock automation, systematic optimization, and scalability in LLM programming.
