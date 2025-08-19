# Declarative Approach

## DSPy: A Declarative Approach

### Introduction to DSPy's Declarative Programming Model

DSPy’s declarative model shifts focus from **how** to solve a task to **what** the task is.\
You specify the desired inputs/outputs and (optionally) high-level instructions; DSPy generates the prompts, wires modules, and can optimize them against a metric. This enables:

* Complex reasoning chains without hand-crafting every prompt,
* Automatic few-shot selection and prompt tuning,
* Reuse across models/datasets with minimal code changes.

> <mark style="color:$danger;">**Core idea:**</mark> declare **Signatures** and compose **Modules**; let teleprompters optimize the _how_.&#x20;

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

A **Signature** is a typed contract for an LM step: field names, which are inputs vs. outputs, and a short docstring as instructions. Fields use `dspy.InputField()` / `dspy.OutputField()` with optional `desc=` text that becomes part of the prompt.&#x20;

```python
import dspy

class QuestionAnswering(dspy.Signature):
    """Answer the question using only the given context, concisely."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often 1–5 words")
```
