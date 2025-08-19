# DSPy Basic: Signature

## Understanding DSPy Signatures

Signatures are at the heart of DSPy. They define **what a task is** in a _declarative_ way. This chapter explains their structure, how to write them, and best practices.

***

### What is a Signature?

A **Signature** tells DSPy what task you want to perform.\
It specifies:

* **Inputs** (what goes in)
* **Outputs** (what comes out)
* **Optional instructions** (a docstring or short description of the task)

This allows DSPy to automatically build and optimize prompts or even train models without hard-coding instructions.

***

### Two Ways to Define a Signature

#### A) String Form

```python
import dspy

# "inputs -> outputs", with optional instructions string
Summarize = dspy.Signature(
    "document -> summary",
    "Write a concise, factual summary."
)
predict = dspy.Predict(Summarize)
```

* First argument: `"inputs -> outputs"` format
* Second argument: optional instructions string

***

#### B) Class Form (Recommended)

```python
import dspy

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField(desc="a single, factual question")
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

* Use `InputField` and `OutputField` to declare inputs/outputs
* `desc` fields help guide the LM in formatting and constraints
* The **docstring** is a high-level declarative description of the task

***

### Minimal vs. Rich Examples

#### Minimal (no instructions)

```python
class Classify(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()
```

Even without a docstring, DSPy uses field names to build prompts.

***

#### With concise docstring and field descriptions

```python
class Classify(dspy.Signature):
    """Classify the text as Positive, Negative, or Neutral."""
    text: str = dspy.InputField(desc="short customer review")
    label: str = dspy.OutputField(desc="one of: Positive | Negative | Neutral")
```

The docstring defines _what_ to do; field descriptions provide details on _how to format outputs_.

***

#### String Form with Instructions

```python
Tag = dspy.Signature(
    "title, body -> tags",
    "Return 3–5 topical tags; no punctuation, lowercase."
)
```

Instructions are optional, but useful when you want to specify constraints.

***

### Best Practices for Writing Signatures

* **Prefer class form** for readability and composability.
* **Keep instructions short**: state _what_ to do, not _how_ to prompt.
* **Use `desc`** fields for formatting rules (e.g., “JSON with keys …”).
* **Avoid overspecifying**: let DSPy’s optimizers handle example formatting.
* Keep descriptions **testable** and **outcome-oriented** (e.g., “≤ 60 words”, “3–5 items”, “valid JSON”).

***

### Worked Examples

#### Classification

```python
class Sentiment(dspy.Signature):
    """Classify the text as Positive, Negative, or Neutral."""
    text: str = dspy.InputField(desc="short customer review")
    label: str = dspy.OutputField(desc="one of: Positive | Negative | Neutral")
```

#### Information Extraction

```python
class ExtractEntities(dspy.Signature):
    """Extract unique person, organization, and location entities."""
    text: str = dspy.InputField(desc="well-formed English paragraph")
    entities: str = dspy.OutputField(desc="JSON with keys: persons, orgs, locations")
```

#### QA (String Form)

```python
AnswerQA = dspy.Signature(
    "question -> answer",
    "Answer with a short factual phrase; no extra words."
)
```

***

### Questions I had (and answers I found)

#### 1) Is the signature more than just input and output?

Yes. A Signature can include **instructions** (via docstring or the second argument in the string form). These instructions are part of the declarative task spec and help the compiler.

#### 2) Is the docstring the declarative description of the task?

Yes. In the class form, the **docstring** serves as the declarative statement of the task. Field `desc` values add fine-grained formatting or constraint guidance.

#### 3) Can we give an empty docstring?

Yes. Instructions are **optional**. You can omit the docstring entirely; DSPy relies on the input/output fields (and any `desc`) to assemble the prompt.

#### 4) How specific should the docstring be?

Aim for **concise, outcome-oriented, testable** specificity:

* ✅ Good: “Return JSON with keys `title` (string), `tags` (list of 3–5 lowercase nouns), `summary` (≤ 60 words). Avoid speculation.”
* ❌ Too vague: “Summarize and tag the article.”
* ❌ Overly prescriptive: long prompt templates or embedded few-shot examples—let DSPy compile/optimize those separately.

***

### Key Takeaways

* $$Signature = Inputs+Outputs+Optional Instructions\text{Inputs} + \text{Outputs} + \text{Optional Instructions}$$
* The docstring is **declarative**, not prescriptive.
* You _can_ omit the docstring, but concise descriptions often improve results.
* Field descriptions (`desc`) are powerful for nudging output structure.
* Keep Signatures **simple, declarative, and testable**.
