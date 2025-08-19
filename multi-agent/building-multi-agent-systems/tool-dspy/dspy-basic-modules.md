# DSPy Basic: Modules

## Chapter: Programming with DSPy **Modules**

Modules are the core building blocks in DSPy. They wrap a **Signature** (the declarative task spec), define a **strategy** for solving it (e.g., plain prediction, chain-of-thought, tool-use), and expose **learnable parameters** that DSPy’s optimizers can tune. Multiple modules compose into larger programs.

> Quick mental model:&#x20;
>
> $$Module=Signature+Strategy+Parameters\text{Module} = \text{Signature} + \text{Strategy} + \text{Parameters}$$&#x20;
>
> (and it’s composable like a PyTorch module).&#x20;

***

### What is a _Module_?

* **Signature-driven** — you pass a `Signature` to tell the module its inputs/outputs (and optional instructions).
* **Composable** — modules can be nested to form programs.
* **Optimizable** — DSPy optimizers can synthesize demos, propose instructions, or even fine-tune weights depending on the optimizer.

DSPy provides a set of **built-in modules** (e.g., `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`) and lets you author **user-defined modules** by subclassing `dspy.Module`.

***

### Built-in Modules (essentials)

#### `dspy.Predict` — plain prediction

The simplest module: given a signature, produce the outputs directly.

```python
import dspy

class SummarizeSig(dspy.Signature):
    """Write a concise, factual summary."""
    document: str = dspy.InputField()
    summary: str  = dspy.OutputField()

summarize = dspy.Predict(SummarizeSig)
out = summarize(document="Alice joined…")
print(out.summary)
```

`Predict` is the baseline many other patterns build on; it supports save/load of compiled state.

***

#### `dspy.ChainOfThought` — show your work

Adds a _reasoning_ step and returns it alongside the answer.

```python
import dspy

class ExplainSig(dspy.Signature):
    """Answer in one sentence."""
    question: str = dspy.InputField()
    answer: str   = dspy.OutputField()

cot = dspy.ChainOfThought(ExplainSig)   # exposes .reasoning in outputs
resp = cot(question="Why is the sky blue?")
print(resp.reasoning)
print(resp.answer)
```

CoT augments your outputs with a `reasoning` field and has the usual module APIs (e.g., `forward`, `save`).

***

#### `dspy.ReAct` — tools + iterative reasoning

Implements the **ReAct** pattern: the model reasons, optionally calls **tools** (Python callables), and decides when to stop.

```python
import dspy

def web_search(query: str) -> str:
    # ... call your search backend ...
    return "top results…"

class ResearchSig(dspy.Signature):
    """Return a sourced, 2-sentence answer."""
    question: str = dspy.InputField()
    answer: str   = dspy.OutputField()

agent = dspy.ReAct(ResearchSig, tools=[web_search], max_iters=6)
result = agent(question="What is CRISPR base editing?")
print(result.answer)
```

DSPy’s ReAct is generalized to **work over any signature** (thanks to signature polymorphism) and takes a list of tools.

***

#### `dspy.Retrieve` — the retrieval layer for RAG

A standard retrieval layer that uses your configured **Retrieval Model (RM)** to fetch relevant context; typically paired with `Predict` or `ChainOfThought` to build RAG.&#x20;

```python
import dspy

class RagAnswer(dspy.Signature):
    """Answer with a short, factual sentence."""
    question: str = dspy.InputField()
    answer: str   = dspy.OutputField()

class RagProgram(dspy.Module):
    def __init__(self, k=4):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.answer   = dspy.Predict(RagAnswer)

    def forward(self, question: str):
        ctx = self.retrieve(question=question)
        return self.answer(question=f"{question}\n\nCONTEXT:\n{ctx}")

rag = RagProgram(k=4)
print(rag(question="When was CRISPR discovered?").answer)
```

Configure LM/RM once via settings, and modules will use them.

***

### User-defined Modules (subclassing)

Create a class that subclasses `dspy.Module`, define submodules in `__init__`, and implement `forward(...)` with typed arguments matching your signature.

```python
import dspy

class GradeSig(dspy.Signature):
    """Return 'correct' or 'incorrect' for the proposed answer."""
    question: str = dspy.InputField()
    proposed: str = dspy.InputField()
    verdict: str  = dspy.OutputField(desc="one of: correct | incorrect")

class Grader(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict(GradeSig)

    def forward(self, question: str, proposed: str):
        return self.judge(question=question, proposed=proposed)

grader = Grader()
print(grader(question="2+2?", proposed="4").verdict)
```

This is the recommended way to package multi-step logic into a reusable, optimizable unit.

***

### Hands-on: One task, three ways (Predict → CoT → ReAct)

Below we solve _the same_ task — “answer with one short, factual sentence” — three ways, and then show how an **optimizer** changes the module’s internals.

> **Setup** — configure your LM (and optionally RM for ReAct with tools that fetch data).

```python
import os, dspy
os.environ["OPENAI_API_KEY"] = "sk-..."  # or any provider supported by dspy.LM
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

The LM configuration uses the global settings; modules will pick it up automatically.

***

#### 1) Plain `Predict`

```python
class QA(dspy.Signature):
    """Answer with one short, factual sentence."""
    question: str = dspy.InputField()
    answer: str   = dspy.OutputField()

qa_predict = dspy.Predict(QA)
print(qa_predict(question="What organ pumps blood?").answer)
```

* _Inspect the last prompt call (optional):_

```python
dspy.inspect_history(n=1)  # prints the last LM call (prompt + choices)
```

`inspect_history` shows recent LM calls for observability.

***

#### 2) `ChainOfThought` (reasoning trace + answer)

```python
qa_cot = dspy.ChainOfThought(QA)
resp = qa_cot(question="What organ pumps blood?")
print(resp.reasoning)
print(resp.answer)
```

CoT returns a `.reasoning` field alongside `.answer`.

***

#### 3) `ReAct` (reason → tool → answer)

```python
def wiki_search(q: str) -> str:
    # ... fetch a short snippet from your backend ...
    return "The heart is a muscular organ..."

qa_react = dspy.ReAct(QA, tools=[wiki_search], max_iters=4)
print(qa_react(question="What organ pumps blood?").answer)
```

ReAct iterates thought→action→observation until it decides to answer.

***

### How optimization changes modules

DSPy **optimizers** can synthesize demos (few-shot examples), propose better instructions, or fine-tune model weights — depending on which optimizer you use. Common ones include `BootstrapRS`/`BootstrapFewShot` (demos), `MIPROv2` (instructions + demos), and `BootstrapFinetune` (weights).

#### Minimal dataset + metric

```python
from dspy import Example

trainset = [
    Example(question="What organ pumps blood?", answer="The heart.").with_inputs("question"),
    Example(question="What gas do plants absorb?", answer="Carbon dioxide.").with_inputs("question"),
]

def exact_match(pred, gold):
    return int(pred.answer.strip().lower() == gold.answer.strip().lower())
```

Examples are the core data type for small train/dev/test sets in DSPy.

#### A) Optimize demos with `BootstrapFewShot` (or `BootstrapRS`)

```python
optimizer = dspy.BootstrapFewShot(k=4, max_rounds=1)  # or dspy.BootstrapRS(...)
qa_predict_tuned = optimizer.compile(
    student=qa_predict, trainset=trainset, metric=exact_match
)
```

This adds synthesized/labeled demos to your module’s prompt. Inspect state before/after:

```python
before = qa_predict.dump_state()
after  = qa_predict_tuned.dump_state()
print(list(before.keys()))   # e.g., signature instructions, demos, etc.
print(list(after.keys()))    # should now include non-empty demos
```

Bootstrapping optimizers build few-shot **demonstrations** and inject them into your predictors. The compiled state contains signature + demos and other configuration, and can be saved/loaded.

#### B) Optimize instructions (+ demos) with `MIPROv2`

```python
mipro = dspy.MIPROv2(num_trials=12)  # try more for better results
qa_predict_mipro = mipro.compile(
    student=qa_predict, trainset=trainset, metric=exact_match
)
```

MIPROv2 proposes improved **instructions** and few-shot examples jointly (Bayesian optimization over combinations).&#x20;

> **Tip:** Use `dspy.inspect_history(n=3)` before/after to see how instructions and demos evolve in the prompts.

#### C) Fine-tune weights with `BootstrapFinetune` (optional, experimental)

```python
dspy.settings.experimental = True
ft = dspy.BootstrapFinetune(num_threads=8)
qa_predict_ft = ft.compile(student=qa_predict, trainset=trainset)  # can also pass teacher, metric
```

`BootstrapFinetune` converts a prompt-based program into **weight updates** for the underlying LM (offline RL / filtered behavior cloning). Often composed with prompt optimizers.

***

### Composition patterns

* **RAG**: `dspy.Retrieve` → `dspy.Predict`/`dspy.ChainOfThought`. Configure your RM once with `dspy.settings.configure(rm=...)`.&#x20;
* **Agentic**: `dspy.ReAct(Signature, tools=[...])` for tool-using agents; cap `max_iters`.
* **Custom programs**: subclass `dspy.Module`, define submodules, and implement `forward`.

***

### Observability, saving, and loading

* **Inspect prompts**: `dspy.inspect_history(n=...)` to print recent LM calls.
* **Save/Load**:
  * _State-only_ (`.json`/`.pkl`): `program.save("state.json", save_program=False)`; later `program.load("state.json")`.
  * _Whole program_ (architecture + state, via cloudpickle): `program.save("dir/", save_program=True)`; later `dspy.load("dir/")`.

***

### Practical tips & pitfalls

* **One responsibility per module** (retrieve vs. reason vs. generate) improves optimizer signal and debuggability.
* **Pick the right optimizer** for your bottleneck: demos (`BootstrapFewShot/RS`), instructions+examples (`MIPROv2`), or weights (`BootstrapFinetune`).
* **CoT outputs** include a `reasoning` field — read `.answer` for the final output.
* **Tool loops**: constrain `max_iters` and be explicit in your signature/instructions.

***

### Quick reference (built-ins)

* **`dspy.Predict(Signature)`** — direct prediction from inputs to outputs. ([DSPy](https://dspy.ai/api/modules/Predict/?utm_source=chatgpt.com))
* **`dspy.ChainOfThought(Signature, ...)`** — adds `reasoning` to outputs and encourages step-by-step thinking. ([DSPy](https://dspy.ai/api/modules/ChainOfThought/?utm_source=chatgpt.com))
* **`dspy.ReAct(Signature, tools=[...], max_iters=...)`** — iterative reasoning + tool use. ([DSPy](https://dspy.ai/api/modules/ReAct/?utm_source=chatgpt.com))
* **`dspy.Retrieve(k=...)`** — retrieval layer powered by the configured RM; use in RAG pipelines. ([Databricks](https://www.databricks.com/blog/dspy-databricks?utm_source=chatgpt.com))

***

#### Further reading

* **Modules (guide)** — overview, composition, usage. ([DSPy](https://dspy.ai/learn/programming/modules/?utm_source=chatgpt.com))
* **Signatures (guide)** — how module inputs/outputs/instructions are declared. ([DSPy](https://dspy.ai/learn/programming/signatures/?utm_source=chatgpt.com))
* **Optimizers** — BootstrapFewShot/RS, MIPROv2, BootstrapFinetune. ([DSPy](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com))
* **ReAct API** — tool integration & iterations. ([DSPy](https://dspy.ai/api/modules/ReAct/?utm_source=chatgpt.com))
* **Saving/Loading** — state vs. whole-program, and `dspy.load(...)`. ([DSPy](https://dspy.ai/tutorials/saving/?utm_source=chatgpt.com))

