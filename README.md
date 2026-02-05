# FlowLang (المسير): Programming for Professions

**FlowLang** is a state-of-the-art Domain Specific Language (DSL) designed to transform "LLM Magic" into "Professional Administration." Unlike traditional workflow engines, FlowLang treats AI agents as a specialized workforce, enforcing strict accountability, causal transparency, and hierarchical auditing.

---

## 🏛 The Conceptual Framework: "Programming for Professions"

FlowLang is built on the principle that AI orchestration should mirror professional administrative structures. It introduces three core archetypes:

### 1. The Conductor (المسير) — The Flow State
In FlowLang, a `flow` is not just a function; it is a **Conductor**. 
- **Maximal Granularity**: We use high-frequency checkpoints to prevent AI hallucination. Each checkpoint forces an "Unload/Load" cycle.
- **Unload/Load Cycle**: At each stage, the context is "unloaded" into a report, pruning irrelevant data, and then "loaded" as a fresh **Order** for the next stage. This keeps the AI focused and context clean.
- **Sequential Handover**: Reports are distributed across team members in turn, ensuring a fair and auditable distribution of professional responsibility.

### 2. The Order (التكليف) — The Atomic Unit of Work
A variable in FlowLang is not just data; it is an **Order**.
- **Lifecycle tracking**: Every Order has a state (`created`, `processing`, `completed`, `failed`).
- **Audit Trail**: Every Order carries its own history—who touched it, what was the verb used, and what was the timestamp.
- **Strict Typology**: Teams are specialized (e.g., `Command<Search>`, `Command<Judge>`). A "Search" team cannot "Judge," enforcing professional boundaries.

### 3. The Maestro (The Process Tree) — Hierarchical Mapping
The `process` structure acts as the **Maestro**, mapping the "family tree" of the product.
- **Binary Path Encoding**: Every node in the process tree has a unique bit-string address (e.g., `0101`). This allows the system to perform "Shortcut Searches" and instant tracing.
- **Work Mapping**: Accomplishments from **Orders** are automatically mapped back to nodes in the Process Tree, visualizing what has been built and what is missing.

---

## 🛠 Language Components

### Specialized Teams
Define a workforce with specific capacities and models.
```flowlang
team Quality_Assurance : Command<Judge> [size=3, model="gpt-4o"];
team Content_Creators : Command<Try>   [size=5, distribution=round_robin];
```

### System Sequences (Data Chains)
Create a "Guiding Thread" for causal links that persist even when context is pruned.
```flowlang
chain build_pipeline {
    nodes: [Research, Design, Implementation, QA];
    propagation: causal(decay=1.0, forward=true);
}
```

### The Maestro (Process Trees)
Define the hierarchical roadmap of your product or system.
```flowlang
process software_map "Product Roadmap" {
    root: "App";
    branch "App" -> ["Auth", "Database", "UI"];
    node "Auth" { priority: "high"; status: "pending"; };
}
```

### Flows & Checkpoints
Orchestrate work through granular stages.
```flowlang
flow production_pipeline(using: my_team) {
    checkpoint "discovery" (report: market_data) {
        market_data = my_team.search("Identify gaps");
    }
    
    checkpoint "design" (report: architecture) {
        # 'market_data' is LOADED here as a fresh Order
        architecture = my_team.try(market_data);
    }
}
```

---

## ⚡️ Production Features

- **AI Resilience**: Automatic retry logic with "Corrective Prompts." If an AI fails schema validation, FlowLang feeds the error back to the model for self-correction.
- **Persistence**: Full state serialization. Flows can be paused, snapshotted to disk, and resumed (`runtime.resume(path)`) after a crash or for human approval.
- **Human-in-the-loop**: Use `confirm("prompt")` to create human gates for high-stakes decisions.
- **Dry Run Mode**: Test complex multi-stage logic without calling actual AI APIs or performing side effects.

---

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run an Example**:
   ```bash
   python run_software_factory.py
   ```

3. **Check the Docs**:
   For deep dives into the conductor logic, see [docs/conductor_logic.md](docs/conductor_logic.md).

---

## 📂 System Structure

- `runtime.py`: The heart of the Conductor.
- `types.py`: Definitions for Orders and Professional Typology.
- `ai_providers.py`: The bridge to LLMs with "Binary Path Awareness."
- `semantic.py`: Ensures teams act within their professional bounds.
- `grammar.lark`: The formal definition of the FlowLang syntax.
