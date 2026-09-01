# FlowLang — Comprehensive Developer Tutorial & Quickstart

Welcome to FlowLang! FlowLang is a domain-specific meta-language designed for **governing multi-agent AI workflows** with deterministic structure, context pruning, and automated quality gates.

---

## Table of Contents
1. [Core Concepts](#1-core-concepts)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Your First FlowLang Program](#3-your-first-flowlang-program)
4. [Defining Teams & Workers](#4-defining-teams--workers)
5. [Process Trees & Causal Chains](#5-process-trees--causal-chains)
6. [Micro-Checkpoints & Team Delegation](#6-micro-checkpoints--team-delegation)
7. [Running & Debugging Flows](#7-running--debugging-flows)

---

## 1. Core Concepts

FlowLang translates complex business and engineering workflows into deterministic execution DAGs using 5 primitives:

- **Workforce Teams**: Homogeneous pools of workers executing structured verbs (`search`, `try`, `judge`, `ask`).
- **Checkpoints**: High-granularity execution stages that prune context noise to prevent AI hallucinations.
- **Micro-Checkpoints**: Batch processing loops powered by dedicated worker teams with pass-rate thresholds.
- **Process Trees**: Living audit blueprints tracking completed vs. pending tasks.
- **Causal Chains**: Inter-stage causal dependency graphs with decay propagation.

---

## 2. Prerequisites & Installation

FlowLang requires **Python 3.10+**.

```bash
git clone https://github.com/flowlang/flowlang.git
cd flowlang
pip install -r requirements.txt
```

---

## 3. Your First FlowLang Program

Create a file named `hello_flowlang.flow`:

```flow
// 1. Declare Worker Teams
team researcher : Command<Search> [size=2, distribution=round_robin];
team reviewer   : Command<Judge>  [size=1];

// 2. Define the Conductor Flow
flow hello_world(using: researcher, reviewer) {
    context retention: checkpoint;
    merge_policy: deep_merge;

    checkpoint "discovery" (report: topics) {
        topics = researcher.search("Latest developments in multi-agent AI systems");
        context.update(topics);
    }

    checkpoint "evaluation" (report: verdict) {
        verdict = reviewer.judge(topics, "Verify accuracy and relevance");

        if (verdict.pass) {
            flow.end;
        } else {
            flow.back_to("discovery");
        }
    }
}
```

---

## 4. Defining Teams & Workers

FlowLang supports 4 typed command verbs:

```flow
// Information retrieval
team dev_searcher : Command<Search> [size=2, distribution=round_robin];

// Code generation or task execution
team dev_builder  : Command<Try>    [size=3, distribution=round_robin];

// Evaluation and decision gates
team qa_judge     : Command<Judge>  [size=2, distribution=round_robin];

// Monologue & self-reflection
team communicator : Command<Communicate> [size=1];
```

You can connect teams to external JavaScript/Python workers:
```flow
team custom_worker : Command<Try> [size=1, connector="node scripts/custom_worker.js"];
```

---

## 5. Process Trees & Causal Chains

Map work to structural roadmap trees and causal chains:

```flow
process project_tree "SaaS Infrastructure Roadmap" {
    root: "System";
    branch "System" -> ["Database", "AuthService", "APIGateway"];
    node "Database" { status: "pending"; };
}

chain build_pipeline {
    nodes: [DataSchema, AuthModule, EndpointIntegration];
    propagation: causal(decay=0.8, backprop=true, forward=true);
}
```

Inside checkpoints, update process nodes and touch chains:
```flow
project_tree.mark("Database", "completed", reason="Schema migrated successfully");
build_pipeline.touch("DataSchema", effect=1.0);
```

---

## 6. Micro-Checkpoints & Team Delegation

Execute parallel batch micro-checks across workforce teams with safety thresholds:

```flow
checkpoint "security_audit" (report: audit_report) {
    modules = ["AuthModule", "PaymentGateway", "UserSettings", "APIKeys"];

    // Parallel micro-checkpoint across qa_judge team
    micro_checkpoint "vulnerability_scan" (using: qa_judge, batch: modules, strategy: parallel, threshold: 0.9) {
        scan_result = qa_judge.judge(item, "Ensure no SQL injection or OWASP Top 10 vulnerabilities");
    }

    audit_report = qa_judge.judge(modules, "Final security clearance");
}
```

---


## 7. Autonomous AI Software Factory Pipeline

FlowLang includes a full-fledged autonomous software generation pipeline capable of synthesizing entire enterprise software projects end-to-end.

### Step 1: Configure Environment (.env)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=Ab8RN...
FLOWLANG_AI_PROVIDER=gemini
FLOWLANG_GEMINI_MODEL=gemini-3.7-flash
```

### Step 3: Run 1-Hour Autonomous Software Factory Test Suite
```bash
cd jol-ide-studio
npm run test:hour
```

The workforce will:
1. Dynamically plan required file manifests for each stage (Market Discovery → Requirements → Core Services → DDL Schema → Deployment).
2. Invoke Python Core Gemini AI providers with multi-model fallback (`gemini-3.7-flash` → `gemini-3.6-flash`).
3. Handle API rate limits (429) automatically via auto-sleep quota reset.
4. Output clean, compile-ready `.ts`, `.tsx`, `.sql`, `.json`, and `.md` source files into `flowlang_test_results/`.

