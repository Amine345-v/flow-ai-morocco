# FlowLang Autonomous AI Software Factory Specification

## Executive Overview

The **FlowLang Autonomous Software Factory** is an enterprise-grade AI synthesis pipeline that turns business objectives (e.g., *"Build Accountant ERP Software"*, *"Build Enterprise SaaS Software Factory"*) into production-grade source code, architecture specifications, database schemas, and configuration manifests completely autonomously.

---

## 🏗 Pipeline Architecture & Flow

```
[ Business Order Prompt ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 🧠 Stage 1: Market Discovery & Requirements Planning  │
│  - AI Workforce Agent: Team=product_thinker, Verb=ask │
│  - Generates dynamically planned file manifest         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 📐 Stage 2: Architecture & System Topology Specification│
│  - AI Workforce Agent: Team=system_architects, Verb=try│
│  - Synthesizes .md briefs & .json schema specs         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 💻 Stage 3: Database & Core Domain Service Code        │
│  - AI Workforce Agent: Team=code_engineers, Verb=try  │
│  - Synthesizes .sql DDL schemas & TypeScript services  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 🧪 Stage 4: Testing & Production Deployment Release    │
│  - AI Workforce Agent: Team=qa_engineers, Verb=judge  │
│  - Synthesizes test suites & Docker deployment scripts  │
└────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Technical Subsystems

### 1. Native Python Core Engine (`flowlang/ai_providers.py`)
- **SDK Standard**: Integrates Google's modern `google.genai` SDK (`genai.Client`) with automatic fallback to legacy `google.generativeai`.
- **Model Fallback Sequence**:
  1. `gemini-3.7-flash` (Primary Flagship)
  2. `gemini-3.6-flash`
  3. `gemini-3.5-flash`
  4. `gemini-flash-latest`
- **429 Rate-Limit Quota Auto-Sleep**:
  - Automatically parses API error messages matching `retry in (\d+)s`.
  - Automatically pauses execution safely for the exact quota reset delay.
  - Automatically retries the live request to ensure 100% genuine LLM content synthesis without unhandled crashes.

### 2. Multi-Candidate Environment Resolution
Discovers `.env` and `.env.local` files across workspace paths:
- `.env`
- `.env.local`
- `jol-ide-studio/.env.local`

### 3. Extension-Aware High-Fidelity Synthesizer (`run_flowlang_ai_step.py`)
Maps file extensions to production code templates when API quotas require emergency fallback:
- **`.ts` / `.js`**: Generates enterprise TypeScript classes with `ExecutionContext`, `ExecutionResponse`, and typed async processing methods.
- **`.tsx` / `.jsx`**: Generates React UI dashboard components with hooks and Tailwind state styling.
- **`.sql`**: Generates DDL database schemas with UUID primary keys, foreign key cascades, and performance indexing.
- **`.json`**: Generates valid JSON Schemas compliant with Draft 2020-12.
- **`.md`**: Generates strategic architecture briefs with executive summaries and telemetry verification checklists.

### 4. 1-Hour Continuous Test Suite (`test_flowlang_hour_suite.ts`)
Executes full factory pipelines continuously over a 1-hour window with rate-limit protection pacing (default: 60s delay between AI requests).

---

## 🛠 Running the Software Factory

### Python Core Software Factory:
```bash
python build_accountant_erp.py
```

### TypeScript Studio 1-Hour Suite:
```bash
cd jol-ide-studio
npm run test:hour
```
