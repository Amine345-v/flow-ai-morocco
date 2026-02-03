# FlowLang (مسير) – Prototype v0.1

FlowLang هي لغة نطاق خاص (DSL) لتنسيق الأوامر المهنية ضمن "مسير" يحتوي على نقاط تفتيش Checkpoints، مع هياكل نظام: Teams، Chains، وProcess Trees.

## الميزات الرئيسية

- **فرق**: مجموعات من الوكلاء مع أدوار وقدرات محددة
- **سلاسل**: سلاسل سببية لنمذجة سير العمل والاعتماديات
- **أشجار العملية**: نمذجة عملية هرمية
- **تكامل الذكاء الاصطناعي**: دعم مدمج لمقدمي الذكاء الاصطناعي المتعددين (OpenAI، Anthropic، Gemini، إلخ)
- **نقاط التفتيش**: إدارة الحالة وسيطرة سير العمل

## التثبيت

```bash
pip install -r requirements.txt
```

## البدء السريع

```bash
python scripts/run.py examples/hospital.flow
```

## مكونات اللغة

### 1. فرق
تمثل الفرق مجموعات من الوكلاء مع قدرات أوامر محددة.

```flowlang
team DiagnosisTeam: Command<Judge> [size=2];
team ResourceTeam: Command<Search> [size=1];
```

### 2. سلاسل
نمذجة العلاقات السببية وسير العمل بين العقد.

```flowlang
chain PatientFlowChain {
  nodes: [Reception, Examination, Treatment];
  propagation: causal(decay=0.7, backprop=true, forward=true);
  labels: { critical: "true" };
  constraints: { min_beds: 1; require_protocol: true; };
}
```

### 3. أشجار العملية
نمذجة عملية هرمية مع الفروع والعقد.

```flowlang
process HospitalTree "Hospital System" {
  root: "QualityCare";
  branch "Emergency" -> ["ER", "ICU"];
  branch "Archive" -> ["PaperRecords"];
}
```

### 4. النتائج والأنواع
تحديد أنواع نتائج بنية للأوامر.

```flowlang
result JudgeResult {
  match: boolean;
  protocol: string;
  latency: number;
  notes: string;
};

result SearchResult {
  beds: number;
  doctors: list;
  drugs: list;
};
```

### 5. سيطرة سير العمل
تحديد سير العمل القابل للتنفيذ مع نقاط التفتيش.

```flowlang
flow PatientAdmission(using: [TriageTeam, ResourceTeam]) {
  checkpoint "InitialAssessment" {
    // عبارات محلية
    result = TriageTeam.judge("Assess patient condition");
    if (result.match) {
      flow.back_to("AdmitPatient");
    }
  }
}
```

## تكامل الذكاء الاصطناعي

تدعم FlowLang مقدمي الذكاء الاصطناعي المتعددين من خلال واجهة موحدة:

```bash
# متغيرات البيئة لمقدمي الذكاء الاصطناعي
export OPENAI_API_KEY=your_key_here  # ل OpenAI
export ANTHROPIC_API_KEY=your_key_here  # ل Anthropic
# إلخ.
```

### مقدمو الدعم
- OpenAI (نماذج GPT)
- Anthropic (Claude)
- Google Gemini
- Mistral AI
- Cohere
- Azure OpenAI
- OpenRouter
- Ollama

## هيكل النظام

- `flowlang/grammar.lark`: تعريف قواعد اللغة
- `flowlang/ast.py`: عقد شجرة النحو المجردة
- `flowlang/types.py`: نظام الأنواع
- `flowlang/parser.py`: تحليل اللغة وتحويل شجرة النحو المجردة
- `flowlang/semantic.py`: تحليل دلالي
- `flowlang/runtime.py`: محرك التنفيذ
- `flowlang/ai_providers.py`: تكاملات مقدمي الذكاء الاصطناعي
- `flowlang/errors.py`: استثناءات مخصصة

## الأمثلة

انظر إلى دليل `examples/` للحصول على أمثلة كاملة:

- `hospital.flow`: نظام إدارة المستشفى
- `example1.flow`: بناء الجملة الأساسي والميزات

## الحصول على المساعدة

للمسائل أو المشكلات، يرجى [فتح قضية](https://github.com/your-repo/issues).

## رخصة

[حدد رخصتك هنا]
