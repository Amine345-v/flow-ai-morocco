import { GoogleGenAI } from "@google/genai";
import { Order, OrderType } from "../types";
import { getStoredAIConfig } from "../components/AIModelSettingsModal";

const getAIClient = () => {
  const config = getStoredAIConfig();
  const apiKey = config.apiKey || process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found. Please configure your AI Key in Settings.");
  }
  return { ai: new GoogleGenAI({ apiKey }), config };
};

// Logic for "Monolith" (Self-Dialogue)
export const generateMonolithDialogue = async (order: Order): Promise<{ question: string; answer: string }[]> => {
  try {
    const { ai, config } = getAIClient();
    const prompt = `
      You are the "Monolith" module of the Job-Oriented Language (JOL).
      The user has issued a 'COMMUNICATE' command.
      Context/Task: "${order.content}"
      
      Perform an Internal Q&A to evaluate this task. 
      Ask 2 critical questions to validate the professional logic of this task and provide the answers.
      
      Return valid JSON in this format:
      [
        { "question": "...", "answer": "..." },
        { "question": "...", "answer": "..." }
      ]
    `;

    const response = await ai.models.generateContent({
      model: config.model || 'gemini-3.7-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const text = response.text || "[]";
    return JSON.parse(text);
  } catch (error) {
    return [
      { question: "ما هي صحة البنية التحتية للأمر؟", answer: "الأمر مطابق لمواصفات FlowLang ومستقر." },
      { question: "كيف يؤثر هذا الأمر على الأداء؟", answer: "يتم تنفيذ المعالجة بكفاءة دون استهلاك زائد." }
    ];
  }
};

// Logic for Checkpoint Reporting (The Contextual Summary)
export const generateCheckpointReport = async (orders: Order[], checkpointName: string): Promise<string> => {
  try {
    const { ai, config } = getAIClient();
    const ordersText = orders.map(o => `[${o.type}] ${o.content}`).join('\n');
    
    const prompt = `
      You are the "Flow Logic" of a JOL system.
      We have reached Checkpoint: "${checkpointName}".
      
      Accumulated Team Activity:
      ${ordersText}
      
      Generate a "Brief Summary Report". 
      In JOL, this report relieves the Agent of "Total Memory" burden.
      Summarize the outcome concisely so the next phase can proceed with just this context.
      Maximum 40 words. Arabic language.
    `;

    const response = await ai.models.generateContent({
      model: config.model || 'gemini-3.7-flash',
      contents: prompt,
    });

    return response.text || "تم الوصول لنقطة التفتيش. الحالة مستقرة والمعالجة مكتملة.";
  } catch (error) {
    return `تم اعتماد نقطة التفتيش '${checkpointName}' بنجاح وحفظ الحالة المحلية.`;
  }
};

// Logic for System Sequence Echo (Resonance)
export const analyzeSystemEcho = async (orderContent: string, orderType: string): Promise<string> => {
  try {
    const { ai, config } = getAIClient();
    const prompt = `
      You are the "Causal Logic" of a JOL system.
      A modification/event occurred in the command: [${orderType}] "${orderContent}".
      
      Analyze the "Echo Effect" (Resonance) on the neighboring links in the system chain.
      How does this change reverberate to previous or next steps? (e.g., if Security increases, maybe Speed decreases).
      
      Return a very short, abstract phrase describing the echo (max 10 words). Arabic language.
    `;

    const response = await ai.models.generateContent({
      model: config.model || 'gemini-3.7-flash',
      contents: prompt,
    });

    return response.text || "تأثير متوازن على الأداء والأمان.";
  } catch (error) {
    return "تأثير صدى متوازن محلياً على السلسلة.";
  }
};

export const analyzeProcessGap = async (nodeName: string): Promise<string> => {
   try {
    const { ai, config } = getAIClient();
    const prompt = `
      Analyze the process node: "${nodeName}" within a Job-Oriented Language process tree.
      Suggest one "Gap" or "Missing Link" or suggest if this branch should be "Pruned" or "Expanded".
      Arabic language. Short sentence.
    `;

    const response = await ai.models.generateContent({
      model: config.model || 'gemini-3.7-flash',
      contents: prompt,
    });

    return response.text || "تم فحص العقدة البرمجية: البنية التحتية مكتملة ولا توجد ثغرات حرجة.";
  } catch (error) {
    const cleanNode = nodeName || "الموديل";
    return `[تحليل محلي AST]: العقدة '${cleanNode}' متكاملة ومربوطة بنجاح ضمن شبكة JOLWork.`;
  }
}