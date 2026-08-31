import { synthesizeFlowArchitectureWithAI, extractFullDirectoryCodebaseWithAI } from './services/geminiService';

async function testFlowLangSynthesis() {
  if (!process.env.VITE_GEMINI_API_KEY && !process.env.GEMINI_API_KEY) {
    // Provide test key for local CLI harness testing if not provided
    process.env.VITE_GEMINI_API_KEY = "TEST_API_KEY";
  }

  console.log("=================================================================");
  console.log("🧪 TESTING FLOWLANG DSL & FULL DIRECTORY AI SYNTHESIS");
  console.log("=================================================================\n");

  const prompt = "clone paypal";
  const domain = "digital";

  // Step 1: Test synthesizeFlowArchitectureWithAI
  console.log(`[1/2] Synthesizing FlowLang DSL Architecture for prompt: "${prompt}"...`);
  const archResult = await synthesizeFlowArchitectureWithAI(prompt, domain);

  console.log("\n--- Synthesized FlowLang DSL (.flow) Content ---");
  console.log(archResult.dslContent);
  console.log("------------------------------------------------\n");

  // Assertions for FlowLang DSL keywords
  const hasOrder = archResult.dslContent.includes("order ");
  const hasProcess = archResult.dslContent.includes("process ");
  const hasChain = archResult.dslContent.includes("chain ");
  const hasTeam = archResult.dslContent.includes("team ");
  const hasFlow = archResult.dslContent.includes("flow ");
  const hasCheckpoint = archResult.dslContent.includes("checkpoint ");

  if (hasOrder && hasProcess && hasChain && hasTeam && hasFlow && hasCheckpoint) {
    console.log("✅ PASS: FlowLang DSL contains all required syntax directives (order, process, chain, team, flow, checkpoint).");
  } else {
    console.error("❌ FAIL: FlowLang DSL is missing one or more syntax directives!");
    process.exit(1);
  }

  // Step 2: Test extractFullDirectoryCodebaseWithAI
  console.log(`\n[2/2] Extracting Full Directory Codebase for prompt: "${prompt}"...`);
  const files = await extractFullDirectoryCodebaseWithAI(prompt, domain);

  console.log(`\n📁 Synthesized Codebase Directory (${files.length} files generated):`);
  files.forEach(f => {
    console.log(`  - [${f.type.toUpperCase()}] ${f.name} (${f.size}) -> Path: ${f.path}`);
  });

  const flowFile = files.find(f => f.type === 'flow');
  if (flowFile) {
    console.log(`\n✅ PASS: Primary pipeline file "${flowFile.name}" synthesized as FlowLang DSL (.flow).`);
  } else {
    console.error("❌ FAIL: Codebase directory missing primary .flow file!");
    process.exit(1);
  }

  console.log("\n=================================================================");
  console.log("🎉 ALL FLOWLANG SYNTHESIS TESTS PASSED SUCCESSFULLY!");
  console.log("=================================================================");
}

testFlowLangSynthesis().catch(err => {
  console.error("❌ Error running synthesis test:", err);
  process.exit(1);
});
