const fs = require('fs');

class FlowResult {
  static search(hits = [], meta = {}) {
    return { tag: 'SEARCH_RESULT', hits, meta, pass: true };
  }
  static try(success = true, output = {}, meta = {}) {
    return { tag: 'TRY_RESULT', success, output, meta, pass: success };
  }
  static judge(pass = true, score = 1.0, reason = "") {
    return { tag: 'JUDGE_RESULT', pass, score, reason };
  }
  static ask(response = "", meta = {}) {
    return { tag: 'ASK_RESULT', response, meta, pass: true };
  }
}

class FlowWorker {
  constructor(opts = {}) {
    this.team = opts.team || 'Worker';
    this.handlers = {};
  }

  on(verb, handler) {
    this.handlers[verb] = handler;
    this.handlers[verb.toLowerCase()] = handler;
    this.handlers[verb.toUpperCase()] = handler;
  }

  async start() {
    try {
      const inputStr = fs.readFileSync(0, 'utf-8');
      if (!inputStr) process.exit(0);
      const input = JSON.parse(inputStr);
      const verb = input.verb || 'try';
      const args = input.args || [];
      const kwargs = input.options || input.kwargs || {};
      const ctx = input.context || {};

      const handler = this.handlers[verb] || this.handlers[verb.toLowerCase()] || this.handlers[verb.toUpperCase()];
      if (handler) {
        const res = await handler(args, kwargs, ctx);
        console.log(JSON.stringify(res));
      } else {
        console.log(JSON.stringify(FlowResult.try(true, { message: `Default worker handler for ${verb}` })));
      }
    } catch (e) {
      console.log(JSON.stringify(FlowResult.judge(false, 0.0, e.message)));
    }
  }
}

module.exports = { FlowWorker, FlowResult };
