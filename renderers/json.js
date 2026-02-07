// renderers/json.js — NDJSON frame output (for piping)

export default {
  init(config) {
    this.stream = config.stream || process.stdout;
  },

  render(frame) {
    this.stream.write(JSON.stringify(frame) + '\n');
  },

  close() {},
};
