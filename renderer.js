// renderer.js — RendererManager: plugin loader/dispatcher
// Contract: renderer = { init(config), render(frame), close() }

import { pathToFileURL } from 'node:url';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export class RendererManager {
  constructor() {
    this.renderer = null;
    this.name = null;
  }

  async use(name, config = {}) {
    const modPath = join(__dirname, 'renderers', `${name}.js`);
    const mod = await import(pathToFileURL(modPath).href);
    this.renderer = mod.default || mod;
    this.name = name;
    if (this.renderer.init) await this.renderer.init(config);
    return this;
  }

  async render(frame) {
    if (!this.renderer) throw new Error('No renderer loaded');
    await this.renderer.render(frame);
  }

  async close() {
    if (this.renderer && this.renderer.close) await this.renderer.close();
    this.renderer = null;
    this.name = null;
  }
}
