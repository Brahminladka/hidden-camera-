import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const outDir = join(root, 'www');

const filesToCopy = [
  'index.html',
  'app.js',
  'style.css',
  'fusion-engine.js',
  'fusion-panel.js',
  'manifest.json',
  'sw.js',
];

const dirsToCopy = [
  'assets',
  'data/brains',
  'models/tfjs',
  'models/tfjs_quantized',
];

function copyPath(relPath) {
  const from = join(root, relPath);
  const to = join(outDir, relPath);
  if (!existsSync(from)) {
    console.warn(`[mobile:prepare] skipping missing path: ${relPath}`);
    return;
  }
  cpSync(from, to, { recursive: true });
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

for (const file of filesToCopy) copyPath(file);
for (const dir of dirsToCopy) copyPath(dir);

console.log('[mobile:prepare] www prepared for Capacitor Android');
