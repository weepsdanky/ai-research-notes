#!/usr/bin/env node
// Render Markdown with LaTeX math ($...$ and $$...$$) to PDF via KaTeX + Puppeteer
// Usage: node render_pdf.js <input.md> [output.pdf]

const fs = require('fs');
const path = require('path');
const os = require('os');
const puppeteer = require('puppeteer-core');
const markdownIt = require('markdown-it');
const texmath = require('markdown-it-texmath');
const katex = require('katex');

// macOS: use Microsoft Edge (Chromium) or detect Chrome
const CHROME_PATHS = [
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
];

function findChrome() {
  for (const p of CHROME_PATHS) {
    if (fs.existsSync(p)) return p;
  }
  console.error('No Chromium browser found. Tried:', CHROME_PATHS.join(', '));
  process.exit(1);
}

// ── Pre-flight ────────────────────────────────────────────────────────────────
function preflightCheck(text) {
  const hasCJK = /[　-鿿豈-﫿＀-￯]/.test(text);
  const hasGreek = /[Ͱ-Ͽ]/.test(text);
  const hasMath = /[∀-⋿℀-⅏]/.test(text);
  if (hasCJK) console.log('[preflight] CJK characters detected → Noto Serif SC');
  if (hasGreek) console.log('[preflight] Greek letters detected');
  if (hasMath) console.log('[preflight] Math symbols detected');
  return { hasCJK };
}

// ── Post-render ───────────────────────────────────────────────────────────────
async function postRenderCheck(page) {
  const tofuCount = await page.evaluate(() => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let count = 0, node;
    while ((node = walker.nextNode())) {
      count += (node.textContent.match(/�/g) || []).length;
    }
    return count;
  });
  if (tofuCount > 0) {
    console.warn(`[post-render] WARNING: ${tofuCount} replacement character(s) found.`);
  } else {
    console.log('[post-render] No tofu characters detected.');
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
const inputFile = process.argv[2];
if (!inputFile) {
  console.error('Usage: node render_pdf.js <input.md> [output.pdf]');
  process.exit(1);
}

const outputFile = process.argv[3] || inputFile.replace(/\.md$/, '.pdf');
const markdown = fs.readFileSync(inputFile, 'utf8');

console.log(`[render] Input:  ${inputFile}`);
console.log(`[render] Output: ${outputFile}`);

preflightCheck(markdown);

const md = markdownIt({ html: true, linkify: true, typographer: true })
  .use(texmath, {
    engine: katex,
    delimiters: 'dollars',
    katexOptions: { macros: { '\\RR': '\\mathbb{R}' } },
  });

const body = md.render(markdown);

const katexCssPath = require.resolve('katex/dist/katex.min.css');
const katexFontsDir = path.dirname(require.resolve('katex/dist/fonts/KaTeX_AMS-Regular.woff2'));
const katexCss = fs.readFileSync(katexCssPath, 'utf8')
  .replace(/url\(fonts\//g, `url(file://${katexFontsDir}/`)
  .replace(/url\("fonts\//g, `url("file://${katexFontsDir}/`)
  .replace(/url\('fonts\//g, `url('file://${katexFontsDir}/`);

const googleFontsUrl = 'https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap';

const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="${googleFontsUrl}" rel="stylesheet">
<style>
${katexCss}
body {
  font-family: 'Noto Serif SC', 'Noto Serif', 'DejaVu Serif', Georgia, serif;
  font-size: 13px;
  max-width: 820px;
  margin: 40px auto;
  line-height: 1.8;
  color: #111;
}
h1 { font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 1.5em; }
h2 { font-size: 1.3em; margin-top: 2em; }
h3 { font-size: 1.1em; margin-top: 1.5em; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f4f4f4; padding: 12px; border-radius: 5px; overflow-x: auto; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; }
th { background: #f0f0f0; }
.katex-display { margin: 1em 0; overflow-x: auto; }
strong { font-weight: bold; }
</style>
</head>
<body>
${body}
</body>
</html>`;

(async () => {
  const chromePath = findChrome();
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const tmpHtml = path.join(os.tmpdir(), `render_pdf_${process.pid}.html`);
  fs.writeFileSync(tmpHtml, html, 'utf8');

  try {
    const page = await browser.newPage();
    console.log('[render] Loading HTML and waiting for fonts...');
    await page.goto(`file://${tmpHtml}`, { waitUntil: 'networkidle0', timeout: 30000 });

    await postRenderCheck(page);

    console.log('[render] Generating PDF...');
    await page.pdf({
      path: outputFile,
      format: 'A4',
      margin: { top: '20mm', bottom: '20mm', left: '20mm', right: '20mm' },
      printBackground: true,
    });

    console.log(`[render] Done: ${path.resolve(outputFile)}`);
  } finally {
    await browser.close();
    fs.unlinkSync(tmpHtml);
  }
})();
