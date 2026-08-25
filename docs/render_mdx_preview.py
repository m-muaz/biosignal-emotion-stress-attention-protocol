#!/usr/bin/env python3
"""Render a plain-Markdown/MDX file (tables + ```mermaid fences, no JSX) to a
local HTML preview. marked.js + mermaid.js are loaded from docs/vendor/ via
relative <script src> (not inlined) so Live Server / other injectors cannot
split the page on a `</body>` string inside the mermaid bundle.

Usage:
    python docs/render_mdx_preview.py docs/label_design.mdx
    # writes docs/label_design.preview.html next to the source, open it in a browser

Re-vendoring the libraries (only needed if upgrading marked/mermaid):
    cd /tmp && npm init -y && npm install marked mermaid --no-audit --no-fund
    cp node_modules/marked/lib/marked.umd.js   <repo>/docs/vendor/marked.umd.js
    cp node_modules/mermaid/dist/mermaid.min.js <repo>/docs/vendor/mermaid.min.js
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "vendor"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — Local Preview</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="{marked_src}"></script>
<script src="{mermaid_src}"></script>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --border: #e5e7eb;
    --code-bg: #f4f4f5; --accent: #2563eb; --table-stripe: #fafafa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1115; --fg:#e5e7eb; --muted:#9ca3af; --border:#2a2e37; --code-bg:#1a1d24; --accent:#60a5fa; --table-stripe:#171a20; }}
  }}
  body {{ background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 980px; margin: 0 auto; padding: 2.5rem 1.5rem 6rem; }}
  h1,h2,h3 {{ line-height: 1.25; }}
  h1 {{ border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: 0.92rem; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.7rem; text-align: left; vertical-align: top; }}
  th {{ background: var(--code-bg); }}
  tr:nth-child(even) {{ background: var(--table-stripe); }}
  code {{ background: var(--code-bg); padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }}
  pre code {{ display: block; padding: 1rem; overflow-x: auto; }}
  a {{ color: var(--accent); }}
  img {{ max-width: 100%; height: auto; }}
  .mermaid {{ background: var(--code-bg); border-radius: 8px; padding: 1rem; margin: 1.2rem 0; overflow-x: auto; }}
  .mermaid svg {{ width: auto !important; max-width: none !important; height: auto; }}
  blockquote {{ border-left: 3px solid var(--accent); margin: 1rem 0; padding: 0.2rem 1rem; color: var(--muted); }}
</style>
</head>
<body>
<div id="content">Loading…</div>
<script>
  const escapeHtml = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const raw = {md_json};
  try {{
    mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose', theme: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default' }});
    const renderer = new marked.Renderer();
    renderer.code = (token) => {{
      const code = token.text || '';
      const lang = (token.lang || '').trim();
      if (lang === 'mermaid') {{
        return `<div class="mermaid">${{escapeHtml(code)}}</div>`;
      }}
      return `<pre><code>${{escapeHtml(code)}}</code></pre>`;
    }};
    marked.setOptions({{ renderer, gfm: true, tables: true }});
    document.getElementById('content').innerHTML = marked.parse(raw);
    mermaid.run({{ querySelector: '.mermaid' }}).catch((err) => console.error(err));
  }} catch (err) {{
    document.getElementById('content').textContent = String(err);
  }}
</script>
</body>
</html>
"""


def _vendor_url(out_path: Path, name: str) -> str:
    rel = os.path.relpath(VENDOR_DIR / name, start=out_path.parent)
    return Path(rel).as_posix()


def render(src_path: Path) -> Path:
    md_content = src_path.read_text(encoding="utf-8")
    out_path = src_path.with_suffix(".preview.html")
    md_json = json.dumps(md_content).replace("<", "\\u003c")
    out_path.write_text(
        HTML_TEMPLATE.format(
            title=src_path.stem,
            md_json=md_json,
            marked_src=_vendor_url(out_path, "marked.umd.js"),
            mermaid_src=_vendor_url(out_path, "mermaid.min.js"),
        ),
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    out = render(src)
    print(f"wrote {out} -- open it in a browser to view")
