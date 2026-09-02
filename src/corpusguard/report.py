from __future__ import annotations

import html
import json
from pathlib import Path

from .models import AuditReport


def write_json(report: AuditReport, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def write_html(report: AuditReport, path: str | Path) -> Path:
    target = Path(path)
    metrics = report.metrics.to_dict()
    metric_rows = "\n".join(
        f"<tr><td>{html.escape(key.replace('_', ' ').title())}</td><td>{value}</td></tr>"
        for key, value in metrics.items()
    )
    finding_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(f.severity)}</td>"
        f"<td>{html.escape(f.code)}</td>"
        f"<td>{'' if f.row is None else f.row}</td>"
        f"<td>{html.escape(f.message)}</td>"
        "</tr>"
        for f in report.findings[:500]
    ) or '<tr><td colspan="4">No findings.</td></tr>'
    recommendations = "".join(
        f"<li>{html.escape(item)}</li>" for item in report.recommendations
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CorpusGuard Report</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#f6f7f9;color:#16181d}}
main{{max-width:1100px;margin:48px auto;padding:0 24px}}
.hero{{background:#111827;color:#fff;border-radius:18px;padding:28px}}
.score{{font-size:56px;font-weight:800}} .muted{{color:#9ca3af}}
.grid{{display:grid;grid-template-columns:1fr 2fr;gap:24px;margin-top:24px}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px;overflow:auto}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:9px;border-bottom:1px solid #eee;text-align:left}}
code{{background:#f3f4f6;padding:2px 5px;border-radius:5px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head><body><main>
<section class="hero"><div class="muted">CorpusGuard audit</div><h1>{html.escape(report.dataset)}</h1>
<div class="score">{report.score}/100 · {report.grade}</div></section>
<div class="grid"><section class="card"><h2>Metrics</h2><table>{metric_rows}</table></section>
<section class="card"><h2>Recommendations</h2><ul>{recommendations}</ul></section></div>
<section class="card" style="margin-top:24px"><h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Code</th><th>Row</th><th>Message</th></tr></thead>
<tbody>{finding_rows}</tbody></table></section>
</main></body></html>"""
    target.write_text(document, encoding="utf-8")
    return target
