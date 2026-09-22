"""把提交材料导出为单一文档（Word .docx + HTML 双格式）。

    python scripts/export_submission.py

产出（写入 docs/export/）：
1. minimal-agent-提交材料.docx  —— 含封面、目录域、正文、附录（真实调用证据）
2. minimal-agent-提交材料.html  —— 同一内容，浏览器打开后可「打印 → 另存为 PDF」

设计说明：
- 不依赖 pandoc / markdown 库，自带一个覆盖本项目文档所用语法的**最小 Markdown 渲染器**
  （标题、段落、有序/无序列表、表格、围栏代码块、粗体、行内代码、链接、引用）；
- 只依赖 `python-docx`（可选）；装不上时仍会生成 HTML，并提示安装命令。
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "export")
os.makedirs(OUT_DIR, exist_ok=True)

#: 文档结构：封面信息 + 正文 + 附录
DOC_TITLE = "minimal-agent · 提交材料"
DOC_SUBTITLE = "从零手写的最小可用 Agent Runtime（不使用任何 Agent 框架）"
REPO_URL = "https://github.com/zhkazhk/minimal-agent"

BODY_FILES = [("正文：提交说明", "docs/SUBMISSION.md")]
APPENDIX_FILES = [
    ("附录 A：真实调用 trace 时间线", "docs/evidence/real_llm_run.md"),
    ("附录 B：多场景调用统计", "docs/evidence/real_llm_stats.md"),
    ("附录 C：AI Prompt（模板 + 运行时渲染）", "docs/evidence/system_prompt.md"),
    ("附录 D：Memory 证据（召回时机与放置方式）", "docs/evidence/memory_trace.md"),
]


# ---------------------------------------------------------------------------
# 最小 Markdown 解析器
# ---------------------------------------------------------------------------
BLOCK_FENCE = re.compile(r"^```")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
TABLE_SEP = re.compile(r"^\|[\s:\-|]+\|\s*$")
ULIST = re.compile(r"^[-*]\s+(.*)$")
OLIST = re.compile(r"^(\d+)\.\s+(.*)$")
QUOTE = re.compile(r"^>\s?(.*)$")
RULE = re.compile(r"^-{3,}$")


def parse_blocks(text: str) -> list[tuple[str, object]]:
    """把 markdown 拆成 (类型, 内容) 序列。类型：h1..h6 / p / ul / ol / table / code / quote / hr。"""
    blocks: list[tuple[str, object]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        if BLOCK_FENCE.match(line):
            index += 1
            buf: list[str] = []
            while index < len(lines) and not BLOCK_FENCE.match(lines[index]):
                buf.append(lines[index])
                index += 1
            index += 1
            blocks.append(("code", "\n".join(buf)))
            continue

        match = HEADING.match(line)
        if match:
            blocks.append((f"h{len(match.group(1))}", match.group(2).strip()))
            index += 1
            continue

        if RULE.match(line.strip()):
            blocks.append(("hr", ""))
            index += 1
            continue

        if TABLE_ROW.match(line) and index + 1 < len(lines) and TABLE_SEP.match(lines[index + 1]):
            header = [c.strip() for c in TABLE_ROW.match(line).group(1).split("|")]
            rows: list[list[str]] = []
            index += 2
            while index < len(lines) and TABLE_ROW.match(lines[index]):
                rows.append([c.strip() for c in TABLE_ROW.match(lines[index]).group(1).split("|")])
                index += 1
            blocks.append(("table", (header, rows)))
            continue

        if ULIST.match(line):
            items: list[str] = []
            while index < len(lines) and ULIST.match(lines[index]):
                items.append(ULIST.match(lines[index]).group(1))
                index += 1
            blocks.append(("ul", items))
            continue

        if OLIST.match(line):
            items = []
            while index < len(lines) and OLIST.match(lines[index]):
                items.append(OLIST.match(lines[index]).group(2))
                index += 1
            blocks.append(("ol", items))
            continue

        if QUOTE.match(line):
            buf = []
            while index < len(lines) and QUOTE.match(lines[index]):
                buf.append(QUOTE.match(lines[index]).group(1))
                index += 1
            blocks.append(("quote", "\n".join(buf)))
            continue

        if not line.strip():
            index += 1
            continue

        buf = []
        while (
            index < len(lines)
            and lines[index].strip()
            and not HEADING.match(lines[index])
            and not BLOCK_FENCE.match(lines[index])
            and not TABLE_ROW.match(lines[index])
            and not ULIST.match(lines[index])
            and not OLIST.match(lines[index])
            and not QUOTE.match(lines[index])
            and not RULE.match(lines[index].strip())
        ):
            buf.append(lines[index])
            index += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def inline_tokens(text: str) -> list[tuple[str, str]]:
    """把行内文本切成 (类型, 文本)：plain / bold / code / link。"""
    tokens: list[tuple[str, str]] = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            tokens.append(("bold", part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            tokens.append(("code", part[1:-1]))
        elif part.startswith("[") and "](" in part:
            label, _, url = part[1:-1].partition("](")
            tokens.append(("link", f"{label}|{url}"))
        else:
            tokens.append(("plain", part))
    return tokens


def strip_inline(text: str) -> str:
    """去掉行内标记，用于 Word 里不支持富文本的位置（如表格单元格简写）。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------
CSS = """
:root { --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --code:#f6f8fa; --accent:#0969da; }
* { box-sizing: border-box; }
body { margin:0; padding:0; background:#fff; color:var(--fg);
       font-family:-apple-system,"Segoe UI","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;
       font-size:15px; line-height:1.75; }
.page { max-width:900px; margin:0 auto; padding:56px 40px 80px; }
.cover { text-align:center; padding:80px 20px 60px; border-bottom:2px solid var(--line); margin-bottom:48px; }
.cover h1 { font-size:34px; margin:0 0 12px; letter-spacing:.5px; }
.cover .sub { color:var(--muted); font-size:17px; margin-bottom:28px; }
.cover .meta { display:inline-block; text-align:left; border:1px solid var(--line); border-radius:8px;
               padding:18px 24px; background:#fbfcfd; font-size:14px; }
.cover .meta div { margin:4px 0; }
.cover .meta b { display:inline-block; min-width:96px; color:var(--muted); font-weight:600; }
h2 { font-size:24px; margin:44px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }
h3 { font-size:19px; margin:32px 0 10px; }
h4 { font-size:16px; margin:24px 0 8px; }
h5,h6 { font-size:15px; margin:20px 0 6px; color:var(--muted); }
p { margin:10px 0; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
code { background:var(--code); padding:2px 5px; border-radius:4px;
       font-family:Consolas,"Courier New",monospace; font-size:13px; }
pre { background:var(--code); border:1px solid var(--line); border-radius:8px;
      padding:14px 16px; overflow-x:auto; }
pre code { background:none; padding:0; font-size:12.5px; line-height:1.55; }
blockquote { margin:14px 0; padding:10px 18px; border-left:4px solid var(--accent);
             background:#f6f8fa; color:#24292f; border-radius:0 6px 6px 0; }
blockquote p { margin:6px 0; }
table { border-collapse:collapse; width:100%; margin:16px 0; font-size:14px; }
th,td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { background:#f6f8fa; font-weight:600; }
tr:nth-child(even) td { background:#fcfcfd; }
ul,ol { margin:10px 0; padding-left:26px; }
li { margin:4px 0; }
hr { border:0; border-top:1px solid var(--line); margin:32px 0; }
.appendix-title { page-break-before:always; }
@media print {
  .page { max-width:none; padding:0; }
  a { color:var(--fg); }
  pre,table,blockquote { page-break-inside:avoid; }
  h2,h3 { page-break-after:avoid; }
}
"""


def render_html(blocks: list[tuple[str, object]]) -> str:
    out: list[str] = []
    for kind, content in blocks:
        if kind == "code":
            out.append(f"<pre><code>{html.escape(content)}</code></pre>")
        elif kind in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = kind[1]
            out.append(f"<h{level}>{render_html_inline(str(content))}</h{level}>")
        elif kind == "hr":
            out.append("<hr/>")
        elif kind == "quote":
            out.append(f"<blockquote>{render_html_inline(str(content)).replace(chr(10), '<br/>')}</blockquote>")
        elif kind == "ul":
            items = "".join(f"<li>{render_html_inline(i)}</li>" for i in content)  # type: ignore[union-attr]
            out.append(f"<ul>{items}</ul>")
        elif kind == "ol":
            items = "".join(f"<li>{render_html_inline(i)}</li>" for i in content)  # type: ignore[union-attr]
            out.append(f"<ol>{items}</ol>")
        elif kind == "table":
            header, rows = content  # type: ignore[misc]
            head = "".join(f"<th>{render_html_inline(c)}</th>" for c in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{render_html_inline(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        else:
            out.append(f"<p>{render_html_inline(str(content))}</p>")
    return "\n".join(out)


def render_html_inline(text: str) -> str:
    parts: list[str] = []
    for kind, value in inline_tokens(text):
        if kind == "bold":
            parts.append(f"<strong>{html.escape(value)}</strong>")
        elif kind == "code":
            parts.append(f"<code>{html.escape(value)}</code>")
        elif kind == "link":
            label, _, url = value.partition("|")
            parts.append(f'<a href="{html.escape(url)}">{html.escape(label)}</a>')
        else:
            parts.append(html.escape(value))
    return "".join(parts)


def build_html() -> str:
    sections = [
        f"""<div class="cover">
  <h1>{html.escape(DOC_TITLE)}</h1>
  <div class="sub">{html.escape(DOC_SUBTITLE)}</div>
  <div class="meta">
    <div><b>代码仓库</b> <a href="{REPO_URL}">{REPO_URL}</a></div>
    <div><b>运行期依赖</b> 零（纯 Python 标准库）</div>
    <div><b>测试</b> 253 个用例通过 · 验收用例 11/11</div>
    <div><b>真实 LLM</b> DeepSeek deepseek-flash · 端到端 10/10 通过</div>
  </div>
</div>"""
    ]
    for title, path in BODY_FILES + APPENDIX_FILES:
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            continue
        with open(full, "r", encoding="utf-8") as fh:
            text = fh.read()
        cls = ' class="appendix-title"' if (title, path) in APPENDIX_FILES else ""
        sections.append(f'<section{cls}><h2>{html.escape(title)}</h2>\n{render_html(parse_blocks(text))}</section>')
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(DOC_TITLE)}</title>
<style>{CSS}</style>
</head>
<body><div class="page">
{body}
</div></body>
</html>
"""


# ---------------------------------------------------------------------------
# Word 渲染
# ---------------------------------------------------------------------------
def build_docx(path: str) -> tuple[bool, str]:
    try:
        import docx  # noqa: F401
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt, RGBColor
    except ImportError:
        return False, "未安装 python-docx"

    doc = Document()

    # 基础样式
    normal = doc.styles["Normal"]
    normal.font.name = "Segoe UI"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    for name, size in (("Heading 1", 20), ("Heading 2", 15), ("Heading 3", 12.5), ("Heading 4", 11)):
        style = doc.styles[name]
        style.font.name = "Segoe UI"
        style.font.color.rgb = RGBColor(0x1F, 0x23, 0x28)
        style.font.size = Pt(size)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    def add_page_number_footer() -> None:
        footer = doc.sections[0].footer
        para = footer.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run()
        for instr in ("begin", "PAGE", "end"):
            element = OxmlElement("w:fldChar") if instr != "PAGE" else OxmlElement("w:instrText")
            if instr == "PAGE":
                element.set(qn("xml:space"), "preserve")
                element.text = " PAGE "
            else:
                element.set(qn("w:fldCharType"), instr)
            run._r.append(element)

    def add_inline(para, text: str, *, code_font: bool = False) -> None:
        for kind, value in inline_tokens(text):
            if kind == "code":
                run = para.add_run(value)
                run.font.name = "Consolas"
                run.font.size = Pt(9.5)
            elif kind == "bold":
                para.add_run(value).bold = True
            elif kind == "link":
                label, _, url = value.partition("|")
                run = para.add_run(f"{label} ({url})" if label != url else url)
                run.font.color.rgb = RGBColor(0x09, 0x69, 0xDA)
            else:
                para.add_run(value)

    def add_code_block(text: str) -> None:
        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Pt(10)
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(8)
        run = para.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x24, 0x29, 0x2F)
        # 浅灰底纹
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:fill"), "F6F8FA")
        para._p.get_or_add_pPr().append(shading)

    # ---------------- 封面 ----------------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(DOC_TITLE)
    run.bold = True
    run.font.size = Pt(26)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(DOC_SUBTITLE)
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)
    doc.add_paragraph()

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for label, value in (
        ("代码仓库", REPO_URL),
        ("运行期依赖", "零（纯 Python 标准库）"),
        ("离线测试", "253 个用例通过 · 验收用例 11/11"),
        ("真实 LLM", "DeepSeek deepseek-flash · 端到端 10/10 通过"),
    ):
        line = meta.add_run(f"{label}：{value}")
        line.font.size = Pt(10.5)
        meta.add_run("\n")
    doc.add_page_break()

    # ---------------- 目录域 ----------------
    head = doc.add_paragraph()
    head.add_run("目录").bold = True
    head.runs[0].font.size = Pt(16)
    toc = doc.add_paragraph()
    run = toc.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "（在 Word 中按 Ctrl+A 然后 F9 可更新目录）"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for element in (fld_begin, instr, fld_sep, placeholder, fld_end):
        run._r.append(element)
    doc.add_page_break()

    # ---------------- 正文 + 附录 ----------------
    for idx, (title_text, rel_path) in enumerate(BODY_FILES + APPENDIX_FILES):
        full = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full):
            continue
        is_appendix = (title_text, rel_path) in APPENDIX_FILES
        if idx > 0:
            doc.add_page_break()
        doc.add_heading(title_text, level=1)
        with open(full, "r", encoding="utf-8") as fh:
            blocks = parse_blocks(fh.read())
        for kind, content in blocks:
            if kind == "code":
                add_code_block(str(content))
            elif kind == "hr":
                doc.add_paragraph("─" * 46).alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif kind.startswith("h") and kind[1:].isdigit():
                level = min(int(kind[1]) + 1, 4)   # markdown h1 → Word Heading 2（避免与章节标题同级）
                doc.add_heading(strip_inline(str(content)), level=level)
            elif kind == "quote":
                para = doc.add_paragraph()
                para.paragraph_format.left_indent = Pt(16)
                add_inline(para, str(content).replace("\n", " "))
                for r in para.runs:
                    r.italic = True
                    r.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)
            elif kind in ("ul", "ol"):
                style = "List Bullet" if kind == "ul" else "List Number"
                for item in content:  # type: ignore[union-attr]
                    add_inline(doc.add_paragraph(style=style), item)
            elif kind == "table":
                header, rows = content  # type: ignore[misc]
                table = doc.add_table(rows=1, cols=len(header))
                table.style = "Light Grid Accent 1"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for cell, text in zip(table.rows[0].cells, header):
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(strip_inline(text))
                    run.bold = True
                    run.font.size = Pt(9.5)
                for row in rows:
                    cells = table.add_row().cells
                    for cell, text in zip(cells, row):
                        cell.text = ""
                        run = cell.paragraphs[0].add_run(strip_inline(text))
                        run.font.size = Pt(9)
                doc.add_paragraph()
            else:
                add_inline(doc.add_paragraph(), str(content))

    add_page_number_footer()
    doc.save(path)
    return True, path


def export_pdf(docx_path: str, pdf_path: str) -> tuple[bool, str]:
    """用 Word COM 把 docx 转成 PDF（仅 Windows + 装了 Word 时可用）。

    两个环境相关的坑（都在实际使用中踩过）：
    1. **管道 stdio 被禁**：某些受限环境禁止子进程管道通信，Python 用
       `subprocess(capture_output=True)` 派发 PowerShell 会 EPERM。因此这里
       先试进程内 COM，再试**继承 stdio** 的 PowerShell，最后打印可复制的手动命令。
    2. **目标文件被占用**：PDF 阅读器（如 WPS 的 wpspdf.exe）会预读目录里的 PDF，
       导致 Word 无法覆盖同名文件并报 `E_FAIL`。因此这里**先导出到临时名**，
       成功后再替换；替换失败就保留临时名并明确告知用户。

    手动命令（沙箱内最可靠）：

        $w = New-Object -ComObject Word.Application; $w.Visible = $false
        $d = $w.Documents.Open('<docx>', $false, $true)
        foreach ($t in $d.TablesOfContents) { $t.Update() }   # 更新目录页码
        $d.ExportAsFixedFormat('<pdf>', 17)                   # 17 = wdExportFormatPDF
        $d.Close(0); $w.Quit()
    """
    if os.name != "nt":
        return False, "非 Windows 平台"

    def convert(target: str) -> bool:
        """把 docx 导出为 target 指定的 PDF；成功返回 True。"""
        # 方式一：进程内 COM（需 pywin32）
        try:
            import win32com.client  # type: ignore

            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            try:
                doc = word.Documents.Open(docx_path, False, True)
                doc.Fields.Update()
                for toc in doc.TablesOfContents:
                    toc.Update()
                doc.ExportAsFixedFormat(target, 17)
                doc.Close(0)
            finally:
                word.Quit()
            return os.path.isfile(target)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001 - 落到方式二
            if os.path.isfile(target):
                return True

        # 方式二：PowerShell（stdio 继承，不做捕获）
        for shell in ("pwsh", "powershell"):
            script = (
                "$ErrorActionPreference='Stop';"
                "$w=New-Object -ComObject Word.Application;$w.Visible=$false;$w.DisplayAlerts=0;"
                "try{"
                f"$d=$w.Documents.Open('{docx_path}',$false,$true);"
                "$d.Fields.Update()|Out-Null;"
                "foreach($t in $d.TablesOfContents){$t.Update()};"
                f"$d.ExportAsFixedFormat('{target}',17);"
                "$d.Close(0)"
                "}finally{$w.Quit()}"
            )
            try:
                subprocess.run([shell, "-NoProfile", "-Command", script], timeout=300)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if os.path.isfile(target):
                return True
        return False

    # 先导出到临时名，避免"目标被 PDF 阅读器占用"导致 Word 报 E_FAIL
    temp_path = pdf_path + ".tmp.pdf"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    if not convert(temp_path):
        return False, "Word COM / PowerShell 均不可用（请用下方手动命令，或用 Word 另存为 PDF）"

    try:
        os.replace(temp_path, pdf_path)
        return True, pdf_path
    except OSError as exc:
        # 目标被占用：保留临时名，并明确告知（不是静默失败）
        alt = pdf_path.replace(".pdf", "-最新.pdf")
        try:
            os.replace(temp_path, alt)
            return True, alt
        except OSError:
            return False, f"PDF 已生成但无法替换目标文件（{exc.strerror}）：{temp_path}"


def pdf_command(docx_name: str, pdf_name: str) -> str:
    """打印可手动执行的 PowerShell 片段（供受限沙箱使用）。"""
    return (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false\n"
        f"$d = $w.Documents.Open('{docx_name}', $false, $true)\n"
        "foreach ($t in $d.TablesOfContents) { $t.Update() }\n"
        f"$d.ExportAsFixedFormat('{pdf_name}', 17)\n"
        "$d.Close(0); $w.Quit()"
    )


def main() -> int:
    html_path = os.path.join(OUT_DIR, "minimal-agent-提交材料.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(build_html())
    size_kb = os.path.getsize(html_path) / 1024
    print(f"✅ HTML 已生成：{os.path.relpath(html_path, ROOT)}（{size_kb:.0f} KB）")

    docx_path = os.path.join(OUT_DIR, "minimal-agent-提交材料.docx")
    ok, detail = build_docx(docx_path)
    if not ok:
        print(f"⚠ Word 未生成（{detail}）—— 可执行 `pip install python-docx` 后重跑")
        return 1
    size_kb = os.path.getsize(docx_path) / 1024
    print(f"✅ Word 已生成：{os.path.relpath(docx_path, ROOT)}（{size_kb:.0f} KB）")

    pdf_path = os.path.join(OUT_DIR, "minimal-agent-提交材料.pdf")
    ok, detail = export_pdf(docx_path, pdf_path)
    if ok:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"✅ PDF  已生成：{os.path.relpath(pdf_path, ROOT)}（{size_kb:.0f} KB，目录页码已更新）")
    else:
        print(f"⚠ PDF 未生成（{detail}）")
        print("   可在 PowerShell 里直接执行下面这段生成 PDF：\n")
        print("   " + pdf_command(docx_path, pdf_path).replace("\n", "\n   "))
        print("\n   或：浏览器打开 HTML → Ctrl+P → 目标选「另存为 PDF」")

    print("\n说明：")
    print("  · Word 里目录处按 Ctrl+A 再按 F9 可重新生成带页码的目录")
    print("  · HTML 自带打印样式（表格/代码块不跨页断开），适合浏览器直接打印")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
