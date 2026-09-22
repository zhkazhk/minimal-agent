# 导出产物（提交用文档）

本目录由 `python scripts/export_submission.py` 生成，含同一份内容的三种格式：

| 文件 | 用途 | 说明 |
| --- | --- | --- |
| `minimal-agent-提交材料.docx` | **主提交文件** | Word。含封面、目录域、正文（提交说明）、附录 A~D（真实调用证据）。打开后目录处按 `Ctrl+A` 再按 `F9` 可生成带页码的目录 |
| `minimal-agent-提交材料.pdf` | 投递 / 打印 | 由上述 docx 用 Word 导出（29 页）。目录页码已更新 |
| `minimal-agent-提交材料.html` | 预览 / 自行转 PDF | 自带打印样式（表格与代码块不跨页断开）。浏览器打开后 `Ctrl+P` → 目标选「另存为 PDF」 |

> 若同时存在 `-最新.pdf`：说明生成时另一个 PDF 阅读器（如 WPS 的 `wpspdf.exe`）占用了
> 目标文件名，新文件便以 `-最新` 后缀落盘。关闭阅读器后重跑脚本即可合并成一个。

## 内容结构

```
封面（标题 / 代码仓库 / 依赖 / 测试数 / 真实 LLM 结论）
目录（Word 域，F9 更新）
正文：提交说明
  0. 一览表                       ← 四项提交要求 → 交付位置
  ① 真实 LLM API                  ← 三次证据：trace 时间线 / 多场景统计 / 10 项端到端验证
  ② 代码链接
  ③ README（章节导航）
  ④ AI Prompt 与问题解决记录
  复现全部结论
附录 A：真实调用 trace 时间线        ← docs/evidence/real_llm_run.md
附录 B：多场景调用统计              ← docs/evidence/real_llm_stats.md
附录 C：AI Prompt（模板 + 渲染结果） ← docs/evidence/system_evidence.md
附录 D：Memory 证据                 ← docs/evidence/memory_trace.md
```

## 重新生成

```bash
python scripts/export_submission.py
```

脚本会自动：读取 `docs/SUBMISSION.md` 与 `docs/evidence/*.md` → 渲染为 HTML 与 docx
→ 尝试用 Word COM 导出 PDF。若当前环境不允许（例如受限沙箱禁止子进程管道通信），
脚本会打印一段可直接粘贴到 PowerShell 的命令来完成 PDF 导出。

**不改文档、只重出 PDF**（Windows + Word）：

```powershell
cd docs\export
$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
$d = $w.Documents.Open((Resolve-Path 'minimal-agent-提交材料.docx').Path, $false, $true)
foreach ($t in $d.TablesOfContents) { $t.Update() }
$d.ExportAsFixedFormat((Join-Path (Get-Location) 'minimal-agent-提交材料.pdf'), 17)
$d.Close(0); $w.Quit()
```

> `17` 是 Word 的 `wdExportFormatPDF` 常量。
