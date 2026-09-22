# 密钥投放区（整个目录已被 git 忽略，不会进版本库）

把含密钥的文件丢到这个目录里，然后告诉我文件名，我会执行：

```bash
python scripts/import_key.py .inbox/<你的文件名>
```

## 支持的三种格式

**1）`.env` / properties 风格（推荐）**

```ini
MINIAGENT_API_KEY=sk-xxxxxxxxxxxxxxxx
MINIAGENT_BASE_URL=https://api.deepseek.com/v1
MINIAGENT_MODEL=deepseek-chat
MINIAGENT_PROVIDER=deepseek
```

**2）整个文件就是一把 key**

```
sk-xxxxxxxxxxxxxxxx
```

**3）JSON**

```json
{"api_key": "sk-xxxx", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"}
```

## 安全说明

- 导入后密钥只写入 `.env`，该文件被 `.gitignore` 明确忽略（已验证 `git check-ignore`）；
- 导入脚本**只打印密钥的前 4 位与长度**，不会回显完整内容；
- 常用网关地址：
  - DeepSeek：`https://api.deepseek.com/v1`（模型 `deepseek-chat`）
  - OpenAI：`https://api.openai.com/v1`（模型 `gpt-4o-mini`）
  - 通义千问：`https://dashscope.aliyuncs.com/compatible-mode/v1`（模型 `qwen-plus`）
  - 本地 Ollama：`http://localhost:11434/v1`（模型 `qwen2.5:7b`，无需 Key）
