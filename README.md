# LLM Learning Notes

个人语言模型与后训练学习笔记，使用 MkDocs Material 构建并发布到 GitHub Pages。

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdocs serve
```

浏览器打开 <http://127.0.0.1:8000>。

## 添加新文章

1. 将 Markdown 文件放入 `docs/` 下合适的主题目录；
2. 在 `mkdocs.yml` 的 `nav` 中加入文章；
3. 本地执行 `mkdocs serve` 检查；
4. 提交并推送到 `main`，GitHub Actions 会自动发布。

