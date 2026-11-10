# 上传到 GitHub

仓库：`TimYu111/Earning-recreation-lab`。
简介：`NLP research on earnings-event returns with TF-IDF, Ridge regression, chronological validation, and Modal deployment.`

这是整理好的独立目录，不要把整个 starter-modal、.venv 或研究档案拖进去。
保留 `.gitignore` 和 `.env.example`；不要添加真实 .env、data、models、artifacts 或任何历史密钥。
notebook 已清除输出。results 只包含聚合实验指标。

先解压、阅读 README；确认展示的研究结果和自己的贡献描述准确，再创建 GitHub 仓库。
Public 可用于作品展示；Private 可先整理后再公开。首次发布不包含模型文件，v2 部署状态也未标为成功。

在解压后的项目目录运行（先用自己的 URL 替换最后一条命令里的占位地址）：

```bash
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Add earnings reaction research and deployment code"
git branch -M main
git remote add origin https://github.com/TimYu111/Earning-recreation-lab.git
git push -u origin main
```

GitHub 上先创建空仓库，不自动生成 README，避免首次 push 发生历史冲突。
不要把 GitHub token 发到聊天或写进 remote URL。使用已有的 GitHub 登录方式完成认证。

README 可以放在简历 Projects 栏链接中。描述建议强调时间划分、官方评分对齐、字符特征实验和部署验证，不要称为已验证盈利的交易策略。
