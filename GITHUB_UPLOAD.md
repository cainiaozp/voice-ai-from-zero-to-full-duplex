# 推送 GitHub（GitHub CLI）

## 推荐：一条命令发布

先在自己的电脑或服务器安装 [GitHub CLI](https://cli.github.com/)，**不要发送 GitHub 密码**。然后运行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
bash scripts/publish_with_gh.sh
```

脚本会检查课程文件、初始化 Git、提交并创建公开的 `voice-ai-from-zero-to-full-duplex` 仓库后推送。

如需先预览，不进行任何提交或联网操作：

```bash
bash scripts/publish_with_gh.sh --dry-run
```

如需改成私有仓库：

```bash
GH_REPO_VISIBILITY=private bash scripts/publish_with_gh.sh
```

如果需要自定义仓库名：

```bash
bash scripts/publish_with_gh.sh my-voice-ai-course
```

**隐私提醒**：为了避免公开私人邮箱，脚本仅在本地 Git 身份未配置时使用 GitHub noreply 邮箱。如已设置私人邮箱且希望隐藏，请先使用 GitHub noreply 邮箱更新本仓库的 `git config --local user.email`，再运行发布脚本。

---

# 推送 GitHub

## 1. 解压并检查

```bash
unzip voice-ai-from-zero-to-full-duplex.zip
cd voice-ai-from-zero-to-full-duplex
python scripts/verify_repo.py
```

**上传前检查**：`git status --short` 和 `git diff --cached --stat`，确认不含自己的录音、密钥或数据集。

## 2. 方法一：GitHub CLI（推荐）

先安装并登录 `gh`：

```bash
gh auth login
```

在仓库根目录执行（把 `YOUR_REPO_NAME` 换成希望使用的名字）：

```bash
git init -b main
git add .
git commit -m "docs: publish 10-part voice AI course and reproducible labs"
gh repo create YOUR_REPO_NAME --public --source=. --remote=origin --push
```

## 3. 方法二：先在 GitHub 创建空仓库

```bash
git init -b main
git add .
git commit -m "Initial course release"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

如你已经在已有仓库内操作，不要再次执行 `git init`；把本项目内容拷入并正常提交即可。

## 4. 推送后检查

- README 首页的10课链接是否能打开。
- 课程目录 `code/` 是否都有相应 Python 文件。
- 本地 `python scripts/verify_repo.py` 是否通过。
- 私有音频是否没有上传；`samples/` 仅包含合成音调。
- 如果需要 GitHub Pages，后续可以将 `docs/` 转换为 MkDocs 站点；当前默认在 GitHub Markdown 中直接阅读。

## 5. 常见问题

如果 GitHub 提示单文件过大，请确认没有加入模型权重、原始大型 WAV 或数据集。默认 `.gitignore` 排除这些文件。

本仓库只将自编的课程材料和教学代码以 MIT 发布；论文、预训练权重、开源项目以及用户自行下载的数据集遵循其原有许可。
