---
name: auto-commit
description: Use this skill whenever the user asks to commit code, save changes to git, push code, or run a git commit. This skill will automatically review the code changes, generate a Vietnamese commit message adhering to the project's Git.md conventions, and immediately execute the git commit command without asking for confirmation.
---

# Auto Commit Skill

When the user asks you to commit their code, you will automate the staging, reviewing, message generation, and committing process.

## Step 1: Check Git Status and Stage Changes
First, examine the current repository state:
1. Execute `git status`.
2. Determine if changes are already staged.
3. If there are unstaged or untracked files and the user hasn't specified exactly what to commit, automatically stage all changes by executing `git add .`.

## Step 2: Analyze the Changes (Review)
1. Execute `git diff --cached` to see exactly what has been staged.
2. Read the diff to understand what files were modified and what the changes accomplished.

## Step 3: Generate Commit Message
Based on the diff and the rules defined in `Git.md`, internally prepare a commit message in **Vietnamese**. The format MUST strictly follow:

```text
<type>(<scope>): <title>
scopes: <scopes>

break:
- ... 
change:
- ...
new:
- ...
```

**Các thành phần trong message:**
- `type`: vai trò của commit, thuộc các loại sau:
  - `fix`: fixbug
  - `feat`: bổ sung tính năng, yêu cầu mới
  - `refactor`: sửa đổi cấu trúc code, không ảnh hưởng logic / hiệu năng
  - `perf`: chỉnh sửa liên quan tới cải tiến hiệu năng
  - `chore`: sửa chút ít, không ảnh hưởng tới luồng / hiệu năng
  - `doc`: bổ sung / điều chỉnh tài liệu
- `scope`: tên tính năng (feature name) hoặc module đang xử lý (VD: auth, order, database, ...)
- `title`: tóm tắt ngắn gọn nội dung trong commit
- `scopes`: các commitId của commit có liên quan tới commit này (nếu không có thì bỏ qua)
- `break`: tóm tắt các thay đổi liên quan mà không tương thích ngược (nếu không có thì bỏ qua)
- `change`: tóm tắt các sửa đổi logic, vẫn đảm bảo tương thích ngược (nếu không có thì bỏ qua)
- `new`: tóm tắt các bổ sung (nếu không có thì bỏ qua)

**Ví dụ minh họa:**
```text
feat(database): Khởi tạo csdl và các lớp thực thể

new:
- Add liquibase cho migration
- Tạo các lớp thực thể
change:
- Dùng .env thay vì hardcode trong application.yaml
- Đổi sang mariadb thay cho mysql
- Bỏ một số field không cần thiết của user
```

*(Lưu ý: Chỉ đưa vào commit message những section (break / change / new / scopes) thực sự có nội dung, phần nào không có thì phải bỏ qua).*

## Step 4: Execute the Commit
You MUST execute the commit automatically without waiting for user confirmation.
Because the generated message is multi-line, use the following Heredoc format to safely run the command:

```bash
git commit -F - <<'EOF'
<The multiline commit message you generated>
EOF
```

## Step 5: Output
Display the result to the user in this format:

**📝 Tóm tắt thay đổi (Code Review):**
[Your concise Vietnamese Review / Summary of changes]

**✅ Đã tự động commit với message:**
```text
[The exact git commit message you used]
```
