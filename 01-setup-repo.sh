#!/usr/bin/env bash

# 01-setup-repo.sh
# Создание/проверка локального репозитория и первичная отправка в GitHub.
#
# Использование:
#   ./01-setup-repo.sh git@github.com:USER/REPO.git
#
# Переменные окружения при необходимости:
#   BRANCH=main
#   INITIAL_MESSAGE="Initial commit"
#   GIT_USER_NAME="Your Name"
#   GIT_USER_EMAIL="you@example.com"

set -Eeuo pipefail

REMOTE_URL="${1:-git@github.com:USER/REPO.git}"
BRANCH="${BRANCH:-main}"
INITIAL_MESSAGE="${INITIAL_MESSAGE:-Initial commit}"
GIT_USER_NAME="${GIT_USER_NAME:-Your Name}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-you@example.com}"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup][error] %s\n' "$*" >&2; exit 1; }

if [[ "$REMOTE_URL" == *"USER/REPO.git" ]]; then
  die "Замените USER/REPO.git на реальный адрес. Пример: ./01-setup-repo.sh git@github.com:login/project.git"
fi

if [[ "$PWD" == "$HOME" && ! -d .git && "${FORCE_SETUP_IN_HOME:-0}" != "1" ]]; then
  die "Отказ: текущий каталог похож на домашний. Перейдите в папку проекта или запустите с FORCE_SETUP_IN_HOME=1."
fi

# 1. Инициализируем репозиторий, если его ещё нет.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "Git-репозиторий уже инициализирован."
else
  log "Создаю git-репозиторий..."
  if ! git init -b "$BRANCH" >/dev/null 2>&1; then
    # fallback для старых версий git
    git init >/dev/null
    git symbolic-ref HEAD "refs/heads/$BRANCH"
  fi
fi

# 2. Проверяем/ставим локальные имя и email для коммитов.
if [[ -z "$(git config user.name || true)" ]]; then
  git config user.name "$GIT_USER_NAME"
fi

if [[ -z "$(git config user.email || true)" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

# 3. Добавляем remote origin, если его ещё нет.
if git remote get-url origin >/dev/null 2>&1; then
  CURRENT_REMOTE="$(git remote get-url origin)"
  log "origin уже задан: $CURRENT_REMOTE"

  if [[ "$CURRENT_REMOTE" != "$REMOTE_URL" ]]; then
    log "Если нужно использовать другой адрес: git remote set-url origin $REMOTE_URL"
  fi
else
  git remote add origin "$REMOTE_URL"
  log "Добавлен origin: $REMOTE_URL"
fi

# 4. Делаем первоначальный коммит только если коммитов ещё нет.
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  if [[ ! -f .gitignore ]]; then
    cat > .gitignore <<'EOF'
# Секреты и локальные настройки
.env
.env.*
*.pem
*.key
credentials*

# ОС
.DS_Store
Thumbs.db

# Редакторы
.idea/
.vscode/
EOF
    log "Создан .gitignore."
  fi

  git add -A

  if git diff --cached --quiet; then
    git commit --allow-empty -m "$INITIAL_MESSAGE"
  else
    git commit -m "$INITIAL_MESSAGE"
  fi

  log "Создан первоначальный коммит."
else
  log "Коммиты уже есть. Этот скрипт только проверяет remote и делает push."
fi

# 5. Отправляем ветку в GitHub.
log "Отправляю в GitHub..."
if git push -u origin "$BRANCH"; then
  log "Готово."
else
  die "Push не удался. Проверьте, что репозиторий на GitHub создан, а адрес правильный. Если на GitHub уже есть коммиты, выполните: git pull --rebase origin $BRANCH (или git pull origin $BRANCH --allow-unrelated-histories), затем повторите."
fi