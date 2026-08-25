#!/usr/bin/env bash

# 02-push-updates.sh
# Добавить изменения, создать коммит, подтянуть изменения с GitHub и отправить коммиты.
#
# Использование:
#   ./02-push-updates.sh
#   ./02-push-updates.sh "Add feature X"
#
# Переменные окружения при необходимости:
#   BRANCH=main

set -Eeuo pipefail

MESSAGE="${1:-Update $(date '+%Y-%m-%d %H:%M:%S')}"

log() { printf '[push] %s\n' "$*"; }
die() { printf '[push][error] %s\n' "$*" >&2; exit 1; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "Не найден git-репозиторий. Сначала запустите 01-setup-repo.sh в этой папке."
fi

if ! git symbolic-ref -q HEAD >/dev/null; then
  die "Сейчас не на ветке, возможно detached HEAD. Переключитесь на ветку, например: git switch main"
fi

BRANCH="${BRANCH:-$(git symbolic-ref --short HEAD)}"

rebase_merge="$(git rev-parse --git-path rebase-merge)"
rebase_apply="$(git rev-parse --git-path rebase-apply)"

if [[ -d "$rebase_merge" || -d "$rebase_apply" ]]; then
  die "Идёт rebase. Завершите его: git rebase --continue или отмените: git rebase --abort."
fi

if [[ -z "$(git config user.name || true)" || -z "$(git config user.email || true)" ]]; then
  die "Не заданы git user.name/user.email. Выполните: git config --global user.name '...'; git config --global user.email '...'"
fi

# Добавляем изменения.
git add -A

if git diff --cached --quiet; then
  log "Нет изменений для нового коммита."
else
  git commit -m "$MESSAGE"
fi

# Если вдруг коммитов ещё нет.
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  die "Нет ни одного коммита. Запустите 01-setup-repo.sh или сделайте коммит вручную."
fi

# Проверяем remote.
if ! git remote get-url origin >/dev/null 2>&1; then
  die "Не настроен origin. Добавьте: git remote add origin <URL>"
fi

# Если ветка уже есть на сервере, сначала обновляемся через rebase.
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  log "Подтягиваю изменения из origin/$BRANCH..."
  if ! git pull --rebase --autostash origin "$BRANCH"; then
    die "Конфликт при rebase. Исправьте файлы, затем: git add . && git rebase --continue; отмена: git rebase --abort. После завершения выполните: git push -u origin $BRANCH"
  fi
fi

# Отправляем.
if git push -u origin "$BRANCH"; then
  log "Готово. Коммиты отправлены в GitHub."
else
  die "Push не удался. Проверьте доступ к GitHub, адрес remote и состояние ветки. Обычно помогает: git pull --rebase origin $BRANCH, затем повторить push."
fi