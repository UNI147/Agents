import os
import re
from pathlib import Path

# Графика для дерева
PIPE = "│   "
ELBOW = "├── "
TEE = "└── "
SPACER = "    "

def extract_classes(filepath):
    """Извлекает имена классов из файла .py (верхнего уровня)."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    pattern = r'^\s*class\s+(\w+)\s*[:\(]'
    matches = re.findall(pattern, content, re.MULTILINE)
    return matches

def is_package(dirpath):
    """Проверяет, является ли каталог Python-пакетом."""
    return os.path.isfile(os.path.join(dirpath, '__init__.py'))

def build_tree(root, prefix="", exclude_dirs=None, is_root=True):
    """
    Генерирует строки дерева.
    exclude_dirs – множество имён папок, которые НЕ нужно раскрывать рекурсивно.
    is_root – флаг, что это корневой уровень (раскрываем все файлы/папки поверхностно).
    """
    if exclude_dirs is None:
        exclude_dirs = set()

    try:
        entries = os.listdir(root)
    except PermissionError:
        return [f"{prefix}{TEE}[доступ запрещён]"]

    dirs = []
    files = []
    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            if entry.startswith('.') or entry == '__pycache__':
                continue
            dirs.append(entry)
        else:
            files.append(entry)

    dirs.sort()
    files.sort()

    lines = []
    total = len(dirs) + len(files)
    idx = 0

    for d in dirs:
        idx += 1
        connector = ELBOW if idx < total else TEE
        comment = " # пакет" if is_package(os.path.join(root, d)) else ""
        lines.append(f"{prefix}{connector}{d}\\{comment}")
        
        # Раскрываем папку, только если она НЕ в чёрном списке
        if d not in exclude_dirs:
            sub_prefix = prefix + (PIPE if idx < total else SPACER)
            lines.extend(build_tree(os.path.join(root, d), sub_prefix, exclude_dirs, is_root=False))

    for f in files:
        idx += 1
        connector = ELBOW if idx < total else TEE
        file_path = os.path.join(root, f)
        comment = ""
        if f.endswith('.py'):
            classes = extract_classes(file_path)
            if len(classes) == 1:
                comment = f" # класс {classes[0]}"
            elif len(classes) > 1:
                comment = f" # классы: {', '.join(classes)}"
        lines.append(f"{prefix}{connector}{f}{comment}")

    return lines

if __name__ == "__main__":
    import sys
    start_path = sys.argv[1] if len(sys.argv) > 1 else "."
    start_path = os.path.abspath(start_path)

    # ЧЁРНЫЙ СПИСОК – папки, которые НЕ раскрываем
    EXCLUDE_DIRS = {
        "agentsenv",    # виртуальное окружение
        "htmlcov",       # отчёты coverage
        "tests",         # тесты (раскомментируйте, если не нужны)
        ".git",          # git
        ".pytest_cache", # кэш pytest
        "__pycache__",   # кэш Python (и так фильтруется, но для надёжности)
    }
    
    # Можно дополнить через командную строку: python scan_tree.py . папка1,папка2
    if len(sys.argv) > 2:
        extra = set(sys.argv[2].split(","))
        EXCLUDE_DIRS.update(extra)

    # Печатаем корень
    root_name = os.path.basename(start_path) or start_path
    print(f"{root_name}\\")
    tree_lines = build_tree(start_path, exclude_dirs=EXCLUDE_DIRS)
    for line in tree_lines:
        print(line)
