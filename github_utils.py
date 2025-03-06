from github import Github, GithubException
import os
import base64
import logging
import time
from typing import Optional

# Настройка логирования
logger = logging.getLogger(__name__)

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "son37ofc"
AI_PROGRESS_FILENAME = "cfr_data.pkl"


def save_ai_progress_to_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    """
    Оптимизированное сохранение прогресса ИИ в GitHub.
    
    Args:
        filename (str): Имя файла для сохранения. По умолчанию AI_PROGRESS_FILENAME.
    
    Returns:
        bool: True если сохранение успешно, False иначе.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN не установлен. Сохранение отключено.")
        return False

    # Проверка существования и размера файла
    if not os.path.exists(filename):
        logger.error(f"Локальный файл {filename} не существует.")
        return False
        
    file_stats = os.stat(filename)
    file_size = file_stats.st_size
    file_mtime = file_stats.st_mtime
    
    if file_size == 0:
        logger.error(f"Файл {filename} пуст. Не сохраняем на GitHub.")
        return False
        
    logger.info(f"Сохранение прогресса ИИ на GitHub. Размер файла: {file_size} байт, " 
                f"Последнее изменение: {time.ctime(file_mtime)}")

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            # Читаем локальный файл
            with open(filename, "rb") as f:
                local_content = f.read()
                
            # Получаем текущую версию с GitHub
            contents = repo.get_contents(filename, ref="main")
            
            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Всегда обновляем файл, без проверки идентичности
            repo.update_file(
                contents.path,
                f"Обновление прогресса ИИ ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                local_content,
                contents.sha,
                branch="main",
            )
            logger.info(f"✅ Прогресс ИИ успешно сохранен на GitHub: {GITHUB_REPOSITORY}/{filename}")
            return True
            
        except GithubException as e:
            if e.status == 404:
                # Создаем новый файл если не существует
                with open(filename, "rb") as f:
                    local_content = f.read()
                    
                repo.create_file(
                    filename,
                    f"Начальный прогресс ИИ ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                    local_content,
                    branch="main",
                )
                logger.info(f"✅ Создан новый файл прогресса на GitHub: {GITHUB_REPOSITORY}/{filename}")
                return True
            else:
                logger.error(f"Ошибка GitHub API: {e.status}, {e.data}")
                return False
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при сохранении: {e}")
        return False


def load_ai_progress_from_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    """
    Оптимизированная загрузка прогресса ИИ из GitHub.
    
    Args:
        filename (str): Имя файла для загрузки. По умолчанию AI_PROGRESS_FILENAME.
    
    Returns:
        bool: True если загрузка успешна, False иначе.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN не установлен. Загрузка отключена.")
        return False

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        
        try:
            logger.info(f"Попытка загрузки прогресса ИИ с GitHub: {GITHUB_REPOSITORY}/{filename}")
            contents = repo.get_contents(filename, ref="main")
            file_content = base64.b64decode(contents.content)
            
            # Проверяем, не пустой ли файл
            if len(file_content) == 0:
                logger.warning("GitHub файл пуст. Отмена загрузки.")
                return False
            
            # Сохраняем скачанный файл
            with open(filename, "wb") as f:
                f.write(file_content)
            
            logger.info(f"✅ Прогресс ИИ успешно загружен с GitHub: {GITHUB_REPOSITORY}/{filename}, " 
                        f"Размер: {len(file_content)} байт")
            return True
            
        except GithubException as e:
            if e.status == 404:
                logger.warning(f"Файл {filename} не найден в репозитории GitHub.")
                return False
            else:
                logger.error(f"Ошибка GitHub API при загрузке: status={e.status}, data={e.data}")
                return False
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при загрузке с GitHub: {e}")
        return False
