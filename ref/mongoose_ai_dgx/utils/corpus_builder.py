"""
TRIZ-raw 语料库构建器
将PDF/DOCX/PPTX/DOC原始材料提取、分块、metadata化，输出JSONL语料。
"""

import json
import logging
import os
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Page:
    """单个页面/幻灯片/章节提取结果"""
    text: str
    page_num: int = 1
    heading: Optional[str] = None


@dataclass
class ExtractedDocument:
    """单个文件提取结果"""
    source_path: str
    title: Optional[str] = None
    category: Optional[str] = None
    file_type: Optional[str] = None
    pages: List[Page] = field(default_factory=list)


class Extractor(ABC):
    """文件提取器基类"""

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """从文件中提取文本，返回ExtractedDocument"""
        pass
