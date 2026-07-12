"""
流水线状态管理工具
跨Notebook的JSON工件注册表，用于追踪数据、模型、结果等工件的状态
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from packaging import version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineState:
    """JSON工件注册表，支持跨Notebook状态追踪"""

    DEFAULT_STATE_FILE = "/home/meerkat/mongoose_ai/data/processed/pipeline_state.json"

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = Path(state_file or self.DEFAULT_STATE_FILE)
        self.state = self._load()
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载状态文件失败 ({e})，创建新状态")
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "artifacts": [],
        }

    def _save(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def register(
        self,
        name: str,
        path: str,
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """注册或更新一个工件"""
        artifact = {
            "name": name,
            "path": str(path),
            "type": artifact_type,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        # Remove existing entry with same name
        self.state["artifacts"] = [
            a for a in self.state["artifacts"] if a["name"] != name
        ]
        self.state["artifacts"].append(artifact)
        self._save()
        logger.info(f"注册工件: {name} ({artifact_type}) -> {path}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """按名称获取工件信息"""
        for a in self.state["artifacts"]:
            if a["name"] == name:
                return a
        return None

    def verify(self, name: str) -> bool:
        """验证工件是否存在且路径有效"""
        artifact = self.get(name)
        if not artifact:
            return False
        return Path(artifact["path"]).exists()

    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有工件，可按类型过滤"""
        if artifact_type:
            return [a for a in self.state["artifacts"] if a["type"] == artifact_type]
        return list(self.state["artifacts"])

    def preflight(
        self,
        required_artifacts: Optional[List[str]] = None,
        required_packages: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        预飞行检查：验证所需工件和包版本

        Args:
            required_artifacts: 必需的工件名称列表
            required_packages: 必需的包及其最低版本，如 {"transformers": "4.45.0"}

        Returns:
            错误信息列表，空列表表示检查通过
        """
        errors = []

        # 检查工件
        if required_artifacts:
            for name in required_artifacts:
                if not self.verify(name):
                    artifact = self.get(name)
                    if artifact:
                        errors.append(f"工件路径不存在: {name} -> {artifact['path']}")
                    else:
                        errors.append(f"未注册工件: {name}")

        # 检查包版本
        if required_packages:
            for pkg_name, min_ver in required_packages.items():
                try:
                    mod = __import__(pkg_name)
                    actual_ver = getattr(mod, "__version__", "unknown")
                    if actual_ver != "unknown":
                        if version.parse(actual_ver) < version.parse(min_ver):
                            errors.append(
                                f"{pkg_name} 版本过低: {actual_ver} < {min_ver}"
                            )
                    else:
                        errors.append(f"{pkg_name} 无法获取版本信息")
                except ImportError:
                    errors.append(f"未安装包: {pkg_name}")

        if errors:
            logger.error("预飞行检查失败:")
            for e in errors:
                logger.error(f"  - {e}")
        else:
            logger.info("预飞行检查通过")

        return errors

    def summary(self) -> Dict[str, Any]:
        """返回状态摘要"""
        artifacts = self.state["artifacts"]
        type_counts = {}
        for a in artifacts:
            t = a["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "state_file": str(self.state_file),
            "total_artifacts": len(artifacts),
            "type_counts": type_counts,
            "artifact_names": [a["name"] for a in artifacts],
        }
