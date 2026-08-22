import io
import json

import requests
from fastapi import Request, UploadFile

from open_webui.models.chats import Chats
from open_webui.models.users import UserModel
from open_webui.routers.files import upload_file_handler

BRIDGE = "http://host.docker.internal:8090"


class Tools:
    async def generate_image(
        self,
        prompt: str,
        annotations: list = None,
        __request__: Request = None,
        __user__: dict = None,
        __event_emitter__=None,
        __chat_id__: str = None,
        __message_id__: str = None,
    ) -> str:
        """
        根据文字描述生成一张图片（本地 FLUX.1-dev 文生图模型），并把图片显示在对话中。

        生成结构图/原理图/示意图时，用 annotations 参数在图上叠加中文标注（零件名、步骤、发明原理编号等）。
        不要要求模型在图片里直接渲染中文文字——文生图模型渲染中文会乱码，正确做法是生成无文字的图，再用 annotations 叠加清晰中文标注。

        :param prompt: 图片描述，中文或英文皆可（结构图/工程图建议英文描述，构图更准确）
        :param annotations: 可选，中文标注列表。每项 {"text": "文字", "x": 像素x, "y": 像素y, "size": 字号, "color": "颜色"}
            坐标范围对应图片尺寸（默认 768x768），size 默认 36，color 默认 red。示例：
            [{"text": "泵体", "x": 60, "y": 80, "size": 42, "color": "red"},
             {"text": "叶轮", "x": 340, "y": 300, "size": 42, "color": "blue"}]
        """
        payload = {"prompt": prompt}
        if annotations:
            payload["annotations"] = annotations
        try:
            resp = requests.post(f"{BRIDGE}/image", json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.content
        except Exception as e:
            return json.dumps({"error": f"图片生成失败: {e}"}, ensure_ascii=False)

        user = UserModel(**__user__) if __user__ else None
        file = UploadFile(
            file=io.BytesIO(data), filename="generated.png", headers={"content-type": "image/png"}
        )
        file_item = await upload_file_handler(
            request=__request__, file=file, metadata={}, process=False, user=user
        )
        url = __request__.app.url_path_for("get_file_content_by_id", id=file_item.id)

        file_entry = {"type": "image", "id": file_item.id, "url": url}
        if __chat_id__ and __message_id__:
            await Chats.add_message_files_by_id_and_message_id(
                __chat_id__, __message_id__, [file_entry]
            )
        if __event_emitter__:
            await __event_emitter__(
                {"type": "chat:message:files", "data": {"files": [file_entry]}}
            )

        return json.dumps(
            {"status": "success", "message": "图片已生成并显示在对话中"},
            ensure_ascii=False,
        )
