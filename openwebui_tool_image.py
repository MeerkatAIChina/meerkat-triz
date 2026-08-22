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
        title: str = "",
        legend: list = None,
        __request__: Request = None,
        __user__: dict = None,
        __event_emitter__=None,
        __chat_id__: str = None,
        __message_id__: str = None,
    ) -> str:
        """
        根据文字描述生成一张图片（本地 FLUX.1-dev 文生图模型），并把图片显示在对话中。

        生成结构图/原理图/示意图时，用 title 和 legend 添加中文标注：
        - 不要要求模型在图片里直接渲染中文文字（会乱码），中文标注由 title/legend 后期合成。
        - title 显示在图片顶部居中；legend 显示在底部白色图例条里，自动编号 (1)(2)(3)...

        :param prompt: 图片描述，中文或英文皆可（结构图/工程图建议英文描述，构图更准确）
        :param title: 可选，图片顶部居中的标题（中文），如 "齿轮传动系统结构示意图"
        :param legend: 可选，底部图例文字列表（中文），如 ["主动轮", "从动轮", "传动轴"]
        """
        payload = {"prompt": prompt}
        if title:
            payload["title"] = title
        if legend:
            payload["legend"] = legend
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

        file_entry = {
            "type": "file",
            "id": file_item.id,
            "url": url,
            "name": "generated.png",
            "meta": {"content_type": "image/png", "size": len(data)},
        }
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
