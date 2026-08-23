import base64
import io
import json
import os

import requests
from fastapi import Request, UploadFile

from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.users import UserModel
from open_webui.routers.files import upload_file_handler

BRIDGE = "http://192.168.5.246:8090"
MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


async def _read_image_by_id(fid):
    """根据文件 id 从数据库查 path 并读取图片，返回 (ext, b64) 或 None。"""
    try:
        fmodel = await Files.get_file_by_id(fid)
        if not fmodel or not fmodel.path or not os.path.exists(fmodel.path):
            return None
        ctype = ((fmodel.meta or {}).get("content_type")) or ""
        if not ctype.startswith("image/"):
            return None
        with open(fmodel.path, "rb") as fp:
            data = fp.read()
        if not data:
            return None
        return ctype.split("/")[-1], base64.b64encode(data).decode("ascii")
    except Exception:
        return None


async def _collect_all_images(files, chat_id):
    """收集对话里的所有图片（用户上传 + 文生图 + 对话历史），返回 (markdown, 图片数)。"""
    ids = []

    def _add(fid):
        if fid and isinstance(fid, str) and fid not in ids:
            ids.append(fid)

    # 1. 用户上传的文件 (__files__)
    for f in files or []:
        if isinstance(f, dict):
            _add(f.get("id"))
            fobj = f.get("file")
            if isinstance(fobj, dict):
                _add(fobj.get("id"))

    # 2. 对话历史里的文件 (含文生图生成的图片)
    try:
        chat = await Chats.get_chat_by_id(chat_id)
        if chat and chat.chat:
            for m in (chat.chat.get("history", {}).get("messages", {})).values():
                for f in (m.get("files") or []):
                    if isinstance(f, dict):
                        _add(f.get("id"))
                        fobj = f.get("file")
                        if isinstance(fobj, dict):
                            _add(fobj.get("id"))
                        url = f.get("url") or ""
                        if "/files/" in url:
                            _add(url.rstrip("/").split("/")[-2])
                        elif url and "/" not in url and len(url) >= 20:
                            _add(url)  # 纯 UUID 形式的文件 id
    except Exception:
        pass

    # 3. 读取图片内容
    imgs = []
    for fid in ids:
        r = await _read_image_by_id(fid)
        if r:
            ext, b64 = r
            imgs.append(f"![图片](data:image/{ext};base64,{b64})")
    return "\n\n".join(imgs), len(imgs)


async def _get_previous_assistant_content(chat_id):
    """从对话历史获取最近一条 assistant 消息的文本内容（模型之前生成的分析/报告）。"""
    try:
        chat = await Chats.get_chat_by_id(chat_id)
        if not chat or not chat.chat:
            return ""
        messages = (chat.chat or {}).get("history", {}).get("messages", {})
        # 按消息顺序取最后一条有内容的 assistant 消息
        items = list(messages.values())
        for m in reversed(items):
            if m.get("role") == "assistant":
                content = m.get("content") or ""
                if content and len(content) > 200:
                    return content
    except Exception:
        pass
    return ""


class Tools:
    async def convert_markdown_to_file(
        self,
        format: str,
        markdown_text: str = "",
        __files__: list = None,
        __request__: Request = None,
        __user__: dict = None,
        __event_emitter__=None,
        __chat_id__: str = None,
        __message_id__: str = None,
    ) -> str:
        """
        将 Markdown 文本转换成 Word / PDF / Excel / PPT 文件，并自动嵌入当前对话里的图片（图文混排）。
        文件会自动附加到当前对话，用户直接点击下载即可。
        调用本工具时，markdown_text 参数可以留空——工具会自动从对话历史里取你刚才生成的分析/报告内容，并自动嵌入对话里的图片。
        不要自行假设或编造任何文件路径。

        :param format: 目标文件格式，只能是 docx / pdf / xlsx / pptx 之一
        :param markdown_text: 要转换的 Markdown 文本内容（可留空，留空时自动用对话历史里你刚生成的内容）
        """
        fmt = (format or "docx").lower().lstrip(".")
        if fmt not in MIME:
            return json.dumps(
                {"error": f"不支持的格式 {format}，支持: docx / pdf / xlsx / pptx"},
                ensure_ascii=False,
            )

        # 如果 markdown_text 太短，从对话历史取模型之前生成的分析/报告内容
        md_text = markdown_text or ""
        if len(md_text.strip()) < 100:
            prev = await _get_previous_assistant_content(__chat_id__)
            if prev and len(prev) > len(md_text):
                md_text = prev

        # 把对话里的图片拼进 Markdown (图文混排): 用户上传 + 文生图 + 对话历史
        img_md, img_count = await _collect_all_images(__files__, __chat_id__)
        full_md = (img_md + "\n\n" + md_text) if img_md else md_text

        try:
            resp = requests.post(
                f"{BRIDGE}/convert", json={"fmt": fmt, "md_text": full_md}, timeout=180
            )
            resp.raise_for_status()
            data = resp.content
        except Exception as e:
            return json.dumps({"error": f"文档转换失败: {e}"}, ensure_ascii=False)

        user = UserModel(**__user__) if __user__ else None
        filename = f"output.{fmt}"
        file = UploadFile(
            file=io.BytesIO(data), filename=filename, headers={"content-type": MIME[fmt]}
        )
        file_item = await upload_file_handler(
            request=__request__, file=file, metadata={}, process=False, user=user
        )
        url = __request__.app.url_path_for("get_file_content_by_id", id=file_item.id)

        file_entry = {
            "type": "file",
            "id": file_item.id,
            "url": url,
            "name": filename,
            "meta": {"content_type": MIME[fmt], "size": len(data)},
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
            {
                "status": "success",
                "filename": filename,
                "size": len(data),
                "embedded_images": img_count,
                "message": (
                    f"文件已生成并自动附加到对话中，用户可直接点击下载。"
                    f"已成功嵌入 {img_count} 张图片到文档中。"
                    "请直接告知用户文件已生成、图片已嵌入即可，"
                    "绝对不要在回复中说图片无法嵌入或提及任何文件路径（例如 sandbox:/mnt/data/ 之类）。"
                ),
            },
            ensure_ascii=False,
        )
