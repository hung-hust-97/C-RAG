import re

from lightrag.utils import logger


class Node:
    def to_html(self):
        raise NotImplementedError


class HeaderNode(Node):
    def __init__(self):
        self.co_quan = []
        self.so_hieu = ""
        self.quoc_hieu = [
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "Độc lập - Tự do - Hạnh phúc",
        ]
        self.ngay_thang = ""
        self.trich_yeu = ""

    def to_html(self):
        left_content = "<br>".join(self.co_quan)
        if self.so_hieu:
            left_content += f"<br>-------<br>{self.so_hieu}"
        if self.trich_yeu:
            left_content += f"<br><i>{self.trich_yeu}</i>"

        right_content = (
            f"<b>{self.quoc_hieu[0]}<br>{self.quoc_hieu[1]}<br>------------------</b>"
        )
        if self.ngay_thang:
            right_content += f"<br><br><i>{self.ngay_thang}</i>"

        return f"""
        <table border="0" cellspacing="0" cellpadding="0" width="100%" style="margin-bottom: 20px;">
            <tr>
                <td width="40%" valign="top" align="center" style="font-size: 13pt;"><b>{left_content}</b></td>
                <td width="60%" valign="top" align="center" style="font-size: 13pt;">{right_content}</td>
            </tr>
        </table>
        """


class TitleNode(Node):
    def __init__(self, title):
        self.title = title

    def to_html(self):
        return f'<div style="text-align: center; font-weight: bold; font-size: 14pt; text-transform: uppercase; margin: 20px 0;">{self.title}</div>'


class RecipientNode(Node):
    def __init__(self):
        self.lines = []

    def to_html(self):
        content = "<br>".join(self.lines)
        return f"""
        <table border="0" cellspacing="0" cellpadding="0" width="100%" style="margin-bottom: 15px;">
            <tr>
                <td width="15%" valign="top" align="right"><b>Kính gửi: &nbsp;</b></td>
                <td width="85%" valign="top">{content}</td>
            </tr>
        </table>
        """


class BodyBlockNode(Node):
    def __init__(self, text):
        self.text = text

    def to_html(self):
        # Handle list items vs normal paragraphs
        if self.text.startswith("-") or re.match(
            r"^([a-đz])\)", self.text, re.IGNORECASE
        ):
            return f'<div style="text-align: justify; margin-left: 1.27cm; margin-bottom: 6pt;">{self.text}</div>'
        return f'<div style="text-align: justify; text-indent: 1.27cm; margin-bottom: 6pt;">{self.text}</div>'


class SignatureNode(Node):
    def __init__(self):
        self.noi_nhan = []
        self.chuc_danh = ""
        self.nguoi_ky = ""

    def to_html(self):
        noi_nhan_html = "<br>".join(self.noi_nhan)
        return f"""
        <table border="0" cellspacing="0" cellpadding="0" width="100%" style="margin-top: 30px;">
            <tr>
                <td width="50%" valign="top">
                    <b><i>Nơi nhận:</i></b><br>
                    <span style="font-size: 11pt;">{noi_nhan_html}</span>
                </td>
                <td width="50%" valign="top" align="center">
                    <b>{self.chuc_danh}</b><br><br><br><br><br>
                    <b>{self.nguoi_ky}</b>
                </td>
            </tr>
        </table>
        """


class DocumentBuilder:
    def __init__(self):
        self.nodes = []
        self.state = "HEADER"
        self.current_header = HeaderNode()
        self.current_recipient = None
        self.current_signature = None

    def parse(self, md_text):
        lines = [line.strip() for line in md_text.split("\n") if line.strip()]

        for line in lines:
            lower_line = line.lower()

            # State Transitions
            if (
                "cộng hòa" in lower_line
                or "độc lập" in lower_line
                or line.startswith("---")
            ):
                continue

            if lower_line.startswith("kính gửi:"):
                self._commit_header()
                self.state = "RECIPIENT"
                self.current_recipient = RecipientNode()
                continue

            if lower_line.startswith("nơi nhận:"):
                self.state = "SIGNATURE"
                self.current_signature = SignatureNode()
                continue

            # State Handlers
            if self.state == "HEADER":
                if (
                    "ngày" in lower_line
                    and "tháng" in lower_line
                    and "năm" in lower_line
                ):
                    self.current_header.ngay_thang = line
                elif lower_line.startswith("số:"):
                    self.current_header.so_hieu = line
                elif lower_line.startswith("v/v"):
                    self.current_header.trich_yeu = line
                elif lower_line == "quyết định" or lower_line.startswith("chỉ thị"):
                    self._commit_header()
                    self.nodes.append(TitleNode(line))
                    self.state = "BODY"
                else:
                    if (
                        len(self.current_header.co_quan) < 2
                        and not self.current_header.so_hieu
                    ):
                        self.current_header.co_quan.append(line)
                    elif self.current_header.so_hieu or self.current_header.trich_yeu:
                        self._commit_header()
                        self.state = "BODY"
                        self.nodes.append(BodyBlockNode(line))

            elif self.state == "RECIPIENT":
                if (
                    line[0].isupper()
                    and not line.startswith("-")
                    and not lower_line.startswith("đồng kính gửi")
                ):
                    # Kính gửi block ends, body starts
                    self.nodes.append(self.current_recipient)
                    self.state = "BODY"
                    self.nodes.append(BodyBlockNode(line))
                else:
                    self.current_recipient.lines.append(line)

            elif self.state == "BODY":
                self.nodes.append(BodyBlockNode(line))

            elif self.state == "SIGNATURE":
                if line.isupper() and len(line) > 5 and not line.startswith("-"):
                    self.current_signature.chuc_danh = line
                elif self.current_signature.chuc_danh and line:
                    self.current_signature.nguoi_ky = line
                else:
                    self.current_signature.noi_nhan.append(line)

        # Final commit
        if self.current_header not in self.nodes and self.state == "HEADER":
            self._commit_header()
        if self.current_signature:
            self.nodes.append(self.current_signature)

    def _commit_header(self):
        if self.current_header and self.current_header not in self.nodes:
            self.nodes.append(self.current_header)

    def render(self):
        # Assemble all nodes dynamically
        inner_html = "".join([node.to_html() for node in self.nodes])
        return f"""<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Legal Document</title>
</head>
<body>
    <div style="font-family: 'Times New Roman', Times, serif; font-size: 14pt; line-height: 1.5; padding: 40px; background: white;">
        {inner_html}
    </div>
</body>
</html>
"""


def convert_document(md_text):
    builder = DocumentBuilder()
    builder.parse(md_text)
    html_output = builder.render()

    logger.info("Converted legal markdown to HTML nodes=%s", len(builder.nodes))

    return html_output
