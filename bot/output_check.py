"""最終出力の再検査: 引用番号・参照リンクの不整合や、モデルの暴走を取り除く.

実際に起きた不具合:
- 会話履歴に含まれる過去の回答 (参考リンク付き) を真似て、存在しない発言へのリンクや参考一覧を捏造した
- 「[1][2]…[478]」のように引用番号を延々と出力し続けた
- 同じ発言が別の番号で重複して参考一覧に載った
- 1文に [1]〜[22] のように、その文を裏付けない発言まで大量に引用した
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("sarashina-bot.output")

MARKER = r"\[(?:参)?\d{1,3}\]"
# 引用番号が4つ以上続いたら、最初の3つだけ残す
CITATION_RUN = re.compile(rf"((?:{MARKER}\s*){{3}})(?:{MARKER}\s*)+")
# Discord の発言リンク (モデルが書いたものは捏造の可能性があるので消す)
DISCORD_LINK = re.compile(r"\[([^\]\n]{0,80})\]\(https?://(?:\w+\.)?discord(?:app)?\.com/channels/[^)\s]+\)")
DISCORD_URL = re.compile(r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\S+")
# モデルが真似して書いた参考一覧の行
FAKE_REFERENCE_LINE = re.compile(r"^\s*(-#\s*)?(\[(参)?\d{1,3}\]\s*(<#|#)|参考にした(過去の)?(発言|会話)|-#\s*$)")
# 同じ短い文字列の繰り返し (暴走)
RUNAWAY = re.compile(r"(.{2,40}?)\1{5,}", re.S)

VERIFY_PROMPT = """あなたは引用のチェック係です。回答の各文と、その文に付けられた引用 (番号付きの発言) が与えられます。
各文について、引用された発言のうち、その文の内容を実際に裏付けている発言の番号だけを残してください。
- 文の内容と関係ない発言、文の主張を支えていない発言は外す
- 1文あたり、特に関係の深いものを最大3つまでにする
JSON だけを返す: {"results": [{"s": 文番号, "keep": [残す番号, ...]}, ...]}"""


def strip_references(text: str) -> str:
    """過去の Bot の回答から、参考一覧・引用番号・リンクを取り除く (会話履歴として渡す用)."""
    lines = [ln for ln in text.splitlines() if not FAKE_REFERENCE_LINE.match(ln)]
    text = "\n".join(lines)
    text = DISCORD_LINK.sub("", text)
    text = DISCORD_URL.sub("", text)
    text = re.sub(MARKER, "", text)
    text = re.sub(r"[ \t]+([。、！？」])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def clean_model_output(text: str, keep_markers: bool) -> str:
    """モデルの生出力から、捏造されうる参考一覧・リンクと、暴走した繰り返しを取り除く.

    keep_markers: 本物の参考一覧を後で付ける場合は引用番号を残す (付けない場合は番号も消す)
    """
    original = text
    text = RUNAWAY.sub(r"\1", text)
    lines = [ln for ln in text.splitlines() if not FAKE_REFERENCE_LINE.match(ln)]
    text = "\n".join(lines)
    text = DISCORD_LINK.sub(r"\1", text)
    text = DISCORD_URL.sub("", text)
    text = CITATION_RUN.sub(r"\1", text)
    if not keep_markers:
        text = re.sub(MARKER, "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([。、！？」])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text != original.strip():
        log.info("出力を整えました (参考一覧・リンク・繰り返しの除去)")
    return text


def split_sentences(text: str) -> list[str]:
    """文 (句点・改行区切り) に分ける. 区切り文字は前の文に含める."""
    parts = re.split(r"(?<=[。！？\n])", text)
    return [p for p in parts if p]


async def verify_citations(sarashina, model: str, answer: str, sources: list[str], max_sentences: int = 30) -> str:
    """各文の引用が本当にその文を裏付けているかを nano で確かめ、裏付けていない番号を外す.

    sources: 引用番号 n に対応する発言のテキスト (sources[n-1])
    """
    sentences = split_sentences(answer)
    targets = []
    for i, sent in enumerate(sentences):
        nums = [int(n) for n in re.findall(r"\[(\d{1,3})\]", sent) if 1 <= int(n) <= len(sources)]
        if nums:
            targets.append((i, list(dict.fromkeys(nums))))
    if not targets:
        return answer
    blocks = []
    for i, nums in targets[:max_sentences]:
        cited = "\n".join(f"  [{n}] {sources[n - 1][:160]}" for n in nums[:10])
        blocks.append(f"(文{i}) {re.sub(MARKER, '', sentences[i]).strip()}\n{cited}")
    try:
        raw = await sarashina.chat(
            [{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": "\n\n".join(blocks)}],
            model=model, max_completion_tokens=800, temperature=0.0,
            response_format={"type": "json_object"},
        )
        results = {r["s"]: set(r.get("keep") or []) for r in json.loads(raw).get("results") or []
                   if isinstance(r, dict) and isinstance(r.get("s"), int)}
    except Exception:
        log.warning("引用の検査に失敗しました (番号の整理だけ行います)", exc_info=True)
        return answer
    removed = 0
    for i, nums in targets[:max_sentences]:
        if i not in results:
            continue
        keep = {n for n in results[i] if isinstance(n, int)}

        def drop(match: re.Match) -> str:
            nonlocal removed
            n = int(match.group(1))
            if n in keep:
                return match.group(0)
            removed += 1
            return ""

        sentences[i] = re.sub(r"\[(\d{1,3})\]", drop, sentences[i])
    if removed:
        log.info("裏付けの無い引用を %d 個外しました", removed)
    return "".join(sentences)
