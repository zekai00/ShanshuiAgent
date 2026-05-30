#!/usr/bin/env python3
"""Normalize priority supplement filenames to put topic before institution."""

from __future__ import annotations

from pathlib import Path

from curate_authority_corpus import (
    build_registry,
    load_sources,
    write_manifest_and_source_list,
    write_sources,
)


RENAMES = {
    "dpm_qingchu_siseng_yimin_2020": "H01_清初四僧_遗民僧人的翰墨情怀.pdf",
    "dpm_hongren_yishu_fengge_2020": "H02_清初四僧_弘仁绘画艺术风格.pdf",
    "dpm_jiang_songs_he_qingquan_qiuri_bianwei_2020": "H03_清初四僧_渐江松壑清泉图秋日山居图辨伪.pdf",
    "dpm_kuncan_huihua_yishu_2020": "H04_清初四僧_髡残及其绘画艺术.pdf",
    "dpm_kuncan_qinggan_yishu_2020": "H05_清初四僧_髡残个人情感与其艺术.pdf",
    "dpm_bada_shuhua_jiexi_2020": "H06_清初四僧_八大山人书画作品解析.pdf",
    "dpm_shitao_shuhua_rensheng_jue_2020": "H07_清初四僧_石涛书画人生中的绝.pdf",
    "dpm_bada_shitao_yishu_jingjie_2020": "H08_清初四僧_八大石涛的艺术境界.pdf",
    "dpm_bada_shitao_qinggan_xinlu_2020": "H09_清初四僧_八大山人与石涛经典绘画导读.pdf",
    "dpm_shitao_jieziyuan_2020": "H10_清初四僧_石涛和芥子园画传.pdf",
    "dpm_qianlong_nansong_sijia_pinshang_2018": "H11_南宋四家_乾隆内府对绘画品赏与收藏.pdf",
    "dpm_songhua_conference_zongshu_2018": "H12_宋画_国际学术会议综述.pdf",
    "dpm_malin_daotong_shengxian_2024": "H13_南宋马麟_道统圣贤图制作脉络.pdf",
    "dpm_shitao_luofu_2024": "H14_清代石涛_罗浮图册与罗浮野乘.pdf",
    "dpm_wangximeng_xingshi_zaocu_2020": "H15_北宋王希孟_希孟姓氏和早卒案蠡测.pdf",
    "dpm_shitao_shiliu_luohan_2020": "H16_清代石涛_十六罗汉图卷核心与寄托.pdf",
}


def main() -> int:
    rows = load_sources()
    changed = 0
    for doc_id, filename in RENAMES.items():
        row = rows.get(doc_id)
        if not row:
            print(f"missing {doc_id}")
            continue

        old_path = Path(row["local_path"])
        new_path = old_path.with_name(filename)
        if old_path != new_path:
            if old_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
            elif not new_path.exists():
                raise FileNotFoundError(f"neither old nor new path exists for {doc_id}: {old_path} / {new_path}")
            row["local_path"] = str(new_path)
            changed += 1
        row["filename"] = filename

    write_sources(rows)
    manifest = write_manifest_and_source_list(rows)
    stats = build_registry(rows)
    print({"renamed": changed, "complete_documents": manifest["complete_documents"], "registry_documents": stats["complete_documents"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
