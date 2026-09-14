#!/usr/bin/env python3
"""24维原子特征冻结表: mendeleev SQLite直读 -> CSV + meta sidecar.

跑法(训练env不动): 用pip --target装mendeleev的脏env/系统python直读其elements.db.
输出: utils/atom_features_mendeleev.csv (Z=1..103, 全精度) + .meta.json (出处/缺失/填充).
缺失: 列内中位数填充(计数写入meta); He等天然缺失同规则, 不可考但可复现.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

DB = Path("/tmp/mend_pkgs/mendeleev/elements.db")
OUT_CSV = Path("/root/home/newstudy/uniARPAT/utils/atom_features_mendeleev.csv")
OUT_META = Path("/root/home/newstudy/uniARPAT/utils/atom_features_mendeleev.meta.json")

# (列名, 取数方式, 出处备注)
COLS = [
    ("atomic_weight", "elements.atomic_weight", "IUPAC标准原子量, mendeleev打包"),
    ("covalent_radius_cordero", "elements.covalent_radius_cordero", "Cordero et al. 2008"),
    ("atomic_radius", "elements.atomic_radius", "Waber-Cromer计算值, mendeleev打包"),
    ("vdw_radius", "elements.vdw_radius", "Bondi 1964体系, mendeleev打包"),
    ("metallic_radius", "elements.metallic_radius", "12配位金属半径, mendeleev打包"),
    ("en_pauling", "elements.en_pauling", "Pauling标度(稀有气体天然缺失)"),
    ("en_allen", "elements.en_allen", "Allen光谱电负性1992"),
    ("ionization_1", "ionizationenergies[charge=1].ionization_energy", "NIST ASD电离能"),
    ("electron_affinity", "elements.electron_affinity", "Andersen/Hotopp体系(部分元素无稳定负离子)"),
    ("melting_point", "phasetransitions.melting_point", "Zhang/CRC相变数据"),
    ("boiling_point", "phasetransitions.boiling_point", "Zhang/CRC相变数据"),
    ("density", "elements.density", "单质密度, mendeleev打包"),
    ("molar_heat_capacity", "elements.molar_heat_capacity", "定压摩尔热容, mendeleev打包"),
    ("thermal_conductivity", "elements.thermal_conductivity", "热导率, mendeleev打包"),
    ("abundance_crust", "elements.abundance_crust", "地壳丰度, CRC"),
    ("group_id", "elements.group_id", "族号1-18(含f区标识, 见mendeleev文档)"),
    ("period", "elements.period", "周期号"),
    ("mendeleev_number", "elements.mendeleev_number", "Mendeleev Pettifor式排序数"),
    ("pettifor_number", "elements.pettifor_number", "Pettifor化学标度1984"),
    ("dipole_polarizability", "elements.dipole_polarizability", "Schwerdtfeger静态极化率(a.u.)"),
    ("c6", "elements.c6", "色散C6系数(Gould/Bucko体系, a.u.)"),
    ("fusion_heat", "elements.fusion_heat", "熔化热, mendeleev打包"),
    ("evaporation_heat", "elements.evaporation_heat", "蒸发热, mendeleev打包"),
    ("en_miedema", "elements.en_miedema", "Miedema电负性(合金热模型配套)"),
]


def main():
    import numpy as np
    db = sqlite3.connect(str(DB))
    db.row_factory = sqlite3.Row
    ion = {}
    for r in db.execute("select atomic_number, ionization_energy from ionizationenergies where ion_charge=1"):
        ion.setdefault(r["atomic_number"], r["ionization_energy"])
    melt, boil = {}, {}
    for r in db.execute("select atomic_number, melting_point, boiling_point from phasetransitions"):
        melt.setdefault(r["atomic_number"], r["melting_point"])
        boil.setdefault(r["atomic_number"], r["boiling_point"])
    els = {r["atomic_number"]: dict(r) for r in db.execute("select * from elements")}
    getters = {
        "ionization_1": lambda z: ion.get(z),
        "melting_point": lambda z: melt.get(z),
        "boiling_point": lambda z: boil.get(z),
    }
    raw = {}
    for name, _, _ in COLS:
        col = []
        for z in range(1, 104):
            if name in getters:
                v = getters[name](z)
            else:
                field = name
                v = els.get(z, {}).get(field)
            try:
                col.append(float(v) if v is not None else None)
            except (TypeError, ValueError):
                col.append(None)
        raw[name] = col
    # 中位数填充 + 记录
    meta_cols = {}
    mat = np.full((103, len(COLS)), np.nan)
    for j, (name, src, ref) in enumerate(COLS):
        c = np.array([np.nan if v is None else v for v in raw[name]])
        miss = int(np.isnan(c).sum())
        med = float(np.nanmedian(c))
        c[np.isnan(c)] = med
        mat[:, j] = c
        meta_cols[name] = {"source": src, "reference": ref,
                           "missing_n": miss, "impute": f"median={med:.6g}"}
    # CSV全精度
    with open(OUT_CSV, "w") as f:
        f.write("Z," + ",".join(n for n, _, _ in COLS) + "\n")
        for i in range(103):
            f.write(str(i + 1) + "," + ",".join(repr(float(v)) for v in mat[i]) + "\n")
    h = hashlib.sha256(open(OUT_CSV, "rb").read()).hexdigest()[:16]
    meta = {"generator": "mendeleev 1.3.0 (pip --target脏env, 训练env未动)",
            "db": "mendeleev/elements.db (83列/118行快照)",
            "elements": "Z=1..103", "n_features": len(COLS),
            "csv_sha16": h, "columns": meta_cols}
    json.dump(meta, open(OUT_META, "w"), indent=1, ensure_ascii=False)
    print(f"[feat] wrote {OUT_CSV} sha={h}", flush=True)
    tot_miss = sum(v["missing_n"] for v in meta_cols.values())
    print(f"[feat] total imputed cells={tot_miss}/{103 * len(COLS)}", flush=True)
    for name in ("en_pauling", "electron_affinity", "metallic_radius", "melting_point"):
        print(f"  {name}: missing={meta_cols[name]['missing_n']}", flush=True)


if __name__ == "__main__":
    main()
