# 工作日志 2026-09-11：JARVIS覆盖映射（A7b）收官

## 改了什么
- 新增 `tools/getdata/a7b_jarvis_coverage.py`（JVASP→legacy→canonical，免费复用A3c 5,510对，
  余量API singles，断点续跑；`--singles-only --workers 8`）；
- 落盘：`jarvis_a7b_map.jsonl`（45,154行）、`jarvis_a7b_summary.json`、
  `jarvis_a7b_nickname.json`（37个别名ID）。

## 结果如何
- 49,942对 → 45,154 JVASP已映射（90.4%），43,872唯一canonical，无碰撞；
  死leg 4,686（`mp-76/77xxxx`等新区段为主，删库）+ 37昵称（`mp-blackP`等）挂unresolved/L2。
- 交集（ID级，注意≠谱确认）：有效全集内43,868；MP声子交15,815；真双谱交13,142；
  **缺口天花板28,053**（有效全集内、MP无声子、JVASP有映射）→ 相对PhononDB增量4,391的6.4倍**候选池**；
  另2,673（有声子无eDOS）得JARVIS第二视角候选；4在集外。
- 教训：
  1. 整批零命中（死ID 10%分散，每批必毒死）——这次直接singles，5/s@8workers；
  2. `200+0条`=definitive dead不重试，省一半死ID耗时；失败原因计数（empty/http/exc） jinak 发现限流；
  3. **映射≠谱**：A3c的4,391是谱已落地的真增量；JARVIS的28,053只是ID天花板，
     JVASP侧有无谱要逐个调`jarvis.db.webpages`才知道（pilot 77的路径已验证，`jarvis`包可用）。

## 下一步（谱清单，按缺口抓，不全抓）
1. 缺口JVASP（→28,053 canonical）的JARVIS phDOS拉取 → 真新增双谱（A4输入）；
2. 交集JVASP（→13,142真双谱）抽样拉取 → A2b三源gap（与PhononDB 4,660交集优先，三源齐全子集）；
3. 死leg 4,686 + 昵称37 + 集外4 → L2结构匹配（A7 L2执行时一并）。
