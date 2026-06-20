# Skills catalog — metaSingleCell

Durable project knowledge lives as **skills** in `.claude/skills/<name>/SKILL.md`. Only each
skill's one-line `description` stays in context; the body loads on demand (progressive disclosure).
Record new durable facts in the relevant skill (new topic → new skill dir + a row here).

| Skill | Loads when… |
|-------|-------------|
| `project` | Planning work, understanding the goal (Metal port of rapids-singlecell), the 3-stage roadmap, or repo layout. |
| `environments` | Setting up/using the dedicated conda env, package install, mlx/Metal backend, fp64 vs fp32 facts. |
| `sparse-kernels` | Writing/validating GPU sparse kernels — MLX custom-kernel API, CSR layout, QC kernel, parity harness. |

Planned (not yet created — add when the work begins):
- `scanpy-dropins` — per-function parity notes vs scanpy/rapids-singlecell.
