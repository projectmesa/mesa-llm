# Release History

All notable changes to this project will be documented in this file. This project follows [Semantic Versioning](https://semver.org/).

# 0.3.0 (2026-03-13)

## What's Changed
### ⚠️ Breaking changes
* fix: allow None for prompt in get_messages, generate, agenerate by @gamal1osama in https://github.com/mesa/mesa-llm/pull/93
### 🛠 Enhancements made
* fix: store unique_ids instead of Agent objects in send_message memory… by @Harsh-617 in https://github.com/mesa/mesa-llm/pull/157
### 🐛 Bugs fixed
* Fix async memory consolidation by @psbuilds in https://github.com/mesa/mesa-llm/pull/69
* fix: async generation endpoint parity in ModuleLLM by @zhaosina in https://github.com/mesa/mesa-llm/pull/105
* Fix critical vision radius bugs in LLMAgent perception system by @hillhack in https://github.com/mesa/mesa-llm/pull/61
* fix: suppress spurious Auto-save failed warning from atexit in tests by @gamal1osama in https://github.com/mesa/mesa-llm/pull/84
* fix: extract text content from LLM response in memory consolidation by @khansalman12 in https://github.com/mesa/mesa-llm/pull/101
* Add ShortTermMemory Tests  + BUG FIXES by @psbuilds in https://github.com/mesa/mesa-llm/pull/76
* fix: align STLTMemory.get_prompt_ready() return type with Memory ABC by @khushiiagrawal in https://github.com/mesa/mesa-llm/pull/118
* fix: CoT and ReWOO reasoning pass `str` to `add_to_memory()` where `dict` is expected by @khushiiagrawal in https://github.com/mesa/mesa-llm/pull/123
* fix: remove duplicate space arg in ContinuousSpace teleport logic(#119) by @BhoomiAgrawal12 in https://github.com/mesa/mesa-llm/pull/126
* fix: add missing obs and ttl params to ReWOOReasoning.plan() and apla… by @BhoomiAgrawal12 in https://github.com/mesa/mesa-llm/pull/131
* fix: use agenerate_obs in ReWOOReasoning.aplan() instead of blocking generate_obs by @khansalman12 in https://github.com/mesa/mesa-llm/pull/133
* Respect ignore_agent flag in docstring validation by @Harsh-617 in https://github.com/mesa/mesa-llm/pull/130
* fix: prevent move_one_step from crashing with OrthogonalMooreGrid by @khushiiagrawal in https://github.com/mesa/mesa-llm/pull/138
* Fix/Add Boundary Check in move_one_step to Prevent Cryptic Grid Errors by @gamal1osama in https://github.com/mesa/mesa-llm/pull/146
* fix: store graded EpisodicMemory entries as MemoryEntry objects and use correct llm instance  by @psbuilds in https://github.com/mesa/mesa-llm/pull/109
* fix: handle non-LLM neighbors in _build_observation by @BhoomiAgrawal12 in https://github.com/mesa/mesa-llm/pull/145
### 📜 Documentation improvements
* Fix broken tutorial link. by @divilian in https://github.com/mesa/mesa-llm/pull/80
* docs: add orphaned pages to toctree in index.md by @yashhzd in https://github.com/mesa/mesa-llm/pull/81
* reorganize readthedocs folder structure by @wang-boyu in https://github.com/mesa/mesa-llm/pull/92
* update issue/pr templates and link to mesa contributor guide by @wang-boyu in https://github.com/mesa/mesa-llm/pull/124
### 🔧 Maintenance
* Remove duplicate entries in __all__ in mesa_llm.__init__ by @gamal1osama in https://github.com/mesa/mesa-llm/pull/94
* fix: replace deprecated AgentSet.__getitem__ in tests by @yashhzd in https://github.com/mesa/mesa-llm/pull/82
* remove redundant linter from GA and  update ruff hook in pre-commit by @wang-boyu in https://github.com/mesa/mesa-llm/pull/103
* fix(parallel_stepping): remove leftover debug print from _agentset_do… by @BhoomiAgrawal12 in https://github.com/mesa/mesa-llm/pull/100
* refactor: replace rich console with standard logging and improve api … by @uday-codes69 in https://github.com/mesa/mesa-llm/pull/85
* Validate llm_model format in ModuleLLM by @Harsh-617 in https://github.com/mesa/mesa-llm/pull/89
* fix: replace deprecated AgentSet indexing for Mesa 4.0 compatibility by @Zain-Naqi in https://github.com/mesa/mesa-llm/pull/110
* test: add memory × reasoning integration tests + fix async signature bugs by @yashhzd in https://github.com/mesa/mesa-llm/pull/128
* test: consolidate and reuse test fixtures in conftest files by @wang-boyu in https://github.com/mesa/mesa-llm/pull/121

## New Contributors
* @psbuilds made their first contribution in https://github.com/mesa/mesa-llm/pull/69
* @gamal1osama made their first contribution in https://github.com/mesa/mesa-llm/pull/94
* @divilian made their first contribution in https://github.com/mesa/mesa-llm/pull/80
* @yashhzd made their first contribution in https://github.com/mesa/mesa-llm/pull/81
* @zhaosina made their first contribution in https://github.com/mesa/mesa-llm/pull/105
* @hillhack made their first contribution in https://github.com/mesa/mesa-llm/pull/61
* @BhoomiAgrawal12 made their first contribution in https://github.com/mesa/mesa-llm/pull/100
* @uday-codes69 made their first contribution in https://github.com/mesa/mesa-llm/pull/85
* @Harsh-617 made their first contribution in https://github.com/mesa/mesa-llm/pull/89
* @khansalman12 made their first contribution in https://github.com/mesa/mesa-llm/pull/101
* @Zain-Naqi made their first contribution in https://github.com/mesa/mesa-llm/pull/110
* @khushiiagrawal made their first contribution in https://github.com/mesa/mesa-llm/pull/118

**Full Changelog**: https://github.com/mesa/mesa-llm/compare/v0.2.0...v0.3.0

# 0.2.0 (2026-02-16)

## What's Changed
### 🛠 Enhancements made
* Memory parent class improvements by @colinfrisch in https://github.com/mesa/mesa-llm/pull/30
### 🐛 Bugs fixed
* Fix Ollama tool calling crash when system prompt is None by @februarysea in https://github.com/mesa/mesa-llm/pull/41
* fix: treat ollama_chat as ollama provider by @februarysea in https://github.com/mesa/mesa-llm/pull/43
* Fix prompt gestion by @colinfrisch in https://github.com/mesa/mesa-llm/pull/48
* add error handling to move one step tool by @apoorvasj in https://github.com/mesa/mesa-llm/pull/62
### 🔍 Examples updated
* Sugarscape g1mt by @sujalgawas in https://github.com/mesa/mesa-llm/pull/35
* feat: Add dialogue history context to negotiation agents by @Nithin9585 in https://github.com/mesa/mesa-llm/pull/56
### 📜 Documentation improvements
* update links for repository transfer by @wang-boyu in https://github.com/mesa/mesa-llm/pull/19
* docs: Add Google Gemini to supported LLMs in README by @sujalgawas in https://github.com/mesa/mesa-llm/pull/24
* Update organization name from `projectmesa` to `mesa` by @EwoutH in https://github.com/mesa/mesa-llm/pull/39
* Update contact email. by @jackiekazil in https://github.com/mesa/mesa-llm/pull/40
* docs: Add examples page and expand API documentation structure by @colinfrisch in https://github.com/mesa/mesa-llm/pull/33
* Add mesa-llm overview by @niteshver in https://github.com/mesa/mesa-llm/pull/60
* Add first mesa-llm model tutorial  by @niteshver in https://github.com/mesa/mesa-llm/pull/45
* add getting_started for mesa_llm by @niteshver in https://github.com/mesa/mesa-llm/pull/64
* Implemented API documentation using Sphinx by @pushpitkamboj in https://github.com/mesa/mesa-llm/pull/46
* docs: add sugarscrap model in example folder  by @niteshver in https://github.com/mesa/mesa-llm/pull/74
### 🔧 Maintenance
* test(reasoning): Add tests for Reasoning base class by @DipayanDasgupta in https://github.com/mesa/mesa-llm/pull/21
* Test episodic memory by @sujalgawas in https://github.com/mesa/mesa-llm/pull/26
* Test lt memory by @sujalgawas in https://github.com/mesa/mesa-llm/pull/27
* test: update module_llm tests for system prompt handling (None -> "") by @februarysea in https://github.com/mesa/mesa-llm/pull/44
* configure automatically generated release notes by @wang-boyu in https://github.com/mesa/mesa-llm/pull/72
### Other changes
* Fix and add dependencies by @colinfrisch in https://github.com/mesa/mesa-llm/pull/49

## New Contributors
* @DipayanDasgupta made their first contribution in https://github.com/mesa/mesa-llm/pull/21
* @sujalgawas made their first contribution in https://github.com/mesa/mesa-llm/pull/24
* @EwoutH made their first contribution in https://github.com/mesa/mesa-llm/pull/39
* @jackiekazil made their first contribution in https://github.com/mesa/mesa-llm/pull/40
* @februarysea made their first contribution in https://github.com/mesa/mesa-llm/pull/41
* @Nithin9585 made their first contribution in https://github.com/mesa/mesa-llm/pull/56
* @niteshver made their first contribution in https://github.com/mesa/mesa-llm/pull/60
* @apoorvasj made their first contribution in https://github.com/mesa/mesa-llm/pull/62
* @pushpitkamboj made their first contribution in https://github.com/mesa/mesa-llm/pull/46

**Full Changelog**: https://github.com/mesa/mesa-llm/compare/v0.1.1...v0.2.0

# 0.1.1 (2025-09-12)

## What's Changed
- First public release.
- Core repository structure and documentation.

**Full Changelog**: https://github.com/mesa/mesa-llm/compare/v0.0.2...v0.1.1
