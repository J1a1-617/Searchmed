"""Build the persistent Skill document embedding index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from searchagent_retrieval.skill_runtime import SkillCatalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-root", type=Path, default=Path("skill_evolution/skills"))
    parser.add_argument("--embed-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("skill_evolution/registry/embedding_cache"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    catalog = SkillCatalog(
        root=args.skills_root,
        embed_model_path=args.embed_model,
        embedding_cache_dir=args.cache_dir,
    )
    result = catalog.ensure_embedding_index(force=args.force)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
