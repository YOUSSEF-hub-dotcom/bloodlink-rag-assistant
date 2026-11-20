"""Live Phase 4 multi-turn verification CLI.

Example::

    python -m memory.debug_multiturn --conversation-id <uuid> \
        --summary-trigger 2 --history-max 2 \
        "إيه الشروط الأساسية للتبرع بالدم؟" "طب لو أنا ست؟"

It runs several turns in one process (so the embedder loads once), prints each
rewrite/intent/top chunk, and then reads ``rag.conversations.summary`` back from
the database to prove the rolling-summary trigger fired.
"""

from __future__ import annotations

import argparse
import sys

from core.db import bootstrap_schema, close_pool
from core.logging import get_logger
from generation.debug_turn import run_turn
from memory.store import get_conversation, new_conversation_id

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a live multi-turn Phase 4 check.")
    parser.add_argument("questions", nargs="+", help="Questions in conversation order.")
    parser.add_argument("--conversation-id", default=None,
                        help="Reuse an existing UUID (default: generate one).")
    parser.add_argument("--summary-trigger", type=int, default=None,
                        help="Temporary SUMMARY_TRIGGER_MESSAGES override.")
    parser.add_argument("--history-max", type=int, default=None,
                        help="Temporary HISTORY_MAX_MESSAGES override.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip Groq model-list verification.")
    args = parser.parse_args(argv)

    conversation_id = args.conversation_id or new_conversation_id()
    bootstrap_schema()
    print("=" * 100)
    print(f"conversation_id: {conversation_id}")
    print(
        f"temporary overrides: summary_trigger={args.summary_trigger} "
        f"history_max={args.history_max}"
    )
    print("=" * 100)

    try:
        for index, question in enumerate(args.questions, start=1):
            print(f"\n--- TURN {index} ---")
            print(f"original: {question}")
            result = run_turn(
                question,
                conversation_id=conversation_id,
                verify=not args.no_verify,
                summary_trigger=args.summary_trigger,
                history_max=args.history_max,
                on_delta=lambda text: print(text, end="", flush=True),
            )
            print()
            print(f"rewritten standalone_question: {result.rewritten_question}")
            print(f"detected language: {result.language} | intent: {result.intent}")
            print(f"history_messages: {result.history_messages} | chunks: {len(result.results)}")
            if result.results:
                top = result.results[0]
                print(
                    f"top chunk: dense_score={top.score:.4f} "
                    f"source={top.metadata.get('doc_id')} section={top.section_path}"
                )
            print(f"summary_triggered this turn: {result.summary_triggered}")
            print(f"answer: {result.answer}")

        conversation = get_conversation(conversation_id)
        summary = conversation.summary if conversation else None
        print("\n" + "=" * 100)
        print("PERSISTED CONVERSATION STATE")
        print(f"id: {conversation_id}")
        print(f"summary_upto_message_id: {conversation.summary_upto_message_id if conversation else None}")
        print(f"summary: {summary if summary else '(empty)'}")
        print("=" * 100)
    finally:
        close_pool()


if __name__ == "__main__":
    main(sys.argv[1:])
