"""Kill a request mid-pipeline and prove nothing was lost.

Restart survival was demonstrated once by accident, when Docker and the
editor were both closed between a pause and a resume. This does it
deliberately: start a request, kill the process partway through, then
inspect what Postgres holds and continue from there.

The claim being tested is that a checkpoint is written after every node,
so an interrupted request resumes from its last completed step rather
than from the beginning — which for a request costing fifteen cents and
three minutes is the difference between a retry and a restart.

    python scripts/kill_and_resume.py start
    python scripts/kill_and_resume.py inspect
    python scripts/kill_and_resume.py resume
"""

import asyncio
import sys

from rti_engine.agents.checkpointing import checkpointer
from rti_engine.agents.graph import build_graph
from rti_engine.agents.state import current_status, initial_state

THREAD = "kill-and-resume-rehearsal"
REQUEST = (
    "I would like to know the average pay for men and women at my level, "
    "and whether I am paid fairly compared to colleagues doing equivalent work."
)
CONFIG = {"configurable": {"thread_id": THREAD}, "recursion_limit": 40}


async def start() -> None:
    """Run the request. Kill this process partway through with Ctrl-C."""
    print(f"thread: {THREAD}")
    print("kill this process (Ctrl-C) once a node or two has completed\n")

    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        async for event in graph.astream(
            initial_state(THREAD, "EMP-00001", REQUEST, "DE"), config=CONFIG
        ):
            for node in event:
                print(f"  completed: {node}", flush=True)


async def inspect() -> None:
    """Report what survived, from a process that ran none of it."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state(CONFIG)

        if not snapshot.values:
            print("nothing stored under this thread")
            return

        state = snapshot.values
        history = [item async for item in graph.aget_state_history(CONFIG)]

        print(f"next node:   {snapshot.next or '(none — the run completed)'}")
        print(f"status:      {current_status(state).value}")
        print(f"checkpoints: {len(history)}")
        print(
            f"spent:       {state.get('tokens_used', 0)} tokens, ${state.get('cost_usd', 0.0):.4f}"
        )
        print("\ncompleted so far:")
        for entry in state.get("audit", []):
            print(f"  {entry.actor.value:12s} {entry.action}")


async def resume() -> None:
    """Continue from the last checkpoint rather than from the beginning."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)

        before = (await graph.aget_state(CONFIG)).values
        if not before:
            print("nothing to resume")
            return

        print(f"resuming from {(await graph.aget_state(CONFIG)).next}")
        print(f"already spent {before.get('tokens_used', 0)} tokens\n")

        async for event in graph.astream(None, config=CONFIG):
            for node in event:
                print(f"  completed: {node}", flush=True)

        after = (await graph.aget_state(CONFIG)).values
        print(f"\nstatus: {current_status(after).value}")
        print(f"total spent: {after.get('tokens_used', 0)} tokens")


COMMANDS = {"start": start, "inspect": inspect, "resume": resume}


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in COMMANDS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}")
        raise SystemExit(1)

    asyncio.run(COMMANDS[command]())


if __name__ == "__main__":
    main()
