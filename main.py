# ─── Entry Point ─────────────────────────────────────────────────────────────
from graph import graph, AgentState


def main() -> None:
    query = (
        "What is the weather in Lahore and who is the current Prime Minister "
        "of Pakistan? Now get the age of PM and tell us will this weather "
        "suits PM health."
    )

    initial_state: AgentState = {
        "input": query,
        "agent_scratchpad": [],
        "final_answer": "",
        "steps": [],
    }

    print(f"\nQuery: {query}")
    print("=" * 70)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FINAL ANSWER:\n")
    print(result["final_answer"])

    if result.get("steps"):
        print("\n--- Steps Taken ---")
        for i, step in enumerate(result["steps"], 1):
            print(f"\n[Step {i}]\n{step}")


if __name__ == "__main__":
    main()
