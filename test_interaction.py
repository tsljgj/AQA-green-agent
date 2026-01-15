"""Test green agent + purple agent interaction locally."""
import asyncio
import json
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
from uuid import uuid4

# Ordered difficulty levels for proper display
DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]


def generate_accuracy_graph(by_difficulty: dict, bar_width: int = 30) -> str:
    """Generate an ASCII bar chart showing accuracy by difficulty level.

    Args:
        by_difficulty: Dictionary mapping difficulty -> stats (with pass_rate)
        bar_width: Width of the bar chart in characters

    Returns:
        ASCII art string representation of the graph
    """
    lines = []
    lines.append("")
    lines.append("  Accuracy by Difficulty Level")
    lines.append("  " + "=" * (bar_width + 20))
    lines.append("")

    # Sort difficulties in proper order
    ordered_diffs = [d for d in DIFFICULTY_ORDER if d in by_difficulty]

    if not ordered_diffs:
        return "  No difficulty data available"

    for diff in ordered_diffs:
        stats = by_difficulty[diff]
        pass_rate = stats.get("pass_rate", 0.0)
        correct = stats.get("correct", 0)
        total = stats.get("total", 0)

        # Create the bar
        filled_width = int(pass_rate * bar_width)
        empty_width = bar_width - filled_width

        # Use different characters for visual appeal
        bar = "\u2588" * filled_width + "\u2591" * empty_width

        # Format the label (pad to 8 chars)
        label = f"{diff.capitalize():8s}"

        # Format percentage and fraction
        pct_str = f"{pass_rate:6.1%}"
        frac_str = f"({correct}/{total})"

        lines.append(f"  {label} |{bar}| {pct_str} {frac_str}")

    lines.append("")
    lines.append("  " + "=" * (bar_width + 20))
    lines.append(f"  Legend: \u2588 = correct, \u2591 = incorrect")
    lines.append("")

    return "\n".join(lines)


async def run_assessment(
    purple_agent_url: str = "http://localhost:9010",
    green_agent_url: str = "http://localhost:9009",
    num_tasks: int = 3,
    difficulties: list[str] = None,
    seed: int = 42
):
    """Run a test assessment.

    Args:
        purple_agent_url: URL of the purple agent to test
        green_agent_url: URL of the green agent (AQA benchmark)
        num_tasks: Number of questions to ask
        difficulties: List of difficulty levels (easy, medium, hard)
        seed: Random seed for reproducible question sampling
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard", "expert"]

    # Assessment request
    request = {
        "participants": {"agent": purple_agent_url},
        "config": {
            "num_tasks": num_tasks,
            "difficulty": difficulties,
            "seed": seed
        }
    }

    # Connect to green agent
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resolver = A2ACardResolver(httpx_client=client, base_url=green_agent_url)
            card = await resolver.get_agent_card()
            print(f"✅ Connected to green agent: {card.name}")
        except Exception as e:
            print(f"❌ Failed to connect to green agent at {green_agent_url}")
            print(f"   Error: {e}")
            print(f"   Make sure the green agent is running!")
            return

        try:
            # Check purple agent is reachable
            purple_resolver = A2ACardResolver(httpx_client=client, base_url=purple_agent_url)
            purple_card = await purple_resolver.get_agent_card()
            print(f"✅ Purple agent reachable: {purple_card.name}")
        except Exception as e:
            print(f"❌ Failed to connect to purple agent at {purple_agent_url}")
            print(f"   Error: {e}")
            print(f"   Make sure the purple agent is running!")
            return

        config = ClientConfig(httpx_client=client, streaming=True)
        factory = ClientFactory(config)
        a2a_client = factory.create(card)

        # Send request
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(request)))],
            message_id=uuid4().hex,
        )

        print("\n" + "="*70)
        print("🚀 Starting AQA Assessment")
        print("="*70)
        print(f"   Purple Agent: {purple_agent_url}")
        print(f"   Questions: {num_tasks}")
        print(f"   Difficulties: {', '.join(difficulties)}")
        print(f"   Seed: {seed}")
        print("="*70 + "\n")

        try:
            async for event in a2a_client.send_message(msg):
                # Print status updates
                if hasattr(event, 'status') and hasattr(event.status, 'message'):
                    if event.status.message:
                        from a2a.utils import get_message_text
                        status_text = get_message_text(event.status.message)
                        print(f"📊 {status_text}")

                # Print final results
                if hasattr(event, 'artifacts') and event.artifacts:
                    print("\n")

                    for artifact in event.artifacts:
                        for part in artifact.parts:
                            if hasattr(part.root, 'text'):
                                # Print the formatted results text from the agent
                                print(part.root.text)
                            elif hasattr(part.root, 'data'):
                                result = part.root.data
                                agg = result.get('aggregate', {})
                                items = result.get('items', [])
                                by_diff = agg.get('by_difficulty', {})

                                # Print accuracy graph
                                if by_diff:
                                    print(generate_accuracy_graph(by_diff))

                                # Print weighted score and level accuracies
                                weighted_score = agg.get('weighted_score', 0.0)
                                print(f"\n  Weighted Score: {weighted_score:.3f}")
                                print(f"  (Weights: Easy=1x, Medium=2x, Hard=3x, Expert=4x)")

                                # Print individual level accuracies for leaderboard
                                print("\n  Leaderboard Metrics:")
                                print("  " + "-" * 50)
                                print(f"  easy_accuracy:   {agg.get('easy_accuracy', 0.0):6.1%}")
                                print(f"  medium_accuracy: {agg.get('medium_accuracy', 0.0):6.1%}")
                                print(f"  hard_accuracy:   {agg.get('hard_accuracy', 0.0):6.1%}")
                                print(f"  expert_accuracy: {agg.get('expert_accuracy', 0.0):6.1%}")
                                print(f"  weighted_score:  {weighted_score:.3f}")

                                # Print detailed by-difficulty breakdown
                                if by_diff:
                                    print("\n  Detailed Breakdown by Difficulty:")
                                    print("  " + "-" * 50)
                                    for diff in DIFFICULTY_ORDER:
                                        if diff not in by_diff:
                                            continue
                                        stats = by_diff[diff]
                                        total = stats.get('total', 0)
                                        correct = stats.get('correct', 0)
                                        if total > 0:
                                            print(f"  {diff.capitalize():8s}: {correct:2d}/{total:<2d} "
                                                  f"({stats['pass_rate']:6.1%}) | avg_score: {stats['avg_score']:.3f}")
                                        else:
                                            print(f"  {diff.capitalize():8s}: --/--  (no questions)")
                                    print()

                                if items:
                                    print("\n  Individual Question Results:")
                                    print("  " + "-" * 50)
                                    for i, item in enumerate(items, 1):
                                        status = "PASS" if item['correct'] else "FAIL"
                                        status_icon = "✓" if item['correct'] else "✗"
                                        print(f"\n  [{status_icon}] Q{i}: {item['question'][:70]}...")
                                        print(f"      Difficulty: {item['difficulty'].capitalize()}")
                                        print(f"      Expected:   {item['reference_answer'][:80]}")
                                        print(f"      Got:        {item['agent_answer'][:80]}")
                                        print(f"      Score: {item['score']:.2f} | Latency: {item['latency_ms']}ms | {status}")
                                        if item['evaluation_reasoning']:
                                            # Truncate reasoning if too long
                                            reasoning = item['evaluation_reasoning'][:120]
                                            if len(item['evaluation_reasoning']) > 120:
                                                reasoning += "..."
                                            print(f"      Reason: {reasoning}")

                    print("\n" + "=" * 50)

        except Exception as e:
            print(f"\n❌ Assessment failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AQA green agent with a purple agent")
    parser.add_argument("--purple-url", default="http://localhost:9010", help="Purple agent URL")
    parser.add_argument("--green-url", default="http://localhost:9009", help="Green agent URL")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of questions")
    parser.add_argument("--difficulty", nargs="+", default=["easy", "medium", "hard", "expert"], help="Difficulty levels (easy, medium, hard, expert)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    asyncio.run(run_assessment(
        purple_agent_url=args.purple_url,
        green_agent_url=args.green_url,
        num_tasks=args.num_tasks,
        difficulties=args.difficulty,
        seed=args.seed
    ))
