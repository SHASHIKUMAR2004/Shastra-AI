import argparse
import sys
import traceback

from agent.graph import agent


def main():
    parser = argparse.ArgumentParser(description="Run AI engineering project generator")
    parser.add_argument(
        "--recursion-limit",
        "-r",
        type=int,
        default=150,
        help="Recursion limit for LangGraph execution",
    )

    args = parser.parse_args()

    try:
        user_prompt = input("Enter your project prompt: ").strip()

        if not user_prompt:
            print("No prompt provided.")
            sys.exit(1)

        result = agent.invoke(
            {"user_prompt": user_prompt},
            {"recursion_limit": args.recursion_limit},
        )

        print("\n==============================")
        print("Generation completed")
        print("==============================")

        project_dir = result.get("project_dir")
        if project_dir:
            print(f"Project directory: {project_dir}")

        test_result = result.get("test_result")
        if test_result:
            print(f"Test summary: {test_result.summary}")

            if test_result.issues:
                print("\nIssues:")
                for issue in test_result.issues:
                    print(f"- {issue}")

        print("\nFinal status:", result.get("status"))

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()