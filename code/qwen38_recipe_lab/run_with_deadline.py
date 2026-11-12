"""Launch one existing budgeted CLI only while its round has time remaining."""
import argparse
from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True, help='Timezone-aware ISO timestamp')
    parser.add_argument('--reserve', type=int, default=180)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or args.reserve < 0 or not 1 <= args.max_seconds <= 1200:
        parser.error('Aware deadline, nonnegative reserve and 1..1200 second maximum required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or any(part == '--seconds' or part.startswith('--seconds=') for part in command):
        parser.error('Supply a command without --seconds; the wrapper owns this time budget')
    now = datetime.now().astimezone()
    seconds = min(args.max_seconds, int((deadline-now).total_seconds())-args.reserve)
    if seconds < 1:
        parser.error('Round deadline reached or reserved cleanup time already needed; command not started')
    print(f'Bounded launch: {now.isoformat()} deadline={deadline.isoformat()} seconds={seconds}', flush=True)
    os.execvp(command[0], command+['--seconds', str(seconds)])


if __name__ == '__main__':
    main()
