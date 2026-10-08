# TendonSpin resource guard entry gate. Allows verifying OS limits before Isaac starts.
import argparse,os,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--ready',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
start=time.monotonic()
while not args.ready.exists():
    if time.monotonic()-start>20:raise SystemExit('watchdog did not release the startup gate')
    time.sleep(.1)
os.execvpe(command[0],command,os.environ)
