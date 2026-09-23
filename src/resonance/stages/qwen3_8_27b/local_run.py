"""Run a local generation subprocess, stopping its whole group on errors or stalls.

Ported from run_dir('answer_only_27b')/scripts/benchmark_local.py (`run_generation`
only), which benchmark_skip_s2_local loaded via runpy.
"""
import os
import signal
import subprocess
import time


def run_generation(command, env, out):
    """Stop the entire worker group on errors or stalled generation."""
    log = out/'worker.log'
    progress = out/'generation/progress.json'
    with log.open('a+') as stream:
        stream.seek(0, 2)
        offset = stream.tell()
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=stream,
                                start_new_session=True)
        last_progress = time.monotonic()
        previous = progress.stat().st_mtime_ns if progress.exists() else None
        generated = False
        try:
            while proc.poll() is None:
                time.sleep(5)
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
                if 'Traceback (most recent call last)' in chunk:
                    raise RuntimeError(f'Benchmark worker failed; see {log}')
                current = progress.stat().st_mtime_ns if progress.exists() else None
                if current != previous:
                    previous = current
                    generated = True
                    last_progress = time.monotonic()
                if time.monotonic() - last_progress > (300 if generated else 1200):
                    raise TimeoutError(f'Benchmark made no progress; see {log}')
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, command)
        finally:
            # Also remove descendants if the parent exited before its workers.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

