import subprocess
import pandas as pd
from pathlib import Path
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class Tools:
    """
    A collection of tools for the Gate Agent to use.
    """

    def __init__(self):
        self.whitelist_commands = ["ffmpeg", "ls", "dir", "cat", "echo", "grep", "head", "tail"]

    def analyze_media_metrics(self, file_path: str) -> str:
        """
        Analyzes media file for audio intensity, duration, and basic video stats.
        Useful for resolving conflicts (e.g., loud audio vs neutral face).
        """
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        
        metrics = {}
        
        try:
            # 1. Get Duration and Audio Volume
            # ffmpeg -i input -filter:a volumedetect -f null /dev/null
            cmd = f'ffmpeg -i "{path}" -filter:a volumedetect -f null -'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = result.stderr # ffmpeg writes stats to stderr
            
            # Parse volume
            import re
            mean_vol = re.search(r"mean_volume: ([\-\d\.]+) dB", output)
            max_vol = re.search(r"max_volume: ([\-\d\.]+) dB", output)
            
            if mean_vol: metrics["mean_volume_db"] = float(mean_vol.group(1))
            if max_vol: metrics["max_volume_db"] = float(max_vol.group(1))
            
            # 2. Get Duration (using ffprobe if available, or parsing ffmpeg output)
            duration = re.search(r"Duration: (\d{2}:\d{2}:\d{2}\.\d{2})", output)
            if duration: metrics["duration"] = duration.group(1)

            return f"Media Metrics: {metrics}"
            
        except Exception as e:
            return f"Error analyzing media: {str(e)}"

    def analyze_video_motion(self, file_path: str) -> str:
        """
        Analyzes video for motion and scene changes.
        Returns a 'Motion Score' (0-10) and notes on static segments.
        """
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        
        try:
            # Detect scene changes (scdet) and frozen frames (freezedetect)
            # scdet threshold 0.1 (10%), freezedetect noise -60dB, duration 2s
            cmd = f'ffmpeg -i "{path}" -vf "select=\'gt(scene,0.1)\',showinfo" -f null -'
            # This is a bit complex to parse reliably in one go without huge output.
            # Let's use a simpler proxy: 'frozenframes' to see if it's mostly static.
            
            cmd_freeze = f'ffmpeg -i "{path}" -vf "freezedetect=n=0.003:d=1.0" -f null -'
            result = subprocess.run(cmd_freeze, shell=True, capture_output=True, text=True)
            output = result.stderr
            
            import re
            freeze_matches = re.findall(r"freezedetect: freeze_start: ([\d\.]+) freeze_duration: ([\d\.]+)", output)
            
            total_freeze_duration = sum(float(m[1]) for m in freeze_matches)
            
            # Get total duration to calculate percentage
            duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", output)
            total_duration = 0
            if duration_match:
                h, m, s = map(float, duration_match.groups())
                total_duration = h*3600 + m*60 + s
            
            report = []
            if total_duration > 0:
                freeze_ratio = total_freeze_duration / total_duration
                if freeze_ratio > 0.8:
                    report.append("Video is mostly static (talking head or slide).")
                elif freeze_ratio > 0.3:
                    report.append("Video has significant static segments.")
                else:
                    report.append("Video is dynamic.")
            else:
                report.append("Could not determine duration.")

            return f"Video Motion Analysis: {'; '.join(report)} (Frozen Duration: {total_freeze_duration:.2f}s)"
            
        except Exception as e:
            return f"Error analyzing video motion: {str(e)}"

    def run_terminal_command(self, command: str) -> str:
        """
        Executes a shell command if it is in the whitelist.
        Securely parses arguments and disables shell execution to prevent injection.
        """
        import shlex
        
        try:
            # Use posix=False for better Windows path handling (backslashes)
            cmd_parts = shlex.split(command, posix=False)
        except Exception as e:
            return f"Error parsing command: {str(e)}"

        if not cmd_parts:
            return "Error: Empty command."
        
        base_cmd = cmd_parts[0]
        if base_cmd not in self.whitelist_commands:
            return f"Error: Command '{base_cmd}' is not in the whitelist. Allowed: {', '.join(self.whitelist_commands)}"
        
        try:
            # Run command safely with shell=False
            result = subprocess.run(
                cmd_parts, 
                shell=False, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return output
        except FileNotFoundError:
            return f"Error: Command '{base_cmd}' not found. Note: shell=False is enforced, so shell builtins (like 'dir') or aliases may not work. Use absolute paths or 'list_dir' tool."
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def run_python_code(self, code: str) -> str:
        """
        Executes arbitrary Python code and returns the output.
        """
        # Create a buffer to capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Create a restricted global environment if needed, but for now we trust the agent 
                # (as per user request for "gate agent write the code is fine")
                # We pass 'pd' so they can use pandas easily
                exec_globals = {"pd": pd, "print": print}
                exec(code, exec_globals)
            
            stdout_val = stdout_buffer.getvalue()
            stderr_val = stderr_buffer.getvalue()
            
            output = ""
            if stdout_val:
                output += f"OUTPUT:\n{stdout_val}\n"
            if stderr_val:
                output += f"ERRORS:\n{stderr_val}\n"
            
            if not output:
                output = "Code executed successfully (no output)."
                
            return output
            
        except Exception as e:
            return f"Python Execution Error: {str(e)}"
