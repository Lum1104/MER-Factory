from flask import Flask, request, jsonify, send_from_directory, Response
import subprocess
import os
import json
import glob
import signal
import sys
import atexit

app = Flask(__name__)
PROMPTS_DIR = os.path.join("utils", "prompts")
current_process = None  # Global variable to track the currently running process


def cleanup_process():
    """Cleanup function to ensure child processes are terminated on program exit"""
    global current_process
    if current_process and current_process.poll() is None:
        print("[DASHBOARD] Terminating child process...")
        try:
            if os.name != "nt":  # Unix systems
                # Terminate the entire process group
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:  # Windows systems
                current_process.terminate()
                current_process.wait(timeout=5)  # Wait for up to 5 seconds
        except subprocess.TimeoutExpired:
            print("[DASHBOARD] Forcefully killing the child process...")
            if os.name != "nt":
                os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
            else:
                current_process.kill()
        except (ProcessLookupError, OSError):
            pass
        except Exception as e:
            print(f"[DASHBOARD] Error while cleaning up the process: {e}")
        finally:
            current_process = None


def signal_handler(signum, frame):
    """Signal handler to handle Ctrl+C"""
    print(f"\n[DASHBOARD] Received signal {signum}, cleaning up...")
    cleanup_process()
    print("[DASHBOARD] Cleanup complete, exiting...")
    sys.exit(0)


# Register signal handler and exit handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_process)


@app.route("/")
def serve_dashboard():
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    return send_from_directory("tools", "tune-dashboard.html")


@app.route("/<path:filename>")
def serve_static_from_tools(filename):
    """
    Serve static files from various locations.
    - First, it checks for absolute paths to allow previewing any file on the system.
    - Then, it checks for files relative to specific project directories ('utils', 'tools').
    - Finally, it falls back to checking for files relative to the current working directory.
    """
    # 1. Handle absolute paths (the requested bug fix).
    # This allows previewing media from any location on the user's machine.
    abs_path_candidate = filename
    if os.name != 'nt' and not filename.startswith('/'):
        abs_path_candidate = '/' + filename
    
    if os.path.isabs(abs_path_candidate) and os.path.isfile(abs_path_candidate):
        try:
            directory, base_filename = os.path.split(abs_path_candidate)
            return send_from_directory(directory, base_filename)
        except Exception as e:
            # Log the error and fall through to return a 404.
            print(f"[DASHBOARD] Error serving absolute path '{abs_path_candidate}': {e}")

    # 2. Handle special 'prompts/' path, which are located in the 'utils' directory.
    if filename.startswith("prompts/"):
        return send_from_directory("utils", filename)

    # 3. Handle paths relative to the 'tools' directory.
    tool_path = os.path.join("tools", filename)
    if os.path.isfile(tool_path):
        return send_from_directory("tools", filename)

    # 4. Handle paths relative to the current working directory as a fallback.
    # This allows serving files from places like the 'output' folder.
    full_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(full_path):
        return send_from_directory(os.getcwd(), filename)

    # If none of the above conditions are met, the file is not found.
    return (
        jsonify(
            {
                "error": "File not found",
                "requested": filename,
            }
        ),
        404,
    )


@app.route("/list-prompts")
def list_prompts():
    try:
        pattern = os.path.join(PROMPTS_DIR, "*.json")
        files = glob.glob(pattern)

        filenames = [os.path.basename(f) for f in files]

        def get_version(filename):
            """Helper to get version number for sorting."""
            if filename == "prompts.json":
                return 0  # Default prompt is version 0
            try:
                # Extract number from prompts_v<number>.json
                return int(filename.replace("prompts_v", "").replace(".json", ""))
            except (ValueError, TypeError):
                return -1  # Other files go to the bottom

        # Sort by version number in descending order
        filenames.sort(key=get_version, reverse=True)
        return jsonify(filenames)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save-prompt", methods=["POST"])
def save_prompt():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No prompt data provided"}), 400

        # Find the next version number
        pattern = os.path.join(PROMPTS_DIR, "prompts_v*.json")
        files = glob.glob(pattern)
        if not files:
            next_version = 1
        else:
            versions = [
                int(os.path.basename(f).replace("prompts_v", "").replace(".json", ""))
                for f in files
            ]
            next_version = max(versions) + 1

        filename = f"prompts_v{next_version}.json"
        filepath = os.path.join(PROMPTS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        print(f"[DASHBOARD] Save failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/run-command", methods=["POST"])
def run_command():
    global current_process
    if current_process and current_process.poll() is None:
        return (
            jsonify({"success": False, "error": "A command is already running."}),
            409,
        )

    data = request.json
    command = data.get("command")
    if not command:
        return jsonify({"success": False, "error": "No command provided"}), 400

    print(f"[DASHBOARD] Running command: {command}")

    def generate_output():
        global current_process
        try:

            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                preexec_fn=(os.setsid if os.name != "nt" else None),
            )
            current_process = process

            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                yield line

            process.stdout.close()
            return_code = process.wait()
            current_process = None

            if return_code != 0:
                print(f"[DASHBOARD] Command ended with error code {return_code}")
                yield f"COMMAND_FAILED_WITH_CODE_{return_code}\n"

        except Exception as e:
            print(f"[DASHBOARD] Runtime exception: {e}")
            yield f"COMMAND_EXCEPTION_{str(e)}\n"
        finally:
            current_process = None

    return Response(generate_output(), mimetype="text/plain")


@app.route("/stop-command", methods=["POST"])
def stop_command():
    global current_process
    if current_process and current_process.poll() is None:
        try:
            if os.name != "nt":
                # Attempt to gracefully terminate the process group
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:
                current_process.terminate()
            current_process.wait(timeout=5)  # Wait for up to 5 seconds
            current_process = None
            print("[DASHBOARD] Command has been terminated")
            return jsonify({"success": True, "message": "Command terminated."})
        except subprocess.TimeoutExpired:
            print(
                "[DASHBOARD] Process did not terminate within 5 seconds, forcefully killing..."
            )
            if os.name != "nt":
                os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
            else:
                current_process.kill()
            current_process = None
            return jsonify({"success": True, "message": "Command forcefully killed."})
        except (ProcessLookupError, OSError):
            current_process = None
            return jsonify({"success": True, "message": "Process already terminated."})
        except Exception as e:
            print(f"[DASHBOARD] Error while terminating the command: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        return jsonify({"success": False, "error": "No command is running."}), 404


@app.route("/export-results", methods=["POST"])
def export_results():
    """Export results by calling export.py script"""
    global current_process
    if current_process and current_process.poll() is None:
        return (
            jsonify({"success": False, "error": "Another command is already running."}),
            409,
        )

    data = request.json
    output_folder = data.get("output_folder", "output_temp")
    file_type = data.get("file_type", "mer")
    export_path = data.get("export_path", "./")

    # Validate file_type
    valid_types = ["au", "image", "mer", "audio", "video"]
    if file_type.lower() not in valid_types:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Invalid file type. Must be one of: {valid_types}",
                }
            ),
            400,
        )

    # Build the export command
    command = f'python export.py --output_folder "{output_folder}" --file_type {file_type.lower()} --export_path "{export_path}" --export_csv'

    print(f"[DASHBOARD] Running export command: {command}")

    def generate_output():
        global current_process
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                preexec_fn=(os.setsid if os.name != "nt" else None),
            )
            current_process = process

            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                yield line

            process.stdout.close()
            return_code = process.wait()
            current_process = None

            if return_code != 0:
                print(f"[DASHBOARD] Export command failed with code {return_code}")
                yield f"EXPORT_FAILED_WITH_CODE_{return_code}\n"
            else:
                print("[DASHBOARD] Export completed successfully")
                yield f"EXPORT_COMPLETED_SUCCESSFULLY\n"

        except Exception as e:
            print(f"[DASHBOARD] Export exception: {e}")
            yield f"EXPORT_EXCEPTION_{str(e)}\n"
        finally:
            current_process = None

    return Response(generate_output(), mimetype="text/plain")


if __name__ == "__main__":
    try:
        app.run(debug=False, port=5000, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Keyboard interrupt received, shutting down...")
        cleanup_process()
    except OSError as e:
        if "Address already in use" in str(e):
            print(
                f"[DASHBOARD] Port 5000 is in use, please free the port or use another one"
            )
            print("[DASHBOARD] You can try running: lsof -ti:5000 | xargs kill -9")
        else:
            print(f"[DASHBOARD] Error starting server: {e}")
        cleanup_process()
    except Exception as e:
        print(f"[DASHBOARD] Runtime error: {e}")
        cleanup_process()
    finally:
        print("[DASHBOARD] Server has been shut down")
