import os
import json
from rich.console import Console
from ..models import LLMModels
from ..prompts import PromptTemplates
from ..tools import Tools

console = Console(stderr=True)

class GateAgent:
    """
    The Gate Agent acts as a quality controller for the multimodal pipeline.
    It evaluates the outputs of individual modality agents (Audio, Video, Image)
    and decides whether to pass them to the final synthesis or trigger a refinement loop.
    """
    
    def __init__(self):
        self.node_map = {
            "audio": "generate_audio_description",
            "video": "generate_video_description",
            "peak_frame": "generate_peak_frame_visual_description",
            "au": None # AU extraction is deterministic and cannot be retried via prompting
        }

    def get_evaluation_prompt(self, current_outputs: str, ground_truth: str = "") -> str:
        current_os = "Windows" if os.name == "nt" else "Linux/Mac"
        
        gt_section = ""
        if ground_truth:
            gt_section = f"\n**Ground Truth Label:**\n{ground_truth}\n(Use this as a strong reference. If analysis contradicts ground truth without strong evidence, reject it.)\n"

        return f"""You are the Gate Agent, an expert Quality Assurance Lead for a high-stakes Multimodal Emotion Reasoning system. Your responsibility is to strictly validate the analysis provided by sub-agents (Audio, Video, Peak Frame, AU) before they are synthesized.

**Objective:**
Ensure every modality provides **evidence-based**, **specific**, and **actionable** insights. Vague or generic descriptions must be rejected.
{gt_section}
**Current Analysis Outputs:**
{current_outputs}

**Available Tools:**
Use these tools to verify data or resolve ambiguity. Do not guess.
1. `run_terminal_command(command)`: Executes shell commands on **{current_os}** (whitelist: ffmpeg, dir/ls, etc.).
2. `run_python_code(code)`: Executes Python code. Use for advanced verification if needed.
3. `analyze_media_metrics(file_path)`: Returns audio intensity (dB) and duration. Use this to resolve conflicts (e.g., if Audio is "angry" but Face is "neutral", check if audio is loud/intense).
4. `analyze_video_motion(file_path)`: Returns a motion analysis (dynamic vs static). Use this to verify if the video is just a talking head (static) or has action.
5. `extract_subtitles(file_path)`: Extracts soft subtitles (if available). **Note:** Not all videos have subtitles. If missing, this tool will return a message stating so. Use this to verify speech content against Audio analysis.

**Evaluation Criteria:**
1.  **Audio**: Should aim to describe relevant emotional cues (e.g., tone, intensity) if present. Detailed acoustics are helpful but not mandatory if the emotion is clear.
2.  **Video**: Should focus on key body language or context that aids understanding. Avoid rejecting based on missing minor details.
3.  **Peak Frame**: Should capture the main visual elements and facial expression. Gaze and subtle details are good to have but not strict requirements.
4.  **Cross-Modality Consistency**: Compare the modalities. If one contradicts the others (e.g., Audio is "happy" but Video/AU are "angry"), reject the outlier unless it has strong evidence.
    *   **Conflict Tip 1**: If Audio is "Angry" but Face/AU is "Neutral", check `analyze_media_metrics`. High volume/intensity suggests the Audio is the primary emotional signal; the face might just be static. **Reject the Neutral AU/Face description** in this case to avoid conflicting signals.
    *   **Conflict Tip 2**: If Video claims "high energy" but looks static, check `analyze_video_motion`. If it says "mostly static", reject the video description.
    *   **Neutral AU Rule**: If the AU analysis is "Neutral", ONLY pass it if the Audio/Video are also low-arousal/neutral. If other modalities are expressive, assume the AU failed to capture the emotion and **reject** the AU result.

**Process:**
1.  **Analyze**: Read the outputs. Are they specific? Do they cite evidence?
2.  **Compare**: Check for conflicts between Audio, Video, and Visual data.
3.  **Verify**: If an output is suspicious, lacks detail, or conflicts with others, use a tool.
4.  **Decide**:
    *   If you need more info -> Output a **Tool Call** in JSON.
    *   If info is sufficient -> Output **Final Decision** in JSON.

**Tool Call Format (JSON):**
To call a tool, output a JSON object inside a markdown block:
```json
{{
  "action": "tool_use",
  "tool": "<tool_name>",
  "arguments": "<argument_string>"
}}
```

**Final Decision Format (JSON):**
To provide the final evaluation, output a JSON object inside a markdown block:
```json
{{
  "action": "final_decision",
  "evaluation": {{
    "audio": {{ "status": "pass" | "fail", "reason": "..." }},
    "video": {{ "status": "pass" | "fail", "reason": "..." }},
    "peak_frame": {{ "status": "pass" | "fail", "reason": "..." }}
  }}
}}
```

**Constraint:**
You are the quality controller. Ensure the analysis is useful and grounded in the data, but do not be pedantic. Focus on major discrepancies or hallucinations. Output ONLY JSON."""

    def get_refinement_prompt(self, modality: str, reason: str, original_prompt: str) -> str:
        return f"""You are the Gate Agent. The {modality} agent's output was rejected.

**Rejection Reason:**
"{reason}"

**Original Instructions:**
"{original_prompt}"

**Your Goal:**
Write a **Refined Instruction** for the {modality} agent to fix this issue.
The new instruction should:
1.  **Directly address the rejection reason.** (e.g., if rejected for being vague, ask for specific examples; if rejected for conflict, ask to double-check for specific cues).
2.  **Encourage precision without hallucination.** (e.g., "Look closely for X, but if it's not there, state that clearly").
3.  **Maintain the core goal** of the original prompt but with better guidance.

**Output:**
Return ONLY the new instruction text. No preamble."""

    async def run(self, state):
        """
        Executes the Gate Agent logic with ReAct loop.
        """
        verbose = state.get("verbose", True)
        if verbose:
            console.rule("[bold purple]Gate Agent Evaluation[/bold purple]")

        model: LLMModels = state["models"].model_instance
        prompts: PromptTemplates = state["prompts"]
        tools = Tools()
        
        # Collect current outputs
        audio_analysis = state.get("audio_analysis_results", "")
        video_description = state.get("video_description", "")
        peak_frame_description = state.get("image_visual_description", "")
        au_description = state.get("peak_frame_au_description", "")
        au_data_path = state.get("au_data_path", "")
        audio_path = state.get("audio_path", "")
        video_path = state.get("video_path", "")
        
        current_outputs = f"""
        Audio Analysis: {audio_analysis}
        Video Description: {video_description}
        Peak Frame Description: {peak_frame_description}
        AU Description: {au_description}
        
        **Data Paths:**
        - Audio File: {audio_path}
        - Video File: {video_path}
        - OpenFace Data: {au_data_path}
        """
        
        # ReAct Loop
        max_turns = 5
        ground_truth = state.get("ground_truth_label", "")
        conversation_history = self.get_evaluation_prompt(current_outputs, ground_truth)
        
        evaluation = {}
        
        for turn in range(max_turns):
            if verbose:
                console.log(f"[cyan]Gate Agent Turn {turn + 1}/{max_turns}[/cyan]")
                
            response = await model.synthesize_summary(conversation_history)
            
            # Parse JSON response
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                parsed_response = json.loads(json_str)
                
                action = parsed_response.get("action")
                
                if action == "tool_use":
                    tool_name = parsed_response.get("tool")
                    tool_args = parsed_response.get("arguments", "")
                    
                    if verbose:
                        console.log(f"[blue]Tool Call: {tool_name} with args: {tool_args}[/blue]")
                    
                    tool_output = ""
                    if tool_name == "run_terminal_command":
                        tool_output = tools.run_terminal_command(tool_args.strip())
                    elif tool_name == "run_python_code":
                        tool_output = tools.run_python_code(tool_args.strip())
                    elif tool_name == "analyze_media_metrics":
                        tool_output = tools.analyze_media_metrics(tool_args.strip())
                    elif tool_name == "analyze_video_motion":
                        tool_output = tools.analyze_video_motion(tool_args.strip())
                    elif tool_name == "extract_subtitles":
                        tool_output = tools.extract_subtitles(tool_args.strip())
                    else:
                        tool_output = f"Error: Unknown tool '{tool_name}'"
                    
                    if verbose:
                        console.log(f"[dim]Tool Output: {tool_output[:100]}...[/dim]")
                    
                    conversation_history += f"\n{response}\nTOOL_OUTPUT: {tool_output}\n"
                    continue # Next turn
                
                elif action == "final_decision":
                    evaluation = parsed_response.get("evaluation", {})
                    break # Exit loop
                
                else:
                    # Fallback for legacy or malformed JSON
                    if "audio" in parsed_response: # Looks like direct evaluation
                         evaluation = parsed_response
                         break
                    
                    if verbose:
                         console.log(f"[red]Unknown action: {action}[/red]")
                    conversation_history += f"\n{response}\nERROR: Unknown action '{action}'. Use 'tool_use' or 'final_decision'.\n"

            except (json.JSONDecodeError, IndexError):
                if verbose:
                    console.log("[red]Failed to parse JSON. Retrying...[/red]")
                conversation_history += f"\n{response}\nERROR: Invalid JSON format. Please provide ONLY the JSON object in a markdown block.\n"
        
        if not evaluation:
            if verbose:
                console.log("[red]Gate Agent failed to produce valid evaluation. Defaulting to PASS.[/red]")
            return {"gate_decision": "pass"}

        gate_feedback = state.get("gate_feedback", {})
        retry_counts = state.get("retry_counts", {"audio": 0, "video": 0, "peak_frame": 0})
        dynamic_prompts = state.get("dynamic_prompts", {})
        
        # State updates to return
        updates = {}
        
        decision = "pass"
        retry_target = None
        
        for modality in ["audio", "video", "peak_frame", "au"]:
            result = evaluation.get(modality, {})
            status = result.get("status", "pass")
            reason = result.get("reason", "")
            
            if status == "pass":
                # Clear dynamic prompt if it exists to prevent redundant re-runs
                if modality in dynamic_prompts:
                    if verbose:
                        console.log(f"[dim]Clearing stale dynamic prompt for {modality} (Passed)[/dim]")
                    del dynamic_prompts[modality]

            elif status == "fail":
                # Handle AU rejection: Do not retry, but remove from state
                if modality == "au":
                    if verbose:
                        console.log(f"[yellow]Gate Agent rejected AU data: {reason}. Removing from synthesis.[/yellow]")
                    updates["peak_frame_au_description"] = "AU data rejected by Gate Agent due to low quality/inconsistency."
                    updates["detected_emotions"] = ["neutral (AU rejected)"]
                    continue

                if retry_counts.get(modality, 0) < 5: # Limit to 5 retries
                    if verbose:
                        console.log(f"[yellow]Gate Agent rejected {modality}: {reason.strip()}[/yellow]")
                    
                    # Generate new prompt
                    original_prompt = ""
                    if modality == "audio":
                         original_prompt = prompts.get_audio_prompt(bool(state.get("ground_truth_label")))
                    elif modality == "video":
                         original_prompt = prompts.get_video_prompt(bool(state.get("ground_truth_label")))
                    elif modality == "peak_frame":
                         original_prompt = prompts.get_image_prompt()

                    refine_prompt = self.get_refinement_prompt(
                        modality=modality,
                        reason=reason,
                        original_prompt=original_prompt
                    )
                    new_prompt = await model.synthesize_summary(refine_prompt)
                    dynamic_prompts[modality] = new_prompt
                    retry_counts[modality] = retry_counts.get(modality, 0) + 1
                    
                    decision = "retry"
                    
                    # Set retry target to the earliest failed node
                    if retry_target is None:
                        retry_target = self.node_map.get(modality)
                else:
                    if verbose:
                         console.log(f"[orange]Max retries reached for {modality}. Proceeding.[/orange]")
        
        if verbose:
            console.log("[bold]Detailed Evaluation:[/bold]")
            for modality, result in evaluation.items():
                status = result.get("status", "unknown")
                reason = result.get("reason", "No reason provided")
                color = "green" if status == "pass" else "red"
                console.log(f"  - {modality.capitalize()}: [{color}]{status.upper()}[/{color}] ({reason})")
            
            console.log(f"[bold]Gate Decision: {decision}, Retry Target: {retry_target}[/bold]")

        updates.update({
            "gate_decision": decision,
            "gate_feedback": gate_feedback,
            "retry_counts": retry_counts,
            "dynamic_prompts": dynamic_prompts,
            "retry_target": retry_target
        })
        return updates
