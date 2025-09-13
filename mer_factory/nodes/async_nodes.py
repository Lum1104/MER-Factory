import json
from rich.console import Console
from pathlib import Path
import asyncio


from mer_factory.prompts import PromptTemplates
from tools.ffmpeg_adapter import FFMpegAdapter
from ..models import LLMModels
from tools.emotion_analyzer import EmotionAnalyzer
from tools.facial_analyzer import FacialAnalyzer

console = Console(stderr=True)


async def setup_paths(state):
    video_path = Path(state["video_path"])
    video_id = state.get("video_id", video_path.stem)
    video_output_dir = state.get("video_output_dir")

    if state.get("verbose", True):
        console.log(f"Processing: {video_id}")
        console.log(f"Output directory set to: [cyan]{video_output_dir}[/cyan]")

    return {}


async def run_au_extraction(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Action Unit (AU) Extraction[/bold]")

    au_data_path = Path(state["au_data_path"])
    if not au_data_path.exists():
        return {
            "error": f"AU data file not found at {au_data_path}. The background OpenFace task may have failed."
        }
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")
    return {"au_data_path": au_data_path}


async def save_au_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ AU Analysis Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_au_analysis.json"
    )
    result_data = {
        "source_path": str(Path(state["video_path"]).resolve(strict=False)),
        "chronological_emotion_peaks": state.get("detected_emotions", []),
        "overall_peak_au_description": state.get("peak_frame_au_description"),
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"AU analysis results saved to [green]{output_path}[/green]")
    return {}


async def generate_audio_description(state):
    # reuse existing audio analysis results if available
    if state.get("processing_type") == "MER" and state.get("cache"):
        output_path = (
            Path(state["video_output_dir"]) / f"{state['video_id']}_audio_analysis.json"
        )
        if output_path.exists():
            verbose = state.get("verbose", True)
            if verbose:
                console.log(
                    f"Cache hit for audio analysis. Loading from [green]{output_path}[/green]"
                )
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"audio_analysis_results": data.get("audio_analysis", "")}

    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Audio Analysis[/bold]")

    audio_path = Path(state["audio_path"])
    if not audio_path.exists():
        return {
            "error": f"Audio file not found at {audio_path}. The background FFMpeg task may have failed."
        }

    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]
    if verbose:
        console.log(f"Analyzing pre-extracted audio at [green]{audio_path}[/green]")

    processing_type = state.get("processing_type")
    ground_truth_label = state.get("ground_truth_label")

    # if processing_type is audio we pass the label to the prompt
    # otherwise (MER), we do not, because emotion cannot be inferred from audio alone.
    has_label = bool(ground_truth_label) if processing_type == "audio" else False
    prompt = prompts.get_audio_prompt(has_label)
    if has_label:
        prompt = prompt.format(label=ground_truth_label)

    audio_analysis = await model.analyze_audio(audio_path, prompt)
    if verbose:
        console.log(f"Audio Analysis Results: [cyan]{audio_analysis}[/cyan]")
    return {"audio_analysis_results": audio_analysis}


async def save_audio_results(state):
    verbose = state.get("verbose", True)
    results = state["audio_analysis_results"]

    if verbose and results:
        console.rule("[bold green]✅ Audio Analysis Complete[/bold green]")
    results = {
        "source_path": str(Path(state["video_path"]).resolve(strict=False)),
        "audio_analysis": results,
    }
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_audio_analysis.json"
    )

    def _save():
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Results saved to [cyan]{output_path}[/cyan]")
    return {}


async def generate_video_description(state):
    if state.get("processing_type") == "MER" and state.get("cache"):
        output_path = (
            Path(state["video_output_dir"]) / f"{state['video_id']}_video_analysis.json"
        )
        if output_path.exists():
            verbose = state.get("verbose", True)
            if verbose:
                console.log(
                    f"Cache hit for video analysis. Loading from [green]{output_path}[/green]"
                )
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"video_description": data.get("llm_video_summary", "")}

    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Video Content Analysis[/bold]")
    video_path = Path(state["video_path"])
    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]

    processing_type = state.get("processing_type")
    ground_truth_label = state.get("ground_truth_label")
    # if processing_type is video, we pass the label to the prompt
    # otherwise (MER), we do not, because emotion cannot be inferred from video alone.
    has_label = bool(ground_truth_label) if processing_type == "video" else False
    prompt = prompts.get_video_prompt(has_label)
    if has_label:
        prompt = prompt.format(label=ground_truth_label)

    video_description = await model.describe_video(video_path, prompt)
    if verbose and video_description:
        console.log(f"Video Description: [cyan]{video_description}[/cyan]")
    return {"video_description": video_description}


async def save_video_results(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Video Analysis Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_video_analysis.json"
    )
    result_data = {
        "source_path": str(Path(state["video_path"]).resolve(strict=False)),
        "llm_video_summary": state["video_description"],
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Video analysis results saved to [green]{output_path}[/green]")
    return {}


async def extract_full_features(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Full MER Feature Extraction[/bold]")

    audio_path = Path(state["audio_path"])
    au_data_path = Path(state["au_data_path"])

    if not audio_path.exists():
        return {"error": f"Audio file not found at {audio_path} for MER pipeline."}
    if not au_data_path.exists():
        return {"error": f"OpenFace CSV not found at {au_data_path} for MER pipeline."}

    if verbose:
        console.log("Confirmed existence of pre-extracted audio and AU data.")

    return {}


async def filter_by_emotion(state):
    """
    Finds all significant emotional peaks in the video and, for each peak,
    analyzes the primary and secondary emotions present.
    """
    # If caching is enabled for an MER run, try to load existing results first.
    if state.get("processing_type") == "MER" and state.get("cache"):
        output_path = (
            Path(state["video_output_dir"]) / f"{state['video_id']}_au_analysis.json"
        )
        if output_path.exists():
            verbose = state.get("verbose", True)
            if verbose:
                console.log(
                    f"Cache hit for AU analysis. Loading from [green]{output_path}[/green]"
                )
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"detected_emotions": data.get("chronological_emotion_peaks", [])}

    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding all emotional peaks and analyzing mixed emotions...")
    au_data_path = Path(state["au_data_path"])

    try:
        # Run blocking FacialAnalyzer methods in a separate thread
        def _analyze():
            analyzer = FacialAnalyzer(au_data_path)
            return analyzer.get_chronological_emotion_summary(
                peak_height=state.get("threshold", 0.8),
                peak_distance=state.get("peak_distance_frames", 20),
                emotion_threshold=state.get("threshold", 0.8),
            )

        summary_list, is_expressive = await asyncio.to_thread(_analyze)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to analyze facial data: {e}"}

    if verbose:
        if is_expressive:
            console.log(
                f"😊 Expressive. Found {len(summary_list)} distinct emotional peaks."
            )
            for peak_summary in summary_list:
                console.log(f"  - {peak_summary}")
        else:
            console.log(
                "😐 Not expressive enough for multi-peak analysis, classifying as Neutral."
            )

    return {"detected_emotions": summary_list}


async def find_overall_peak_au(state):
    """Finds the single most expressive frame, its emotion, and describes its AUs."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Finding overall peak frame AUs...")
    au_data_path = Path(state["au_data_path"])

    try:
        # Run blocking facial analysis in a separate thread
        def _analyze():
            analyzer = FacialAnalyzer(au_data_path)
            peak_frame_info = analyzer.get_overall_peak_frame_info()
            peak_aus = peak_frame_info["top_aus_intensities"]
            au_description = EmotionAnalyzer.extract_au_description(peak_aus)
            if "Neutral expression" in au_description:
                au_description = "Neutral expression at the overall peak frame."
            return peak_frame_info, au_description

        peak_frame_info, au_description = await asyncio.to_thread(_analyze)

    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to find peak frame AU data: {e}"}

    if verbose:
        console.log(f"Overall Peak AU Description: [yellow]{au_description}[/yellow]")

    return {
        "peak_frame_info": peak_frame_info,
        "peak_frame_au_description": au_description,
    }


async def extract_peak_image(state):
    """Extracts the peak emotional frame from the video for the MER pipeline."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Extracting peak frame image...")

    peak_frame_info = state.get("peak_frame_info")
    if not peak_frame_info:
        return {"error": "Peak frame information not found in state."}

    peak_timestamp = peak_frame_info["timestamp"]
    video_path = Path(state["video_path"])
    peak_frame_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_peak_frame.png"
    )

    if not await FFMpegAdapter.extract_nearby_frame(
        video_path, peak_timestamp, peak_frame_path, verbose
    ):
        return {"error": f"Failed to extract peak frame at timestamp {peak_timestamp}."}

    if verbose:
        console.log(
            f"Extracted peak frame at [yellow]{peak_timestamp:.2f}s[/yellow] to [green]{peak_frame_path}[/green]."
        )
    return {"peak_frame_path": peak_frame_path}


async def generate_peak_frame_visual_description(state):
    """Generates a visual description for the peak frame image."""
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Generating visual description for peak frame...")
    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]
    peak_frame_path = Path(state["peak_frame_path"])

    # No label for peak frame, since this is use for MER.
    prompt = prompts.get_image_prompt()
    visual_obj_desc = await model.describe_image(peak_frame_path, prompt)

    if verbose:
        console.log(f"Peak Frame Visual Description: [cyan]{visual_obj_desc}[/cyan]")
    return {"image_visual_description": visual_obj_desc}


async def synthesize_summary(state):
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final summary...")
    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]
    ground_truth_label = state.get("ground_truth_label")
    task = state.get("task")

    # Dynamically build the context based on available data
    clues = []

    # If a ground truth label exists, it's the most important clue.
    if ground_truth_label:
        label_type = "Sentiment" if task == "Sentiment Analysis" else "Emotion"
        clues.append(f"- Ground Truth {label_type} Label: {ground_truth_label}")

    # Chronological emotions
    detected_emotions = state.get("detected_emotions")
    clues.append(
        f"- Chronological Emotion Peaks by Facial Action Unit: {'; '.join(detected_emotions)}"
    )

    # Peak frame facial expression
    peak_frame_au_desc = state.get("peak_frame_au_description")
    timestamp = state.get("peak_frame_info", {}).get("timestamp", 0)

    clues.append(
        f"- Facial Expression Clues (at overall peak {timestamp:.2f}s): {peak_frame_au_desc}"
    )
    # Peak frame visual context
    image_visual_desc = state.get("image_visual_description")
    clues.append(f"- Visual Context (at overall peak): {image_visual_desc}")

    # Audio analysis
    audio_analysis = state.get("audio_analysis_results", "")
    if audio_analysis:
        clues.append(f"- Audio clues: {audio_analysis}")

    # Video description
    video_description = state.get("video_description", "N/A")
    if video_description and video_description.strip() and video_description != "N/A":
        clues.append(f"- Video Content Overview: {video_description}")

    coarse_summary = "\n".join(clues)

    if verbose:
        console.log("--- Sending Following Clues to LLM for Synthesis ---")
        console.log(coarse_summary)
        console.log("----------------------------------------------------")

    # Select the appropriate prompt template based on the task
    prompt = prompts.get_synthesis_prompt(
        task, has_label=bool(ground_truth_label)
    ).format(context=coarse_summary)

    final_summary = await model.synthesize_summary(prompt)
    return {"final_summary": final_summary}


async def save_mer_results(state):
    verbose = state.get("verbose", True)
    task = state.get("task")
    if verbose:
        console.rule(f"[bold green]✅ Full {task} Complete[/bold green]")
    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_merr_data.json"
    )

    descriptions = {
        "visual_expression": state.get("peak_frame_au_description", "N/A"),
        "visual_objective": state.get("image_visual_description", "N/A"),
        "audio_analysis": state.get("audio_analysis_results", ""),
        "video_content": state.get("video_description", "N/A"),
    }

    result_data = {
        "source_path": str(Path(state["video_path"]).resolve(strict=False)),
        "chronological_emotion_peaks": state.get("detected_emotions", []),
        "overall_peak_frame_info": state["peak_frame_info"],
        "coarse_descriptions_at_peak": descriptions,
        "final_summary": state["final_summary"],
    }

    # Include the ground truth label in the output if it was used
    if state.get("ground_truth_label"):
        result_data["ground_truth_label"] = state["ground_truth_label"]

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Full analysis saved to [green]{output_path}[/green]")
    return {}


async def handle_error(state):
    error_msg = state.get("error", "An unknown error occurred.")
    video_id = state.get("video_id", "unknown_video")
    error_logs_dir = state.get("error_logs_dir", Path("./error_logs"))
    await asyncio.to_thread(error_logs_dir.mkdir, exist_ok=True)
    error_log_path = error_logs_dir / f"{video_id}_error.log"

    console.rule(f"[bold red]❌ Error processing {video_id}[/bold red]")
    console.log(error_msg)
    console.log(f"Saving error details to [cyan]{error_log_path}[/cyan]")

    def _save():
        with open(error_log_path, "w") as f:
            f.write(
                f"Error processing video: {video_id}\n" + "=" * 20 + f"\n{error_msg}"
            )

    await asyncio.to_thread(_save)
    return {"error": error_msg}


async def run_image_analysis(state):
    """
    Runs the full analysis for a single image: AU extraction, AU description, and visual description.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold]Executing: Image Analysis[/bold]")

    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]
    image_path = Path(
        state["video_path"]
    )  # for simplicity, we use video_path for image_path
    au_data_path = Path(state["au_data_path"])

    if not au_data_path.exists():
        return {
            "error": f"AU data file not found at {au_data_path}. The background OpenFace task may have failed."
        }
    if verbose:
        console.log(f"Confirmed OpenFace output at [green]{au_data_path}[/green]")

    try:
        analyzer = await asyncio.to_thread(FacialAnalyzer, au_data_path)
        au_text_desc = analyzer.get_frame_au_summary(threshold=0.8)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"Failed to analyze image facial data: {e}"}

    if verbose:
        console.log(f"Detected AUs: [yellow]{au_text_desc}[/yellow]")

    async def get_precomputed_value(value):
        return value

    if au_text_desc == "Neutral expression.":
        llm_au_desc_task = get_precomputed_value(
            "A neutral facial expression was detected."
        )
    else:
        prompt = prompts.get_facial_expression_prompt().format(au_text=au_text_desc)
        llm_au_desc_task = model.describe_facial_expression(prompt)
    ground_truth_label = state.get("ground_truth_label")
    has_label = bool(ground_truth_label)
    visual_prompt = prompts.get_image_prompt(has_label)
    if has_label:
        visual_prompt = visual_prompt.format(label=ground_truth_label)
    visual_desc_task = model.describe_image(image_path, visual_prompt)

    llm_au_description, image_visual_description = await asyncio.gather(
        llm_au_desc_task, visual_desc_task
    )

    if verbose:
        console.log(f"LLM AU Description: [cyan]{llm_au_description}[/cyan]")
        console.log(f"LLM Visual Description: [cyan]{image_visual_description}[/cyan]")

    return {
        "au_text_description": au_text_desc,
        "llm_au_description": llm_au_description,
        "image_visual_description": image_visual_description,
    }


async def synthesize_image_summary(state):
    """
    Synthesizes the final summary for the image analysis.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.log("Synthesizing final image summary...")
    model: LLMModels = state["models"].model_instance
    prompts: PromptTemplates = state["prompts"]
    ground_truth_label = state.get("ground_truth_label")
    task = state.get("task")

    clues = []
    if ground_truth_label:
        clues.append(f"- Ground Truth Label: {ground_truth_label}")

    clues.append(f"- Facial Expression Clues: {state['llm_au_description']}")
    clues.append(f"- Visual Context: {state['image_visual_description']}")

    context = "\n".join(clues)
    prompt = prompts.get_image_synthesis_prompt(
        task=task, has_label=bool(ground_truth_label)
    ).format(context=context)

    final_summary = await model.synthesize_summary(prompt)
    if verbose:
        console.log(f"Final Summary: [magenta]{final_summary}[/magenta]")

    return {"final_summary": final_summary}


async def save_image_results(state):
    """
    Saves the image analysis results to a JSON file.
    """
    verbose = state.get("verbose", True)
    if verbose:
        console.rule("[bold green]✅ Image Analysis Complete[/bold green]")

    output_path = (
        Path(state["video_output_dir"]) / f"{state['video_id']}_image_analysis.json"
    )

    result_data = {
        "source_path": str(Path(state["video_path"]).resolve(strict=False)),
        "au_text_description": state["au_text_description"],
        "llm_au_description": state["llm_au_description"],
        "image_visual_description": state["image_visual_description"],
        "final_summary": state["final_summary"],
    }

    if state.get("ground_truth_label"):
        result_data["ground_truth_label"] = state["ground_truth_label"]

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(_save)
    if verbose:
        console.print(f"Image analysis results saved to [green]{output_path}[/green]")
    return {}
