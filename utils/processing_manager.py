import asyncio
import functools
import traceback
from pathlib import Path
from typing import List, Dict, Coroutine, Any

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskID,
)

from mer_factory.state import MERRState
from mer_factory.models import LLMModels
from mer_factory.prompts import PromptTemplates
from .config import (
    AppConfig,
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    FINAL_OUTPUT_FILENAMES,
    ProcessingType,
)
from tools.ffmpeg_adapter import FFMpegAdapter
from tools.openface_adapter import OpenFaceAdapter

console = Console(stderr=True)


def build_initial_state(
    file_path: Path, config: AppConfig, models: LLMModels, prompts: PromptTemplates
) -> MERRState:
    """Builds the initial state dictionary for a file to be processed by the graph."""
    file_id = file_path.stem
    file_output_dir = config.output_dir / file_id
    file_output_dir.mkdir(exist_ok=True)
    is_image = file_path.suffix.lower() in IMAGE_EXTENSIONS
    is_audio = file_path.suffix.lower() in AUDIO_EXTENSIONS

    if is_audio:
        current_processing_type = "audio"
        video_path = file_path
        audio_path = file_path
        au_data_path = None
    elif is_image:
        current_processing_type = "image"
        video_path = file_path
        audio_path = None
        au_data_path = file_output_dir / f"{file_id}.csv"
    else:  # It's a video
        current_processing_type = config.processing_type.value
        video_path = file_path
        audio_path = file_output_dir / f"{file_id}.wav"
        au_data_path = file_output_dir / f"{file_id}.csv"

    state = {
        "video_path": video_path,
        "audio_path": audio_path,
        "au_data_path": au_data_path,
        "output_dir": config.output_dir,
        "processing_type": current_processing_type,
        "task": config.task.value,
        "models": models,
        "prompts": prompts,
        "threshold": config.threshold,
        "peak_distance_frames": config.peak_distance_frames,
        "verbose": config.verbose,
        "error_logs_dir": config.error_logs_dir,
        "video_id": file_id,
        "video_output_dir": file_output_dir,
        "cache": config.cache,
    }

    ground_truth_label = config.labels.get(file_id)
    if ground_truth_label:
        state["ground_truth_label"] = ground_truth_label
        if config.verbose:
            console.log(
                f"Found ground truth label for {file_id}: [bold yellow]{ground_truth_label}[/bold yellow]"
            )

    return state


async def _run_async_job(
    coro: Coroutine,
    file_path: Path,
    task_id: TaskID,
    progress: Progress,
    results: Dict[str, int],
    semaphore: asyncio.Semaphore,
):
    """A helper to run a single asynchronous processing job with a semaphore and progress updates."""
    async with semaphore:
        try:
            final_state = await coro
            if final_state and final_state.get("error"):
                results["failure"] += 1
            else:
                results["success"] += 1
        except Exception as e:
            results["failure"] += 1
            console.print(
                f"[bold red]FATAL ERROR processing {file_path.name}: {e}[/bold red]"
            )
            traceback.print_exc()
        finally:
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Finished [cyan]{file_path.name}[/cyan]",
                )


def _run_sync_job(
    invoke_func: Any,
    file_path: Path,
    task_id: TaskID,
    progress: Progress,
    results: Dict[str, int],
):
    """A helper to run a single synchronous processing job with progress updates."""
    try:
        final_state = invoke_func()
        if final_state and final_state.get("error"):
            results["failure"] += 1
        else:
            results["success"] += 1
    except Exception as e:
        results["failure"] += 1
        console.print(
            f"[bold red]FATAL ERROR processing {file_path.name}: {e}[/bold red]"
        )
        traceback.print_exc()
    finally:
        if progress and task_id is not None:
            progress.update(
                task_id,
                advance=1,
                description=f"Finished [cyan]{file_path.name}[/cyan]",
            )


async def run_main_processing(
    files_to_process: List[Path],
    graph_app: Any,
    initial_state_builder: functools.partial,
    config: AppConfig,
    is_sync: bool,
) -> Dict[str, int]:
    """
    Manages the main processing loop, checking for cached final outputs first,
    then dispatching to sync or async helpers.
    """
    results = {"success": 0, "failure": 0, "skipped": 0}
    total_files = len(files_to_process)
    semaphore = asyncio.Semaphore(config.concurrency)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=config.verbose,
    ) as progress:
        task = progress.add_task("Processing files...", total=total_files)

        final_output_suffix = FINAL_OUTPUT_FILENAMES.get(config.processing_type)

        files_to_run = []
        if config.cache and final_output_suffix:
            for file_path in files_to_process:
                file_id = file_path.stem
                file_output_dir = config.output_dir / file_id
                final_output_path = file_output_dir / f"{file_id}{final_output_suffix}"

                if final_output_path.exists():
                    if config.verbose:
                        console.log(
                            f"Cache hit: Final output for [cyan]{file_path.name}[/cyan] found. Skipping."
                        )
                    progress.update(
                        task,
                        advance=1,
                        description=f"Skipped [cyan]{file_path.name}[/cyan]",
                    )
                    results["skipped"] += 1
                else:
                    files_to_run.append(file_path)
        else:
            files_to_run = files_to_process

        if not files_to_run:
            return results

        if is_sync:
            # Synchronous execution
            for file_path in files_to_run:
                initial_state = initial_state_builder(file_path=file_path)
                invoke_func = functools.partial(graph_app.invoke, initial_state)
                _run_sync_job(invoke_func, file_path, task, progress, results)
        else:
            # Asynchronous execution
            tasks = []
            for file_path in files_to_run:
                initial_state = initial_state_builder(file_path=file_path)
                coro = graph_app.ainvoke(initial_state)
                tasks.append(
                    _run_async_job(coro, file_path, task, progress, results, semaphore)
                )
            if tasks:
                await asyncio.gather(*tasks)

    return results


async def run_feature_extraction(files_to_process: List[Path], config: AppConfig):
    """
    Asynchronously runs FFmpeg and OpenFace feature extraction on the files.
    """
    extraction_semaphore = asyncio.Semaphore(4)

    files_for_openface = [
        f
        for f in files_to_process
        if (
            f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in VIDEO_EXTENSIONS
        )
        and config.processing_type in [ProcessingType.MER, ProcessingType.AU]
    ]
    files_for_ffmpeg = [
        f
        for f in files_to_process
        if f.suffix.lower() in VIDEO_EXTENSIONS
        and config.processing_type in [ProcessingType.MER, ProcessingType.AUDIO]
    ]

    async def run_openface_job(file_path: Path, task_id: TaskID, progress: Progress):
        async with extraction_semaphore:
            file_output_dir = config.output_dir / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
            if not (file_output_dir / f"{file_path.stem}.csv").exists():
                try:
                    await OpenFaceAdapter.run_feature_extraction(
                        file_path, file_output_dir, config.verbose
                    )
                except Exception as e:
                    console.log(
                        f"[bold red]OpenFace extraction failed for {file_path.name}: {e}[/bold red]"
                    )
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

    async def run_ffmpeg_job(file_path: Path, task_id: TaskID, progress: Progress):
        async with extraction_semaphore:
            file_output_dir = config.output_dir / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
            if not (file_output_dir / f"{file_path.stem}.wav").exists():
                try:
                    await FFMpegAdapter.extract_audio(
                        file_path,
                        file_output_dir / f"{file_path.stem}.wav",
                        config.verbose,
                    )
                except Exception as e:
                    console.log(
                        f"[bold red]FFmpeg extraction failed for {file_path.name}: {e}[/bold red]"
                    )
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=config.verbose,
    ) as progress:
        tasks = []
        if files_for_openface:
            openface_task_id = progress.add_task(
                "[bold blue]OpenFace", total=len(files_for_openface)
            )
            tasks.extend(
                [
                    run_openface_job(fp, openface_task_id, progress)
                    for fp in files_for_openface
                ]
            )

        if files_for_ffmpeg:
            ffmpeg_task_id = progress.add_task(
                "[bold green]FFmpeg", total=len(files_for_ffmpeg)
            )
            tasks.extend(
                [
                    run_ffmpeg_job(fp, ffmpeg_task_id, progress)
                    for fp in files_for_ffmpeg
                ]
            )

        if tasks:
            await asyncio.gather(*tasks)
        elif not config.silent:
            console.log("No feature extraction needed for the provided files.")
