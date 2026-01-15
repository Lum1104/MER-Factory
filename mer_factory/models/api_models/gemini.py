from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from rich.console import Console
import base64
import mimetypes
import asyncio


console = Console(stderr=True)


class GeminiModel:
    """
    A class to handle all interactions with the Google Gemini API.
    """

    def __init__(self, api_key: str, verbose: bool = True):
        self.verbose = verbose
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, temperature=0
        )
        self.vision_model = self.model  # Gemini models are multimodal
        if self.verbose:
            console.log("Gemini models initialized.")

    async def describe_facial_expression(self, prompt: str) -> str:
        """Generates a description from AU text using Gemini."""
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(
                f"[bold red]❌ Error describing facial expression: {e}[/bold red]"
            )
            return f"Error generating facial description: {e}"

    async def describe_image(self, image_path: Path, prompt: str) -> str:
        """Generates a description for an image file using Gemini."""
        try:
            image_data = await asyncio.to_thread(
                lambda: base64.b64encode(image_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing image: {e}[/bold red]")
            return f"Error: {e}"

    async def analyze_audio(self, audio_path: Path, prompt: str) -> dict:
        """Analyzes an audio file and returns a structured dictionary using Gemini."""
        try:
            audio_data = await asyncio.to_thread(
                lambda: base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "media", "data": audio_data, "mime_type": mime_type},
                ]
            )
            chain = self.vision_model | StrOutputParser()
            str_response = await chain.ainvoke([message])

            return str_response

        except Exception as e:
            console.log(f"[bold red]❌ Error analyzing audio: {e}[/bold red]")
            return {
                "transcript": "",
                "tone_description": "",
            }

    async def describe_video(self, video_path: Path, prompt: str) -> str:
        """Generates a description for a video using Gemini."""
        try:
            video_data = await asyncio.to_thread(
                lambda: base64.b64encode(video_path.read_bytes()).decode("utf-8")
            )
            mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "media", "data": video_data, "mime_type": mime_type},
                ]
            )
            chain = self.vision_model | StrOutputParser()
            return await chain.ainvoke([message])
        except Exception as e:
            console.log(f"[bold red]❌ Error describing video: {e}[/bold red]")
            raise

    async def synthesize_summary(self, prompt: str) -> str:
        """Synthesizes a final summary from context using Gemini."""
        try:
            chain = self.model | StrOutputParser()
            return await chain.ainvoke(prompt)
        except Exception as e:
            console.log(f"[bold red]❌ Error synthesizing summary: {e}[/bold red]")
            raise
