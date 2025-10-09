from __future__ import annotations

from typing import Dict, Optional, List
from PIL import Image
import open_clip
import torch
import laion_clap
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def compute_clip_image_text_score(
    image_path: Optional[str], 
    text: Optional[str],
    clip_model=None,
    clip_preprocess=None,
    clip_tokenizer=None
) -> float:
    """
    Reference-free image-text grounding score using CLIP cosine similarity.
    Returns 0.0 if dependencies are missing or inputs unavailable.
    
    Args:
        image_path: Path to image file
        text: Text to compare
        clip_model: Pre-initialized CLIP model (optional, will initialize if None)
        clip_preprocess: Pre-initialized CLIP preprocess function (optional)
        clip_tokenizer: Pre-initialized CLIP tokenizer (optional)
    """
    if not image_path or not text:
        return 0.0

    try:
        # Use provided models or initialize new ones
        if clip_model is None or clip_preprocess is None or clip_tokenizer is None:
            if hasattr(open_clip, "create_model_and_transforms"):
                model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
            else:
                model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        else:
            model = clip_model
            preprocess = clip_preprocess
            tokenizer = clip_tokenizer
            device = next(model.parameters()).device
        
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text_tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).item()
        return float(sim)
    except Exception:
        return 0.0


def compute_clap_audio_text_score(audio_path: Optional[str], text: Optional[str], clap_model=None) -> float:
    """
    Reference-free audio-text grounding using LAION-CLAP cosine similarity.
    Returns 0.0 if dependencies are missing or inputs unavailable.
    
    Args:
        audio_path: Path to audio file
        text: Text to compare
        clap_model: Pre-initialized CLAP model (optional, will initialize if None)
    """
    if not audio_path or not text:
        return 0.0
    try:
        # Use provided model or initialize new one
        if clap_model is None:
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        else:
            model = clap_model
            device = next(model.parameters()).device
            
        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True).to(device)
            text_embed = model.get_text_embedding([text], use_tensor=True).to(device)
            audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            sim = (audio_embed @ text_embed.T).item()
        return float(sim)
    except Exception:
        return 0.0


def compute_nli_consistency_scores(
    premise: Optional[str], 
    hypotheses: List[str],
    nli_model=None,
    nli_tokenizer=None
) -> Dict[str, float]:
    """
    Use MNLI model to compute entailment vs contradiction rates of hypotheses given premise.
    Returns zeros if dependencies are missing.
    
    Args:
        premise: The premise text
        hypotheses: List of hypothesis texts
        nli_model: Pre-initialized NLI model (optional, will initialize if None)
        nli_tokenizer: Pre-initialized NLI tokenizer (optional)
    """
    if not premise or not hypotheses:
        return {"nli_entail_rate": 0.0, "nli_contra_rate": 0.0}
    try:
        # Use provided models or initialize new ones
        if nli_model is None or nli_tokenizer is None:
            model_name = "microsoft/deberta-large-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        else:
            model = nli_model
            tokenizer = nli_tokenizer
            device = next(model.parameters()).device
            
        entail_cnt = 0
        contra_cnt = 0
        for hyp in hypotheses:
            inputs = tokenizer(premise, hyp, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[0]
            # MNLI label order: contradiction, neutral, entailment
            probs = torch.softmax(logits, dim=-1)
            if probs[2].item() >= 0.5:  # Lower threshold for entailment
                entail_cnt += 1
            if probs[0].item() >= 0.5:  # Lower threshold for contradiction
                contra_cnt += 1
        n = max(1, len(hypotheses))
        # Calculate consistency score: positive for more entailment, negative for more contradiction
        consistency_score = (entail_cnt - contra_cnt) / n
        return {
            "nli_consistency_score": max(0.0, consistency_score),  # Only keep positive consistency
            "nli_entail_rate": entail_cnt / n,
            "nli_contra_rate": contra_cnt / n,
        }
    except Exception:
        return {"nli_consistency_score": 0.0, "nli_entail_rate": 0.0, "nli_contra_rate": 0.0}


def compute_asr_wer(reference_transcript: Optional[str], audio_path: Optional[str], whisper_model=None) -> float:
    """
    Compute WER between a strong ASR transcript (Whisper) and the model transcript.
    If Whisper missing, returns 0.0 (neutral).
    
    Args:
        reference_transcript: The reference transcript to compare against
        audio_path: Path to audio file
        whisper_model: Pre-initialized Whisper model (optional, will initialize if None)
    """
    if not audio_path or not reference_transcript:
        return 0.0
    
    # Check if audio file exists
    try:
        from pathlib import Path
        if not Path(audio_path).exists():
            return 0.0
    except Exception:
        return 0.0
    
    try:
        # Use provided model or initialize new one
        if whisper_model is None:
            # Initialize HuggingFace pipeline on-the-fly
            try:
                from transformers import pipeline
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    device=device
                )
                result = pipe(audio_path)
                asr_text = result["text"] if isinstance(result, dict) else str(result)
            except Exception:
                # If Whisper fails, return neutral score
                return 0.0
        else:
            # Use provided HuggingFace pipeline model
            try:
                result = whisper_model(audio_path)
                asr_text = result["text"] if isinstance(result, dict) else str(result)
            except Exception:
                return 0.0

        if not asr_text:
            return 0.0
            
        return _wer(asr_text, reference_transcript)
    except Exception:
        return 0.0


def _wer(hyp: str, ref: str) -> float:
    hyp_tokens = _tokenize(hyp)
    ref_tokens = _tokenize(ref)
    if not ref_tokens:
        return 0.0
    # Levenshtein distance
    dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        dp[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    dist = dp[-1][-1]
    return dist / len(ref_tokens)


def _tokenize(s: str) -> List[str]:
    """
    Tokenize text for WER calculation with Chinese support.
    For Chinese: character-level tokenization after normalization.
    For other languages: word-level tokenization.
    """
    import re
    
    # Normalize the text
    normalized = _normalize_text(s)
    
    # Check if text contains Chinese characters
    if re.search(r'[\u4e00-\u9fff]', normalized):
        # Chinese text: use character-level tokenization
        # Remove non-Chinese characters and spaces, then split into characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', normalized)
        return chinese_chars
    else:
        # Non-Chinese text: use word-level tokenization
        return re.findall(r"\w+", normalized.lower())


def _normalize_text(s: str) -> str:
    """
    Normalize text for better comparison, especially for Chinese.
    """
    import re
    
    # Basic cleanup
    text = s.strip()
    
    # Try to convert traditional Chinese to simplified Chinese
    try:
        # Try using opencc if available (optional dependency)
        import opencc  # type: ignore
        converter = opencc.OpenCC('t2s.json')  # Traditional to Simplified
        text = converter.convert(text)
    except ImportError:
        # Fallback: basic manual conversion for common characters
        traditional_to_simplified = {
            '學': '学', '習': '习', '語': '语', '話': '话', '時': '时', '間': '间',
            '個': '个', '們': '们', '來': '来', '這': '这', '那': '那', '裡': '里',
            '說': '说', '聽': '听', '會': '会', '點': '点', '還': '还', '過': '过',
            '現': '现', '發': '发', '經': '经', '準': '准', '標': '标', '課': '课',
            '題': '题', '問': '问', '答': '答', '開': '开', '關': '关', '門': '门',
            '窗': '窗', '書': '书', '讀': '读', '寫': '写', '字': '字', '詞': '词',
            '義': '义', '思': '思', '想': '想', '記': '记', '憶': '忆', '識': '识',
            '認': '认', '知': '知', '覺': '觉', '感': '感', '情': '情', '愛': '爱',
            '喜': '喜', '歡': '欢', '討': '讨', '厭': '厌', '興': '兴', '趣': '趣'
        }
        for trad, simp in traditional_to_simplified.items():
            text = text.replace(trad, simp)
    
    # Remove extra whitespace and punctuation for Chinese
    if re.search(r'[\u4e00-\u9fff]', text):
        # For Chinese text, remove punctuation and normalize spaces
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        text = re.sub(r'\s+', '', text)  # Remove all spaces for Chinese
    else:
        # For non-Chinese text, normalize spaces only
        text = re.sub(r'\s+', ' ', text)
    
    return text
