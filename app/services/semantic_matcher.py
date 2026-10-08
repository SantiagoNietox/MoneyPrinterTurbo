import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    TextEmbedding = None
    HAS_FASTEMBED = False

from app.models.schema import MaterialInfo
from app.utils import utils

_DEFAULT_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model_instance = None
_model_lock = threading.Lock()


def _ensure_model_loaded() -> Optional[Any]:
    """Carga de forma perezosa y segura en hilos el modelo de embeddings semánticos."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if not HAS_FASTEMBED:
        logger.warning("[SemanticMatcher] fastembed no está instalado; búsqueda semántica omitida.")
        return None

    with _model_lock:
        if _model_instance is None:
            try:
                logger.info(f"[SemanticMatcher] Cargando modelo de embeddings: {_DEFAULT_MULTILINGUAL_MODEL}")
                _model_instance = TextEmbedding(model_name=_DEFAULT_MULTILINGUAL_MODEL)
                logger.success("[SemanticMatcher] Modelo de embeddings inicializado correctamente.")
            except Exception as exc:
                logger.error(f"[SemanticMatcher] Error al cargar el modelo de embeddings: {exc}")
                _model_instance = None
    return _model_instance


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Genera vectores L2-normalizados para una lista de textos."""
    model = _ensure_model_loaded()
    if model is None or not texts:
        return None

    cleaned_texts = [t.strip() if t and t.strip() else "footage" for t in texts]
    try:
        embeddings = list(model.embed(cleaned_texts))
        arr = np.array(embeddings, dtype=np.float32)
        # Normalizar L2 para que similitud coseno = producto punto
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return arr / norms
    except Exception as exc:
        logger.error(f"[SemanticMatcher] Error al vectorizar textos: {exc}")
        return None


def split_script_into_scenes(script: str, total_duration: float = 0.0) -> List[Dict[str, Any]]:
    """
    Divide el guion en oraciones y cláusulas con estimación temporal proporcional.
    Cada escena representa un momento narrativo específico del video.
    """
    clean_script = utils.remove_pause_tags(script or "").strip()
    if not clean_script:
        return []

    # Dividir por puntos, signos de interrogación/exclamación o saltos de línea
    raw_sentences = re.split(r'(?<=[.!?…\n])\s+', clean_script)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

    if not sentences:
        sentences = [clean_script]

    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= 0:
        total_words = 1

    scenes = []
    current_time = 0.0
    for idx, s in enumerate(sentences):
        w_count = len(s.split())
        if total_duration > 0:
            scene_duration = (w_count / total_words) * total_duration
        else:
            scene_duration = max(2.5, w_count * 0.35)

        scenes.append({
            "index": idx,
            "text": s,
            "start": round(current_time, 2),
            "end": round(current_time + scene_duration, 2),
            "duration": round(scene_duration, 2),
            "word_count": w_count,
        })
        current_time += scene_duration

    return scenes


def get_material_text_representation(item: MaterialInfo) -> str:
    """Extrae la mejor descripción textual disponible de un MaterialInfo para vectorizarla."""
    parts: List[str] = []

    # 1. Extraer de source_info si es un diccionario
    s_info = getattr(item, "source_info", None)
    if isinstance(s_info, dict):
        if s_info.get("search_term"):
            parts.append(str(s_info["search_term"]))
        if s_info.get("title"):
            parts.append(str(s_info["title"]))
        if s_info.get("description"):
            parts.append(str(s_info["description"]))
        if s_info.get("source_page"):
            slug = re.sub(r'https?://[^/]+/', '', str(s_info["source_page"]))
            slug_words = re.sub(r'[\-_/.]', ' ', slug).strip()
            if len(slug_words) > 5 and not slug_words.isdigit():
                parts.append(slug_words)

    # 2. Extraer atributos directos si existen
    for attr in ("search_term", "title", "description", "tag", "tags"):
        val = getattr(item, attr, None)
        if val and isinstance(val, str) and val not in parts:
            parts.append(val)
        elif val and isinstance(val, (list, tuple)):
            parts.extend([str(v) for v in val if str(v) not in parts])

    # 3. Si la URL contiene un nombre de archivo descriptivo
    url_val = getattr(item, "url", None)
    if url_val:
        basename = os.path.basename(str(url_val).split("?")[0])
        clean_base = re.sub(r'[\-_.]', ' ', basename)
        if len(clean_base) > 5 and not clean_base.lower().startswith("scene clip"):
            parts.append(clean_base)

    combined = " ".join(parts).strip()
    return combined if combined else "cinematic film footage"


def match_and_order_materials_semantically(
    script: str,
    candidate_materials: List[MaterialInfo],
    total_duration: float,
    clip_duration: float = 3.0,
) -> List[MaterialInfo]:
    """
    Empareja los materiales candidatos con el guion según máxima similitud semántica.
    Para cada intervalo de tiempo en la línea temporal (t = 0, 3s, 6s...), calcula qué
    oración del guion se está diciendo y selecciona el candidato de metraje más relevante,
    evitando repeticiones consecutivas y penalizando el sobreuso.
    """
    if not candidate_materials:
        return []

    if len(candidate_materials) <= 1 or not script:
        return candidate_materials

    scenes = split_script_into_scenes(script, total_duration=total_duration)
    if not scenes:
        return candidate_materials

    scene_texts = [s["text"] for s in scenes]
    scene_vectors = embed_texts(scene_texts)

    material_texts = [get_material_text_representation(m) for m in candidate_materials]
    material_vectors = embed_texts(material_texts)

    if scene_vectors is None or material_vectors is None:
        logger.warning("[SemanticMatcher] Fallo en vectorización; usando orden original.")
        return candidate_materials

    # Matriz de similitud coseno: Forma (num_scenes, num_materials)
    sim_matrix = np.dot(scene_vectors, material_vectors.T)

    num_slots = max(1, int(math.ceil(total_duration / max(1.0, clip_duration))))
    ordered_materials: List[MaterialInfo] = []
    used_counts = {i: 0 for i in range(len(candidate_materials))}
    last_chosen_idx = -1

    for slot_idx in range(num_slots):
        slot_time = slot_idx * clip_duration

        # Encontrar qué escena corresponde a este instante de tiempo
        scene_idx = 0
        for s_i, s in enumerate(scenes):
            if s["start"] <= slot_time < s["end"]:
                scene_idx = s_i
                break
            elif slot_time >= s["end"]:
                scene_idx = s_i

        scores = sim_matrix[scene_idx].copy()

        # Penalizaciones
        for m_i in range(len(candidate_materials)):
            if used_counts[m_i] > 0:
                scores[m_i] -= (used_counts[m_i] * 0.35)
            if m_i == last_chosen_idx:
                scores[m_i] -= 0.60

        best_idx = int(np.argmax(scores))
        raw_sim = float(sim_matrix[scene_idx, best_idx])

        selected_item = candidate_materials[best_idx]
        ordered_materials.append(selected_item)
        used_counts[best_idx] += 1
        last_chosen_idx = best_idx

        logger.debug(
            f"[SemanticMatcher] Slot {slot_idx+1}/{num_slots} (t={slot_time:.1f}s) -> "
            f"Escena {scene_idx+1} [Sim={raw_sim:.3f}]: {material_texts[best_idx][:50]!r}"
        )

    logger.success(
        f"[SemanticMatcher] Emparejamiento semántico completado: {len(ordered_materials)} clips organizados "
        f"a lo largo de {len(scenes)} escenas narrativas."
    )
    return ordered_materials


def match_and_order_contextual_materials(
    script: str,
    video_candidates: List[MaterialInfo],
    photo_candidates: List[MaterialInfo],
    total_duration: float,
    clip_duration: float = 3.0,
) -> List[Tuple[str, MaterialInfo]]:
    """
    Empareja y organiza metraje real (video clips) y fotos de archivo (Ken Burns)
    según la progresión semántica del guion y la cadencia cinematográfica requerida.

    Cadencia objetivo:
    - Alternancia dinámica entre video real en movimiento y fotos de archivo (Ken Burns).
    - Máxima afinidad semántica: Cada clip representa visualmente la frase o idea dicha en ese instante.
    - Evita repeticiones consecutivas y diversifica el uso del catálogo.

    Retorna una lista de tuplas: [("video", MaterialInfo), ("photo", MaterialInfo), ...]
    """
    num_slots = max(1, int(math.ceil(total_duration / max(1.0, clip_duration))))

    # Si solo hay uno de los tipos disponible
    if not video_candidates and not photo_candidates:
        return []
    if not video_candidates:
        ordered_photos = match_and_order_materials_semantically(
            script=script, candidate_materials=photo_candidates,
            total_duration=total_duration, clip_duration=clip_duration,
        )
        return [("photo", p) for p in ordered_photos]
    if not photo_candidates:
        ordered_videos = match_and_order_materials_semantically(
            script=script, candidate_materials=video_candidates,
            total_duration=total_duration, clip_duration=clip_duration,
        )
        return [("video", v) for v in ordered_videos]

    scenes = split_script_into_scenes(script, total_duration=total_duration)
    if not scenes:
        # Fallback por alternancia simple
        results: List[Tuple[str, MaterialInfo]] = []
        v_idx, p_idx = 0, 0
        for slot in range(num_slots):
            if slot % 2 == 0 and v_idx < len(video_candidates):
                results.append(("video", video_candidates[v_idx % len(video_candidates)]))
                v_idx += 1
            else:
                results.append(("photo", photo_candidates[p_idx % len(photo_candidates)]))
                p_idx += 1
        return results

    scene_texts = [s["text"] for s in scenes]
    scene_vectors = embed_texts(scene_texts)

    video_texts = [get_material_text_representation(m) for m in video_candidates]
    video_vectors = embed_texts(video_texts)

    photo_texts = [get_material_text_representation(m) for m in photo_candidates]
    photo_vectors = embed_texts(photo_texts)

    if scene_vectors is None or video_vectors is None or photo_vectors is None:
        logger.warning("[SemanticMatcher] Vectorización incompleta; recurriendo a alternancia round-robin.")
        results = []
        v_idx, p_idx = 0, 0
        for slot in range(num_slots):
            if slot % 2 == 0:
                results.append(("video", video_candidates[v_idx % len(video_candidates)]))
                v_idx += 1
            else:
                results.append(("photo", photo_candidates[p_idx % len(photo_candidates)]))
                p_idx += 1
        return results

    video_sim = np.dot(scene_vectors, video_vectors.T)
    photo_sim = np.dot(scene_vectors, photo_vectors.T)

    results: List[Tuple[str, MaterialInfo]] = []
    used_v_counts = {i: 0 for i in range(len(video_candidates))}
    used_p_counts = {i: 0 for i in range(len(photo_candidates))}
    last_chosen_type: Optional[str] = None
    last_chosen_idx: int = -1

    for slot_idx in range(num_slots):
        slot_time = slot_idx * clip_duration

        # Ubicar escena activa
        scene_idx = 0
        for s_i, s in enumerate(scenes):
            if s["start"] <= slot_time < s["end"]:
                scene_idx = s_i
                break
            elif slot_time >= s["end"]:
                scene_idx = s_i

        preferred_type = "video" if (slot_idx % 2 == 0) else "photo"

        # Puntuaciones de video
        v_scores = video_sim[scene_idx].copy()
        for v_i in range(len(video_candidates)):
            if used_v_counts[v_i] > 0:
                v_scores[v_i] -= (used_v_counts[v_i] * 0.35)
            if last_chosen_type == "video" and last_chosen_idx == v_i:
                v_scores[v_i] -= 0.60
            if preferred_type == "video":
                v_scores[v_i] += 0.18  # Bono por cadencia rítmica

        # Puntuaciones de fotos
        p_scores = photo_sim[scene_idx].copy()
        for p_i in range(len(photo_candidates)):
            if used_p_counts[p_i] > 0:
                p_scores[p_i] -= (used_p_counts[p_i] * 0.35)
            if last_chosen_type == "photo" and last_chosen_idx == p_i:
                p_scores[p_i] -= 0.60
            if preferred_type == "photo":
                p_scores[p_i] += 0.18  # Bono por cadencia rítmica

        best_v_idx = int(np.argmax(v_scores))
        best_p_idx = int(np.argmax(p_scores))

        best_v_val = float(v_scores[best_v_idx])
        best_p_val = float(p_scores[best_p_idx])

        if best_v_val >= best_p_val:
            chosen_type = "video"
            chosen_idx = best_v_idx
            chosen_item = video_candidates[best_v_idx]
            used_v_counts[best_v_idx] += 1
            raw_sim = float(video_sim[scene_idx, best_v_idx])
            desc = video_texts[best_v_idx]
        else:
            chosen_type = "photo"
            chosen_idx = best_p_idx
            chosen_item = photo_candidates[best_p_idx]
            used_p_counts[best_p_idx] += 1
            raw_sim = float(photo_sim[scene_idx, best_p_idx])
            desc = photo_texts[best_p_idx]

        results.append((chosen_type, chosen_item))
        last_chosen_type = chosen_type
        last_chosen_idx = chosen_idx

        logger.debug(
            f"[SemanticMatcher] Slot {slot_idx+1}/{num_slots} (t={slot_time:.1f}s) -> "
            f"Tipo: {chosen_type.upper()} [Sim={raw_sim:.3f}] matching Escena {scene_idx+1}: {desc[:40]!r}"
        )

    logger.success(
        f"[SemanticMatcher] Orquestación semántica contextual lista: {len(results)} cortes planificados "
        f"({sum(1 for t, _ in results if t == 'video')} videos, {sum(1 for t, _ in results if t == 'photo')} fotos)."
    )
    return results
