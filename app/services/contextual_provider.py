import hashlib
import io
import math
import os
import re
import shutil
import subprocess
import urllib.parse
from typing import Any, List, Optional
from uuid import uuid4

import requests
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    RatelimitException = Exception
    DuckDuckGoSearchException = Exception

_DOCUMENTARY_USER_AGENT = (
    "MoneyPrinterTurbo/1.3.8 (https://github.com/harry0703/MoneyPrinterTurbo; "
    "archival cinema documentary engine)"
)
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_MIN_IMAGE_DIMENSION = 450


class ContextualMediaProvider:
    """
    Motor editorial de imágenes reales de archivo, prensa y rodaje
    para el canal 'La Otra Pantalla'.
    
    Busca fotografías históricas, fotogramas de producción y rodaje real mediante:
    1. Yandex Images (motor primario para fotogramas de cine, maquillaje, rodaje y prensa)
    2. DuckDuckGo Images (respaldo abierto)
    3. Wikimedia Commons REST API (documentos y archivos históricos abiertos)
    4. Internet Archive (hemeroteca y archivos de época)
    """

    def __init__(self, min_dimension: int = DEFAULT_MIN_IMAGE_DIMENSION):
        self.min_dimension = min_dimension

    def _generate_search_variations(self, raw_query: str) -> List[str]:
        """
        Descompone la consulta en variantes optimizadas para motores de búsqueda visuales.
        """
        variations: List[str] = []
        seen = set()

        def add_v(q: str):
            clean = " ".join(q.split()).strip()
            if clean and len(clean) >= 3 and clean.lower() not in seen:
                seen.add(clean.lower())
                variations.append(clean)

        add_v(raw_query)

        # 1. Eliminar sufijos documentales comunes
        no_suffixes = re.sub(
            r"\b(behind the scenes|vintage press photo|production still|archival document|"
            r"on set photo|raw footage frame|rare archival photo|documentary interview|"
            r"original poster|vintage photo|on set|newspaper clipping vintage|original storyboard)\b",
            "",
            raw_query,
            flags=re.IGNORECASE,
        ).strip()
        add_v(no_suffixes)

        # 2. Agregar variante cinematográfica directa
        if no_suffixes:
            add_v(f"{no_suffixes} behind the scenes")
            add_v(f"{no_suffixes} production still")

        # 3. Limpiar conectores y palabras secundarias
        core = re.sub(
            r"\b(directing|directed by|filming|talking|interview with|looking at|during|"
            r"and|with|the set of|at|in|of|the|a|an)\b",
            " ",
            no_suffixes,
            flags=re.IGNORECASE,
        ).strip()
        add_v(core)

        words = core.split()
        if len(words) >= 4:
            add_v(" ".join(words[:3]))
            add_v(" ".join(words[-3:]))

        return variations

    def search_images(
        self,
        search_term: str,
        minimum_duration: int = 4,
        video_aspect: VideoAspect = VideoAspect.portrait,
        save_dir: str = "",
        max_results: int = 5,
    ) -> List[MaterialInfo]:
        """
        Busca, descarga y valida imágenes relevantes de alta calidad para un término.
        """
        raw_query = (search_term or "").strip()
        if not raw_query:
            return []

        if not save_dir:
            save_dir = utils.storage_dir("materials", create=True)
        else:
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[Contextual] Searching real production media for: {raw_query!r}")
        variations = self._generate_search_variations(raw_query)

        candidates: List[dict[str, Any]] = []

        # 1. MOTOR PRIMARIO: Yandex Images (extensa base de datos de cine, rodaje y prensa real)
        for v in variations[:3]:
            yandex_cand = self._search_yandex(v, max_results=max_results * 2)
            if yandex_cand:
                candidates.extend(yandex_cand)
                if len(candidates) >= max_results * 2:
                    break

        # 2. Si Yandex no tiene suficientes, intentar DuckDuckGo
        if len(candidates) < max_results:
            for v in variations[:2]:
                ddg_cand = self._search_duckduckgo(v, max_results=max_results)
                if ddg_cand:
                    candidates.extend(ddg_cand)
                    break

        # 3. Respaldo histórico: Wikimedia Commons
        if len(candidates) < max_results:
            for v in variations:
                wiki_cand = self._search_wikimedia(v, max_results=max_results)
                if wiki_cand:
                    candidates.extend(wiki_cand)
                    if len(candidates) >= max_results:
                        break

        # 4. Respaldo de hemeroteca: Internet Archive
        if len(candidates) < max_results:
            for v in variations:
                archive_cand = self._search_internet_archive(v, max_results=max_results)
                if archive_cand:
                    candidates.extend(archive_cand)
                    if len(candidates) >= max_results:
                        break

        if not candidates:
            logger.warning(f"[Contextual] No archival candidates found for: {raw_query!r}")
            return []

        # Descargar y validar candidatos
        materials: List[MaterialInfo] = []
        seen_hashes = set()

        for cand in candidates:
            image_url = cand.get("image_url")
            if not image_url:
                continue

            local_file, width, height = self._download_and_validate_image(
                image_url=image_url,
                save_dir=save_dir,
            )
            if not local_file:
                continue

            file_hash = hashlib.sha256(open(local_file, "rb").read()).hexdigest()
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)

            item = MaterialInfo()
            item.provider = "contextual"
            item.url = local_file
            item.duration = minimum_duration
            item.source_info = {
                "provider": "contextual",
                "source_engine": cand.get("source_engine", "web"),
                "search_term": raw_query,
                "title": cand.get("title", ""),
                "source_page": cand.get("source_page", ""),
                "original_url": image_url,
                "rendition": {
                    "width": width,
                    "height": height,
                },
            }
            materials.append(item)
            logger.success(
                f"[Contextual] Image validated ({width}x{height}) from {cand.get('source_engine')}: {local_file}"
            )

            if len(materials) >= max_results:
                break

        return materials

    def _search_yandex(self, query: str, max_results: int = 10) -> List[dict[str, Any]]:
        """Busca imágenes reales de prensa, rodaje y archivo en Yandex Images."""
        candidates: List[dict[str, Any]] = []
        url = f"https://yandex.com/images/search?text={urllib.parse.quote_plus(query)}"
        headers = {
            "User-Agent": _BROWSER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            r = requests.get(
                url,
                headers=headers,
                proxies=config.proxy,
                timeout=(8, 15),
            )
            if r.status_code != 200:
                logger.debug(f"[Contextual] Yandex returned HTTP {r.status_code}")
                return candidates

            # Extraer URLs de alta resolución de la respuesta
            raw_urls = re.findall(r'&quot;origUrl&quot;:&quot;(https?://[^&]+?)&quot;', r.text)
            if not raw_urls:
                raw_urls = re.findall(r'\"origUrl\":\"(https?://[^\"]+?)\"', r.text)

            for raw_u in raw_urls:
                img_url = urllib.parse.unquote(raw_u)
                if img_url and img_url.startswith(("http://", "https://")):
                    # Descartar avatares o logotipos
                    if any(bad in img_url.lower() for bad in ("avatar", "logo", "icon", "pixel.gif")):
                        continue
                    candidates.append({
                        "source_engine": "yandex_archive",
                        "image_url": img_url,
                        "title": query,
                        "source_page": url,
                    })
                    if len(candidates) >= max_results:
                        break

            if candidates:
                logger.info(f"[Contextual] Yandex Images found {len(candidates)} real photos for {query!r}")
        except Exception as exc:
            logger.debug(f"[Contextual] Yandex search error for {query!r}: {exc}")

        return candidates

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[dict[str, Any]]:
        """Busca imágenes en DuckDuckGo con manejo robusto de proxies y errores."""
        if not HAS_DDGS:
            return []

        proxy = None
        if config.proxy:
            proxy = config.proxy.get("https") or config.proxy.get("http")

        candidates: List[dict[str, Any]] = []
        try:
            with DDGS(proxy=proxy, timeout=10) as ddgs:
                results = list(
                    ddgs.images(
                        keywords=query,
                        region="wt-wt",
                        safesearch="off",
                        size="Large",
                        type_image="photo",
                        layout="all",
                        max_results=max_results,
                    )
                )
                for item in results:
                    img_url = item.get("image")
                    if img_url and img_url.startswith(("http://", "https://")):
                        candidates.append({
                            "source_engine": "duckduckgo",
                            "image_url": img_url,
                            "title": item.get("title", ""),
                            "source_page": item.get("url", ""),
                        })
                if candidates:
                    logger.info(f"[Contextual] DuckDuckGo found {len(candidates)} images for {query!r}")
        except Exception as exc:
            logger.debug(f"[Contextual] DuckDuckGo search: {type(exc).__name__}: {exc}")

        return candidates

    def _search_wikimedia(self, query: str, max_results: int = 5) -> List[dict[str, Any]]:
        """Busca imágenes históricas y de producción en Wikimedia Commons REST API."""
        candidates: List[dict[str, Any]] = []
        url = "https://commons.wikimedia.org/w/api.php"

        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(max_results * 2),
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "format": "json",
        }
        headers = {
            "User-Agent": _DOCUMENTARY_USER_AGENT,
            "Accept": "application/json",
        }

        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                proxies=config.proxy,
                timeout=(8, 15),
            )
            if r.status_code != 200:
                return candidates

            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                imageinfo = page.get("imageinfo", [])
                if not imageinfo:
                    continue
                info = imageinfo[0]
                mime = info.get("mime", "")
                if not mime.startswith("image/") or "svg" in mime:
                    continue

                width = int(info.get("width") or 0)
                height = int(info.get("height") or 0)
                if max(width, height) < 500:
                    continue

                img_url = info.get("url")
                if img_url:
                    candidates.append({
                        "source_engine": "wikimedia",
                        "image_url": img_url,
                        "title": page.get("title", ""),
                        "source_page": info.get("descriptionurl", ""),
                        "width": width,
                        "height": height,
                    })
                    if len(candidates) >= max_results:
                        break

            if candidates:
                logger.info(f"[Contextual] Wikimedia Commons found {len(candidates)} images for {query!r}")
        except Exception as exc:
            logger.debug(f"[Contextual] Wikimedia error for {query!r}: {exc}")

        return candidates

    def _search_internet_archive(self, query: str, max_results: int = 5) -> List[dict[str, Any]]:
        """Busca documentos y fotos de archivo en la API de Internet Archive."""
        candidates: List[dict[str, Any]] = []
        url = "https://archive.org/advancedsearch.php"

        params = {
            "q": f"({query}) AND mediatype:(image)",
            "fl[]": ["identifier", "title", "mediatype"],
            "rows": str(max_results),
            "output": "json",
        }
        headers = {
            "User-Agent": _DOCUMENTARY_USER_AGENT,
            "Accept": "application/json",
        }

        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                proxies=config.proxy,
                timeout=(8, 15),
            )
            if r.status_code != 200:
                return candidates

            data = r.json()
            docs = data.get("response", {}).get("docs", [])
            for doc in docs:
                ident = doc.get("identifier")
                if ident:
                    img_url = f"https://archive.org/services/img/{ident}"
                    candidates.append({
                        "source_engine": "internet_archive",
                        "image_url": img_url,
                        "title": doc.get("title", ident),
                        "source_page": f"https://archive.org/details/{ident}",
                    })
            if candidates:
                logger.info(f"[Contextual] Internet Archive found {len(candidates)} images for {query!r}")
        except Exception as exc:
            logger.debug(f"[Contextual] Internet Archive error for {query!r}: {exc}")

        return candidates

    def _download_and_validate_image(
        self,
        image_url: str,
        save_dir: str,
    ) -> tuple[Optional[str], int, int]:
        """
        Descarga una imagen, verifica validez bitmap y resolución.
        """
        try:
            headers = {
                "User-Agent": _BROWSER_USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": image_url,
            }
            r = requests.get(
                image_url,
                headers=headers,
                proxies=config.proxy,
                timeout=(8, 15),
                stream=True,
            )
            if r.status_code != 200:
                return None, 0, 0

            content = r.raw.read(30 * 1024 * 1024)
            if len(content) < 4096:
                return None, 0, 0

            img = Image.open(io.BytesIO(content))
            width, height = img.size

            if max(width, height) < 450 or min(width, height) < 200:
                return None, 0, 0

            # Normalizar a RGB
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (0, 0, 0))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
            file_name = f"contextual_{url_hash}.jpg"
            dest_path = os.path.join(save_dir, file_name)

            img.save(dest_path, "JPEG", quality=95, optimize=True)
            return dest_path, width, height

        except (UnidentifiedImageError, OSError):
            return None, 0, 0
        except Exception:
            return None, 0, 0

    def _extract_scene_search_queries(
        self,
        query: str,
        video_subject: str = "",
        video_script: str = "",
    ) -> List[str]:
        """
        Detecta el título cinematográfico y extrae consultas para escenas clave
        específicas mencionadas en el guion (ej. explosión del hospital, interrogatorio, etc.).
        """
        combined = f"{query} {video_subject}".strip()
        movie_title = ""
        m_year = re.search(
            r"((?:[A-Z][a-zA-Z0-9\x27:-]+\s+){1,4}(?:\(?19\d\d\)?|\(?20\d\d\)?))",
            combined,
        )
        if m_year:
            movie_title = m_year.group(1).strip()
            movie_title = re.sub(r"[()]", "", movie_title).strip()

        if not movie_title:
            clean_q = re.sub(
                r"\b(behind the scenes|vintage press photo|production still|archival document|"
                r"on set photo|raw footage frame|rare archival photo|documentary interview|"
                r"original poster|vintage photo|on set|newspaper clipping vintage|original storyboard|"
                r"directed by|director|film|movie|pelicula|película)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            movie_title = clean_q or query.strip()

        queries: List[str] = []
        script_lower = (video_script or "").lower()

        scene_triggers = [
            (r"hospital", "hospital explosion scene"),
            (r"interrogatorio|interrogation", "interrogation scene"),
            (r"carcel|cárcel|celda|aplauso", "jail cell scene"),
            (r"atraco|banco", "bank heist scene"),
            (r"bate\b|beisbol|béisbol", "baseball bat stairs scene"),
            (r"hacha\b|puerta", "heres johnny axe door scene"),
            (r"cama\b|levitacion|levitación|exorcismo", "bed shaking exorcism scene"),
            (r"vomito|vómito", "pea soup vomiting scene"),
            (r"escalera\b|escaleras\b|baile", "stairs dance scene"),
            (r"baño|espejo", "bathroom dance scene"),
            (r"trinity|nuclear|atomo|átomo|bomba", "trinity test explosion scene"),
            (r"oso\b|grizzly", "bear attack scene"),
            (r"nieve|caballo|congelad", "horse snow scene"),
            (r"batmovil|batmóvil|lluvia", "batmobile car chase scene"),
            (r"ducha|cuchillo", "shower scene 1960"),
            (r"disparo|balazo|accidente", "shooting scene movie clip"),
            (r"explosion|explosión", "explosion scene"),
            (r"barco\b|naufragio", "ship sinking scene"),
            (r"persecucion|persecución", "car chase scene"),
        ]

        for pattern, suffix in scene_triggers:
            if re.search(pattern, script_lower):
                q = f"{movie_title} {suffix}"
                if q not in queries:
                    queries.append(q)

        if len(queries) < 2:
            queries.append(f"{movie_title} iconic scene movie clip")
        queries.append(f"{movie_title} official trailer HD")

        return queries[:5]

    def fetch_trailer_clips(
        self,
        query: str,
        num_clips: int = 12,
        clip_duration: int = 3,
        video_aspect: VideoAspect = VideoAspect.portrait,
        save_dir: str = "",
        video_subject: str = "",
        video_script: str = "",
    ) -> List[MaterialInfo]:
        """
        Descarga automáticamente fragmentos de metraje real (escenas clave y tráiler)
        de la película o tema usando yt-dlp y los corta en clips dinámicos en 9:16.
        """
        if not query and not video_subject:
            return []
        if not save_dir:
            save_dir = utils.storage_dir("materials", create=True)
        else:
            os.makedirs(save_dir, exist_ok=True)

        scene_queries = self._extract_scene_search_queries(
            query=query,
            video_subject=video_subject,
            video_script=video_script,
        )
        logger.info(f"[Contextual] Target movie scenes to fetch: {scene_queries}")

        materials: List[MaterialInfo] = []
        seen_video_ids = set()

        try:
            aspect = VideoAspect(video_aspect)
        except Exception:
            aspect = VideoAspect.portrait
        target_w, target_h = aspect.to_resolution()
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        for s_query in scene_queries:
            if len(materials) >= num_clips:
                break
            try:
                logger.info(f"[Contextual] Searching real movie scene for: {s_query!r}")
                search_cmd = [
                    "yt-dlp",
                    "--no-playlist",
                    "--get-id",
                    f"ytsearch1:{s_query}",
                ]
                search_proc = subprocess.run(
                    search_cmd, capture_output=True, text=True, timeout=20
                )
                if search_proc.returncode != 0:
                    continue
                lines = [l.strip() for l in search_proc.stdout.strip().split("\n") if l.strip()]
                if not lines:
                    continue
                video_id = lines[0]
                if len(video_id) != 11 or video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)

                # Descargar sección de 30 segundos del momento de acción en formato rápido 720p (video-only)
                download_template = os.path.join(save_dir, f"scene_{video_id}.%(ext)s")
                download_cmd = [
                    "yt-dlp",
                    "-f", "bestvideo[height<=720]/best[height<=720]/best",
                    "--download-sections", "*00:15-00:45",
                    "--force-keyframes-at-cuts",
                    "--no-playlist",
                    "-o", download_template,
                    f"https://www.youtube.com/watch?v={video_id}",
                ]
                down_proc = subprocess.run(
                    download_cmd, capture_output=True, text=True, timeout=60
                )
                if down_proc.returncode != 0:
                    continue

                downloaded_raw = None
                for fname in os.listdir(save_dir):
                    if fname.startswith(f"scene_{video_id}."):
                        candidate_path = os.path.join(save_dir, fname)
                        if os.path.isfile(candidate_path) and os.path.getsize(candidate_path) > 10000:
                            downloaded_raw = candidate_path
                            break

                if not downloaded_raw:
                    continue

                # Extraer hasta 3 clips dinámicos de esta escena
                clips_to_cut = min(3, num_clips - len(materials))
                for cut_idx in range(clips_to_cut):
                    start_sec = cut_idx * (clip_duration + 3)
                    clip_unique_id = uuid4().hex[:6]
                    out_clip = os.path.join(
                        save_dir, f"scene_clip_{video_id}_{cut_idx}_{clip_unique_id}.mp4"
                    )
                    vf = (
                        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                        f"crop={target_w}:{target_h},"
                        f"vignette=PI/4,"
                        f"eq=contrast=1.06:saturation=1.06,"
                        f"format=yuv420p"
                    )
                    cut_cmd = [
                        ffmpeg_bin, "-y",
                        "-ss", str(start_sec),
                        "-i", downloaded_raw,
                        "-t", str(clip_duration),
                        "-vf", vf,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        out_clip,
                    ]
                    cut_proc = subprocess.run(
                        cut_cmd, capture_output=True, text=True, timeout=20
                    )
                    if (
                        cut_proc.returncode == 0
                        and os.path.isfile(out_clip)
                        and os.path.getsize(out_clip) > 5000
                    ):
                        item = MaterialInfo()
                        item.provider = "contextual"
                        item.url = out_clip
                        item.duration = clip_duration
                        item.source_info = {
                            "provider": "contextual",
                            "source_engine": "youtube_scene",
                            "search_term": s_query,
                            "title": f"Scene Clip ({s_query})",
                            "video_id": video_id,
                        }
                        materials.append(item)
                        logger.success(
                            f"[Contextual] Real movie scene clip extracted ({len(materials)}/{num_clips}): {out_clip} from '{s_query}'"
                        )

                # Limpiar bruto de esta escena
                try:
                    if os.path.exists(downloaded_raw):
                        os.remove(downloaded_raw)
                except OSError:
                    pass

            except Exception as e_scene:
                logger.warning(f"[Contextual] Failed processing scene {s_query!r}: {e_scene}")
                continue

        return materials
