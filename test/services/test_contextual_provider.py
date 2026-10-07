import os
import unittest
from unittest.mock import patch, MagicMock

from app.models.schema import MaterialInfo, VideoAspect
from app.services.contextual_provider import ContextualMediaProvider
from app.services import video, material, llm


class TestContextualMediaProvider(unittest.TestCase):
    def setUp(self):
        self.provider = ContextualMediaProvider(min_dimension=300)

    def test_generate_search_variations(self):
        query = "Stanley Kubrick directing Shelley Duvall The Shining 1980 behind the scenes"
        variations = self.provider._generate_search_variations(query)
        self.assertIn(query, variations)
        # Should have variations without suffixes
        self.assertTrue(any("behind the scenes" not in v for v in variations))
        # Should produce at least 2 variations
        self.assertGreaterEqual(len(variations), 2)

    def test_search_images_empty_query(self):
        results = self.provider.search_images("")
        self.assertEqual(results, [])

    def test_render_image_ken_burns_video(self):
        # Use existing test image
        test_img = "test/resources/1.png"
        self.assertTrue(os.path.exists(test_img), "Test image 1.png must exist")
        
        output_mp4 = video.render_image_ken_burns_video(
            image_path=test_img,
            clip_duration=2,
            video_aspect=VideoAspect.portrait,
            effect="pan_lr",
        )
        self.assertTrue(os.path.isfile(output_mp4))
        self.assertGreater(os.path.getsize(output_mp4), 1024)

    def test_render_image_zoom_video_alias(self):
        test_img = "test/resources/2.png"
        output_mp4 = video.render_image_zoom_video(test_img, clip_duration=2)
        self.assertTrue(os.path.isfile(output_mp4))
        self.assertGreater(os.path.getsize(output_mp4), 1024)

    def test_contextual_prompt_used_when_source_is_contextual(self):
        with patch.object(llm, "_generate_response", return_value='["Stanley Kubrick Shining 1980 set behind the scenes"]') as mock_gen:
            terms = llm.generate_terms(
                video_subject="The Shining",
                video_script="Stanley Kubrick on set",
                amount=1,
                video_source="contextual"
            )
            self.assertEqual(terms, ["Stanley Kubrick Shining 1980 set behind the scenes"])
            prompt_arg = mock_gen.call_args[1].get("prompt") or mock_gen.call_args[0][0]
            self.assertIn("Cinema & Archival Media Search Expert", prompt_arg)
            self.assertIn("behind the scenes", prompt_arg)


if __name__ == "__main__":
    unittest.main()
