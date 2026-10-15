import numpy as np
import pytest

from luchtfotos_kadaster.matching import Matcher, Matches, MatchRequest, _rotate_bound

cv2 = pytest.importorskip("cv2")


def _image(path, seed):
    cv2.imwrite(str(path), np.random.default_rng(seed).integers(0, 255, (60, 80), dtype=np.uint8))
    return path


def test_cache_is_keyed_on_content_and_rotation(tmp_path, monkeypatch):
    a, b = _image(tmp_path / "a.png", 0), _image(tmp_path / "b.png", 1)
    calls = []

    def fake(self, request):
        calls.append(request)
        return Matches(np.zeros((3, 2)), np.ones((3, 2)), np.full(3, 0.5))

    monkeypatch.setattr(Matcher, "_match", fake)
    matcher = Matcher(tmp_path / "cache")
    matcher.match_many([MatchRequest(a, b)])
    matcher.match_many([MatchRequest(a, b)])
    assert len(calls) == 1 and matcher.stats["cache_hits"] == 1
    matcher.match_many([MatchRequest(a, b, rotation_deg=90)])
    assert len(calls) == 2
    _image(b, 2)  # same path, new content
    (result,) = matcher.match_many([MatchRequest(a, b)])
    assert len(calls) == 3 and result.points1.shape == (3, 2)


def test_rotate_bound_round_trips_coordinates():
    image = np.zeros((60, 80), np.uint8)
    rotated, forward = _rotate_bound(image, 30.0)
    points = np.array([[0.0, 0.0], [79.0, 59.0], [40.0, 10.0]], np.float32)
    there = cv2.transform(points[None], forward)[0]
    # The canvas is rounded to whole pixels, so corners may sit just outside.
    assert (there > -1).all() and (there[:, 0] < rotated.shape[1] + 1).all() and (there[:, 1] < rotated.shape[0] + 1).all()
    back = cv2.transform(there[None], cv2.invertAffineTransform(forward))[0]
    assert np.allclose(back, points, atol=1e-3)
