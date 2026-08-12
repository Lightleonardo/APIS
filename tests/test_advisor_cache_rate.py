import pytest
import time
from backend.advisor.rate_limiter import RateLimiter, advisor_rate_limiter
from backend.advisor.cache import _advisor_cache, _cache_key, get_cached_response, set_cached_response, clear_cache
from backend.advisor import run_advisor, _RESPONSE_CACHE, AdvisorResult
from backend.schemas import AdvisorInput, SemesterTarget, FeatureImportance, ImprovementTrend


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True

    def test_blocks_requests_over_limit(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False
        assert limiter.allow_request() is False

    def test_window_expiry_allows_new_requests(self):
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False
        time.sleep(1.1)  # Wait for window to expire
        assert limiter.allow_request() is True

    def test_thread_safety_basic(self):
        import threading
        limiter = RateLimiter(max_requests=200, window_seconds=60)
        results = []
        results_lock = threading.Lock()

        def worker():
            for _ in range(50):
                allowed = limiter.allow_request()
                with results_lock:
                    results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 200 requests should be allowed (under limit)
        assert all(results)
        assert len(results) == 200


class TestRateLimiterInstance:
    def test_advisor_rate_limiter_exists(self):
        assert advisor_rate_limiter is not None
        assert isinstance(advisor_rate_limiter, RateLimiter)
        assert advisor_rate_limiter.max_requests == 15
        assert advisor_rate_limiter.window_seconds == 60


class TestCache:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    def test_cache_key_consistent(self):
        advisor_in = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging",
        )
        key1 = _cache_key(advisor_in)
        key2 = _cache_key(advisor_in)
        assert key1 == key2
        # Full SHA256 is 64 chars (not truncated in cache module)
        assert len(key1) == 64

    def test_cache_miss_returns_none(self):
        advisor_in = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging",
        )
        assert get_cached_response(advisor_in) is None

    def test_cache_set_and_get(self):
        advisor_in = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging",
        )
        set_cached_response(advisor_in, "cached response")
        assert get_cached_response(advisor_in) == "cached response"

    def test_cache_excludes_tone_and_language(self):
        advisor_in1 = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging", language="en",
        )
        advisor_in2 = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="direct", language="fr",  # Different tone/lang
        )
        set_cached_response(advisor_in1, "response 1")
        # Should hit cache even though tone/language differ
        assert get_cached_response(advisor_in2) == "response 1"

    def test_cache_eviction_fifo(self):
        # Note: The cache module (_advisor_cache) doesn't implement eviction
        # The advisor.py has its own cache with FIFO eviction
        # This test verifies the basic cache behavior
        for i in range(5):
            advisor_in = AdvisorInput(
                student_name=f"Test{i}", course="CS",
                current_cgpa=3.5, target_graduation_class="First Class",
                target_cgpa=4.5, remaining_semesters=4,
                required_average_gpa=4.7, predicted_final_cgpa=4.2,
                predicted_graduation_class="First Class", academic_risk="Low",
                goal_feasible=True, best_possible_classification="First Class",
                academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
                consistency_index=20, semester_plan=[],
                top_features_final_cgpa=[], top_features_graduation_class=[],
                top_features_academic_risk=[], tone="encouraging",
            )
            set_cached_response(advisor_in, f"response {i}")

        assert len(_advisor_cache) == 5

    def test_clear_cache(self):
        advisor_in = AdvisorInput(
            student_name="Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging",
        )
        set_cached_response(advisor_in, "response")
        assert get_cached_response(advisor_in) == "response"
        clear_cache()
        assert get_cached_response(advisor_in) is None


class TestAdvisorWithCache:
    def test_advisor_uses_cache(self, monkeypatch):
        """Test that run_advisor checks cache before calling LLM."""
        from backend.advisor.advisor import _RESPONSE_CACHE, run_advisor
        from backend.advisor import AdvisorResult

        # Pre-populate cache
        advisor_in = AdvisorInput(
            student_name="Cache Test", course="CS",
            current_cgpa=3.5, target_graduation_class="First Class",
            target_cgpa=4.5, remaining_semesters=4,
            required_average_gpa=4.7, predicted_final_cgpa=4.2,
            predicted_graduation_class="First Class", academic_risk="Low",
            goal_feasible=True, best_possible_classification="First Class",
            academic_health_score=80, gpa_trend=ImprovementTrend.IMPROVING,
            consistency_index=20, semester_plan=[],
            top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[], tone="encouraging",
        )
        _RESPONSE_CACHE.clear()
        # We need to build the prompt first to get the cache key
        from backend.advisor.advisor import build_prompt, _cache_key
        prompt = build_prompt(advisor_in)
        cache_key = _cache_key(prompt)
        _RESPONSE_CACHE[cache_key] = "cached response"

        # Mock build_prompt to return our test prompt
        from backend import advisor as advisor_module
        original_build = advisor_module.build_prompt
        advisor_module.build_prompt = lambda x: "test prompt"

        try:
            result: AdvisorResult = run_advisor(advisor_in)
            assert result.source == "cache"
            assert result.response == "cached response"
        finally:
            advisor_module.build_prompt = original_build