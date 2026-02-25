"""
Test rate limiting logic (unit test, no server).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rate_limit_disabled_by_default():
    """Rate limit is disabled when RATE_LIMIT_REQUESTS=0."""
    import worker_api
    assert worker_api.RATE_LIMIT_REQUESTS == 0
    err = worker_api._rate_limit_check("127.0.0.1")
    assert err is None


def test_rate_limit_enforcement():
    """When enabled, excess requests get blocked."""
    import worker_api
    orig = worker_api.RATE_LIMIT_REQUESTS
    worker_api.RATE_LIMIT_REQUESTS = 2
    worker_api.RATE_LIMIT_WINDOW_SEC = 10
    try:
        ip = "test_ip_12345"
        assert worker_api._rate_limit_check(ip) is None
        assert worker_api._rate_limit_check(ip) is None
        err = worker_api._rate_limit_check(ip)
        assert err is not None
        assert "rate" in err.lower() or "limit" in err.lower()
    finally:
        worker_api.RATE_LIMIT_REQUESTS = orig


if __name__ == "__main__":
    test_rate_limit_disabled_by_default()
    test_rate_limit_enforcement()
    print("test_rate_limit passed")
