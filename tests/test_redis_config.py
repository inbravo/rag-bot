"""
Unit tests for Redis and session configuration in AppConfig
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAppConfigRedis(unittest.TestCase):
    """Test Redis and session configuration in AppConfig"""

    @patch.dict(os.environ, {
        'REDIS_HOST': 'testhost',
        'REDIS_PORT': '6380',
        'REDIS_DB': '1',
        'REDIS_PASSWORD': 'testpass',
        'CONVERSATION_TTL_SECONDS': '3600',
        'SESSION_LIFETIME_DAYS': '14',
        'FLASK_SECRET_KEY': 'test-secret-key'
    })
    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_redis_config_from_env(self, mock_retriever, mock_llm):
        """Test that Redis configuration is loaded from environment variables"""
        # Import after mocking to get fresh config
        from AppConfig import AppConfig
        
        self.assertEqual(AppConfig.REDIS_HOST, 'testhost')
        self.assertEqual(AppConfig.REDIS_PORT, 6380)
        self.assertEqual(AppConfig.REDIS_DB, 1)
        self.assertEqual(AppConfig.REDIS_PASSWORD, 'testpass')
        self.assertEqual(AppConfig.CONVERSATION_TTL_SECONDS, 3600)
        self.assertEqual(AppConfig.SESSION_LIFETIME_DAYS, 14)
        self.assertEqual(AppConfig.FLASK_SECRET_KEY, 'test-secret-key')

    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_redis_config_defaults(self, mock_retriever, mock_llm):
        """Test that Redis configuration has safe defaults"""
        from AppConfig import AppConfig
        
        # Check defaults are present
        self.assertIsNotNone(AppConfig.REDIS_HOST)
        self.assertIsNotNone(AppConfig.REDIS_PORT)
        self.assertIsNotNone(AppConfig.REDIS_DB)
        self.assertIsNotNone(AppConfig.CONVERSATION_TTL_SECONDS)
        self.assertIsNotNone(AppConfig.SESSION_LIFETIME_DAYS)
        self.assertIsNotNone(AppConfig.FLASK_SECRET_KEY)

    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_num_relevant_docs_default(self, mock_retriever, mock_llm):
        """Test that NUM_RELEVANT_DOCS has a safe default"""
        from AppConfig import AppConfig
        
        # Should not raise an error even if env var is not set
        self.assertIsInstance(AppConfig.NUM_RELEVANT_DOCS, int)
        # Default should be 5
        if os.getenv('NUM_RELEVANT_DOCS') is None:
            self.assertEqual(AppConfig.NUM_RELEVANT_DOCS, 5)

    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_conv_key_method(self, mock_retriever, mock_llm):
        """Test conv_key static method"""
        from AppConfig import AppConfig
        
        sid = "test-session-123"
        key = AppConfig.conv_key(sid)
        self.assertEqual(key, f"conv:{sid}")

    @patch('AppConfig.redis.Redis')
    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_redis_client_method(self, mock_retriever, mock_llm, mock_redis_cls):
        """Test redis_client classmethod"""
        from AppConfig import AppConfig
        
        # Mock Redis client
        mock_client = Mock()
        mock_redis_cls.return_value = mock_client
        
        client = AppConfig.redis_client()
        
        # Verify Redis was called with correct parameters
        if client is not None:
            mock_redis_cls.assert_called_once()
            call_kwargs = mock_redis_cls.call_args[1]
            self.assertEqual(call_kwargs['host'], AppConfig.REDIS_HOST)
            self.assertEqual(call_kwargs['port'], AppConfig.REDIS_PORT)
            self.assertEqual(call_kwargs['db'], AppConfig.REDIS_DB)
            self.assertTrue(call_kwargs['decode_responses'])

    @patch('AppConfig.LLMFactory')
    @patch('AppConfig.RAGRetriever')
    def test_apply_session_config_method(self, mock_retriever, mock_llm):
        """Test apply_session_config classmethod"""
        from AppConfig import AppConfig
        
        # Mock Flask app
        mock_app = Mock()
        mock_app.config = {}
        
        # Mock Redis client
        mock_redis = Mock()
        
        # Apply session config
        AppConfig.apply_session_config(mock_app, mock_redis)
        
        # Verify secret key was set
        self.assertEqual(mock_app.secret_key, AppConfig.FLASK_SECRET_KEY)


if __name__ == '__main__':
    unittest.main()
