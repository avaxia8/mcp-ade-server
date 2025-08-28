"""
Unit tests for MCP ADE Server
"""

import pytest
import json
import base64
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the agentic_doc imports before importing the server
with patch.dict('sys.modules', {
    'agentic_doc': MagicMock(),
    'agentic_doc.parse': MagicMock(),
    'agentic_doc.common': MagicMock(),
    'agentic_doc.config': MagicMock(),
}):
    import mcp_ade_server


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_validate_file_path_valid(self, tmp_path):
        """Test file path validation with valid file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        assert mcp_ade_server._validate_file_path(str(test_file)) is True
    
    def test_validate_file_path_invalid(self):
        """Test file path validation with invalid file."""
        assert mcp_ade_server._validate_file_path("/nonexistent/file.pdf") is False
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        assert mcp_ade_server._validate_url("https://example.com/file.pdf") is True
        assert mcp_ade_server._validate_url("http://example.org/doc") is True
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        assert mcp_ade_server._validate_url("not-a-url") is False
        assert mcp_ade_server._validate_url("file:///path/to/file") is False
        assert mcp_ade_server._validate_url("") is False


class TestFormatResponse:
    """Test response formatting."""
    
    def test_format_raw_response(self):
        """Test formatting of raw response."""
        # Create mock result
        mock_chunk = Mock()
        mock_chunk.chunk_type = Mock(value="text")
        mock_chunk.text = "Sample text"
        mock_chunk.chunk_id = "chunk-1"
        mock_chunk.grounding = []
        
        mock_result = Mock()
        mock_result.markdown = "# Sample Document"
        mock_result.chunks = [mock_chunk]
        mock_result.extraction = None
        mock_result.extraction_error = None
        
        response = mcp_ade_server._format_raw_response(mock_result)
        
        assert "markdown" in response
        assert response["markdown"] == "# Sample Document"
        assert "chunks" in response
        assert len(response["chunks"]) == 1
        assert response["chunks"][0]["content"] == "Sample text"
        assert "metadata" in response
        assert response["metadata"]["total_chunks"] == 1


class TestSchemaValidation:
    """Test JSON schema validation."""
    
    @pytest.mark.asyncio
    async def test_validate_valid_schema(self):
        """Test validation of valid schema."""
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        ctx = Mock()
        result = await mcp_ade_server.ade_validate_json_schema(ctx, valid_schema)
        result_data = json.loads(result)
        
        assert result_data["valid"] is True
        assert "valid" in result_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_invalid_schema(self):
        """Test validation of invalid schema."""
        invalid_schema = {
            "type": "array",  # Should be "object" at top level
            "items": {"type": "string"}
        }
        
        ctx = Mock()
        result = await mcp_ade_server.ade_validate_json_schema(ctx, invalid_schema)
        result_data = json.loads(result)
        
        assert result_data["valid"] is False
        assert len(result_data["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_schema_with_prohibited_keywords(self):
        """Test validation with prohibited keywords."""
        schema_with_prohibited = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"}
            },
            "allOf": [  # Prohibited keyword
                {"required": ["field1"]}
            ]
        }
        
        ctx = Mock()
        result = await mcp_ade_server.ade_validate_json_schema(ctx, schema_with_prohibited)
        result_data = json.loads(result)
        
        assert result_data["valid"] is False
        assert any("allOf" in error for error in result_data["errors"])


class TestServerInfo:
    """Test server info endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_server_info(self):
        """Test server info retrieval."""
        ctx = Mock()
        ctx.app_context = Mock(batch_size=10, max_workers=5)
        
        with patch.dict(os.environ, {"VISION_AGENT_API_KEY": "test_key"}):
            result = await mcp_ade_server.ade_get_server_info(ctx)
            info = json.loads(result)
            
            assert info["server"] == "ADE MCP Server"
            assert info["version"] == "1.0.0"
            assert "capabilities" in info
            assert "raw_extraction" in info["capabilities"]
            assert info["configuration"]["batch_size"] == 10
            assert info["configuration"]["max_workers"] == 5
            assert info["configuration"]["api_key_configured"] is True


class TestEnvironmentLoading:
    """Test environment variable loading."""
    
    def test_load_environment_with_key(self):
        """Test loading environment with API key."""
        with patch.dict(os.environ, {"VISION_AGENT_API_KEY": "test_key"}):
            # Should not raise an exception
            mcp_ade_server.load_environment_variables()
    
    def test_load_environment_without_key(self):
        """Test loading environment without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="VISION_AGENT_API_KEY"):
                mcp_ade_server.load_environment_variables()


class TestExtractionTools:
    """Test extraction tool endpoints."""
    
    @pytest.mark.asyncio
    async def test_extract_from_path_invalid_file(self):
        """Test extraction from invalid file path."""
        ctx = Mock()
        result = await mcp_ade_server.ade_extract_from_path(ctx, "/nonexistent/file.pdf")
        result_data = json.loads(result)
        
        assert "error" in result_data
        assert "Invalid or non-existent file path" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_extract_from_url_invalid_url(self):
        """Test extraction from invalid URL."""
        ctx = Mock()
        result = await mcp_ade_server.ade_extract_from_url(ctx, "not-a-url")
        result_data = json.loads(result)
        
        assert "error" in result_data
        assert "Invalid URL format" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_extract_batch_with_invalid_paths(self):
        """Test batch extraction with invalid paths."""
        ctx = Mock()
        file_paths = ["/valid/path.pdf", "/invalid/path.pdf"]
        
        with patch.object(mcp_ade_server, '_validate_file_path') as mock_validate:
            mock_validate.side_effect = [True, False]
            
            result = await mcp_ade_server.ade_extract_batch(ctx, file_paths)
            result_data = json.loads(result)
            
            assert "error" in result_data
            assert "Invalid file paths found" in result_data["error"]
            assert "/invalid/path.pdf" in result_data["invalid_paths"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])