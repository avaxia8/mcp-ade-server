"""
MCP Server for LandingAI's Agentic Document Extraction (ADE)

This server provides tools for extracting structured data from documents
using LandingAI's Vision Agent API.
"""

from typing import Any, AsyncIterator, Optional, Dict, List, Union
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv
import os
import json
import base64
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
import sys
import asyncio
import tempfile
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Import agentic-doc with stdout suppressed to prevent config output
@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Import with suppressed output
with suppress_output():
    from agentic_doc.parse import parse
    from agentic_doc.common import ParsedDocument
    from agentic_doc.config import ParseConfig

def _format_raw_response(result: ParsedDocument) -> Dict[str, Any]:
    """Helper function to format the raw chunk extraction response."""
    return {
        "markdown": result.markdown,
        "chunks": [
            {
                "type": chunk.chunk_type.value if hasattr(chunk, 'chunk_type') and hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                "content": chunk.text,
                "page": chunk.grounding[0].page if chunk.grounding else None,
                "chunk_id": chunk.chunk_id,
                "grounding": [
                    {
                        "bbox": {"l": g.box.l, "t": g.box.t, "r": g.box.r, "b": g.box.b}, 
                        "page": g.page
                    } for g in chunk.grounding
                ] if chunk.grounding else []
            } for chunk in result.chunks
        ],
        "metadata": {
            "total_chunks": len(result.chunks),
            "has_extraction": hasattr(result, 'extraction') and result.extraction is not None,
            "extraction_error": getattr(result, 'extraction_error', None)
        }
    }

def _validate_file_path(path: str) -> bool:
    """Validate that a file path exists and is readable."""
    try:
        path_obj = Path(path).resolve()
        return path_obj.exists() and path_obj.is_file()
    except Exception:
        return False

def _validate_url(url: str) -> bool:
    """Validate that a URL is well-formed."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

async def _download_url_to_temp(url: str) -> Optional[str]:
    """Download URL content to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            response = await asyncio.to_thread(urllib.request.urlopen, url)
            tmp_file.write(response.read())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Failed to download URL: {e}")
        return None

def load_environment_variables() -> None:
    """Loads environment variables from a .env file."""
    load_dotenv()
    if not os.getenv("VISION_AGENT_API_KEY"):
        logger.warning("VISION_AGENT_API_KEY not found in environment variables")
        raise ValueError(
            "Missing required environment variable: VISION_AGENT_API_KEY\n"
            "Please set it in your .env file or environment.\n"
            "Get your API key from: https://cloud.landing.ai/api-keys"
        )

@dataclass
class AppContext:
    """Application context for storing shared state."""
    batch_size: int = 10
    max_workers: int = 5

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Lifecycle manager for the MCP server."""
    batch_size = int(os.getenv("BATCH_SIZE", "10"))
    max_workers = int(os.getenv("MAX_WORKERS", "5"))
    
    logger.info(f"Initializing ADE MCP Server (batch_size={batch_size}, max_workers={max_workers})")
    yield AppContext(batch_size=batch_size, max_workers=max_workers)
    logger.info("Shutting down ADE MCP Server")

# Initialize the MCP server
mcp = FastMCP("ade-server", lifespan=app_lifespan)

@mcp.tool()
async def ade_extract_raw_chunks(ctx: Context, pdf_base64: str) -> str:
    """
    Performs basic extraction of all raw text chunks and their metadata from a document.
    
    Args:
        pdf_base64: Base64 encoded PDF or image document
        
    Returns:
        JSON string containing markdown text and detailed chunk information
    """
    try:
        with suppress_output():
            results = await asyncio.to_thread(parse, base64.b64decode(pdf_base64))
        
        if not results:
            return json.dumps({"error": "No results returned from parsing"})
        
        response = _format_raw_response(results[0])
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Raw extraction error: {e}")
        return json.dumps({"error": f"Extraction failed: {str(e)}"})

@mcp.tool()
async def ade_extract_from_path(ctx: Context, path: str) -> str:
    """
    Extracts raw chunks from a single local file path (PDF, image, etc.).
    
    Args:
        path: Local file path to the document
        
    Returns:
        JSON string containing extraction results
    """
    try:
        # Validate file path
        if not _validate_file_path(path):
            return json.dumps({"error": f"Invalid or non-existent file path: {path}"})
        
        with suppress_output():
            results = await asyncio.to_thread(parse, path)
        
        if not results:
            return json.dumps({"error": "No results returned from parsing"})
        
        result = results[0]
        response = {
            "file_path": path,
            "extraction_result": _format_raw_response(result)
        }
        return json.dumps(response, indent=2)
        
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except Exception as e:
        logger.error(f"File extraction error: {e}")
        return json.dumps({"error": f"Extraction failed: {str(e)}"})

@mcp.tool()
async def ade_extract_from_url(ctx: Context, url: str) -> str:
    """
    Downloads and extracts content from a URL.
    
    Args:
        url: URL of the document to process
        
    Returns:
        JSON string containing extraction results
    """
    try:
        # Validate URL
        if not _validate_url(url):
            return json.dumps({"error": f"Invalid URL format: {url}"})
        
        # Download to temporary file
        temp_path = await _download_url_to_temp(url)
        if not temp_path:
            return json.dumps({"error": f"Failed to download content from URL: {url}"})
        
        try:
            with suppress_output():
                results = await asyncio.to_thread(parse, temp_path)
            
            if not results:
                return json.dumps({"error": "No results returned from parsing"})
            
            result = results[0]
            response = {
                "source_url": url,
                "extraction_result": _format_raw_response(result)
            }
            return json.dumps(response, indent=2)
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"URL extraction error: {e}")
        return json.dumps({"error": f"URL extraction failed: {str(e)}"})

@mcp.tool()
async def ade_extract_batch(ctx: Context, file_paths: List[str]) -> str:
    """
    Process multiple documents in batch for improved performance.
    
    Args:
        file_paths: List of local file paths to process
        
    Returns:
        JSON string containing results for all documents
    """
    try:
        # Validate all paths
        invalid_paths = [p for p in file_paths if not _validate_file_path(p)]
        if invalid_paths:
            return json.dumps({
                "error": f"Invalid file paths found",
                "invalid_paths": invalid_paths
            })
        
        with suppress_output():
            results = await asyncio.to_thread(parse, file_paths)
        
        if not results:
            return json.dumps({"error": "No results returned from batch parsing"})
        
        response = {
            "total_documents": len(file_paths),
            "successful": len(results),
            "results": [
                {
                    "file_path": file_paths[i] if i < len(file_paths) else f"document_{i}",
                    "extraction": _format_raw_response(result)
                }
                for i, result in enumerate(results)
            ]
        }
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Batch extraction error: {e}")
        return json.dumps({"error": f"Batch extraction failed: {str(e)}"})

@mcp.tool()
async def ade_extract_with_pydantic(ctx: Context, pdf_base64: str, pydantic_model_code: str) -> str:
    """
    Extracts structured data using a Pydantic model defined in Python code.
    The last defined Pydantic BaseModel in the code will be used.
    
    Args:
        pdf_base64: Base64 encoded document
        pydantic_model_code: Python code defining a Pydantic model
        
    Returns:
        JSON string containing extracted structured data
    """
    try:
        # Prepare the code for execution with necessary imports
        full_code = f"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime

{pydantic_model_code}
"""
        
        local_scope = {}
        exec(full_code, globals(), local_scope)
        
        # Find the last defined Pydantic model in the executed code
        extraction_model = None
        for var in reversed(local_scope.values()):
            if isinstance(var, type) and issubclass(var, BaseModel) and var is not BaseModel:
                extraction_model = var
                break
        
        if not extraction_model:
            return json.dumps({"error": "No Pydantic BaseModel class found in the provided code"})

        config_obj = ParseConfig(extraction_model=extraction_model)
        
        with suppress_output():
            results = await asyncio.to_thread(
                parse, 
                base64.b64decode(pdf_base64), 
                config=config_obj
            )
        
        if not results:
            return json.dumps({"error": "No results returned from parsing"})
        
        result = results[0]
        response = {
            "extraction_error": result.extraction_error,
            "extracted_data": result.extraction.dict() if result.extraction else None,
            "field_details": {
                field: {
                    "confidence": meta.confidence,
                    "raw_text": meta.raw_text,
                    "chunk_references": meta.chunk_references
                } for field, meta in result.extraction_metadata.items() if meta
            } if result.extraction_metadata else {},
            "model_used": extraction_model.__name__
        }
        return json.dumps(response, indent=2, default=str)

    except SyntaxError as e:
        return json.dumps({"error": f"Invalid Python syntax in model code: {str(e)}"})
    except Exception as e:
        logger.error(f"Pydantic extraction error: {e}")
        return json.dumps({"error": f"Pydantic extraction failed: {str(e)}"})

@mcp.tool()
async def ade_validate_json_schema(ctx: Context, schema: Dict[str, Any]) -> str:
    """
    Validates a JSON schema against ADE's requirements and best practices.
    
    Args:
        schema: JSON schema to validate
        
    Returns:
        Validation result with any errors or warnings
    """
    errors = []
    warnings = []
    
    # Check top-level type
    if schema.get("type") != "object":
        errors.append("Top-level 'type' must be 'object'")

    # Prohibited keywords that may cause issues
    prohibited_keywords = {
        'allOf', 'not', 'dependentRequired', 
        'dependentSchemas', 'if', 'then', 'else'
    }

    def traverse(obj, path="root", depth=0):
        """Recursively validate schema structure."""
        if depth > 5:
            warnings.append(f"Schema depth exceeds 5 at path '{path}' - may impact performance")
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}"
                
                # Check for prohibited keywords
                if key in prohibited_keywords:
                    errors.append(f"Prohibited keyword '{key}' found at path '{new_path}'")
                
                # Check type arrays
                if key == "type" and isinstance(value, list):
                    if any(t in value for t in ["object", "array"]):
                        errors.append(
                            f"Type array at path '{new_path}' should not contain 'object' or 'array'. "
                            "Use 'anyOf' instead for complex types"
                        )
                
                # Check object requirements
                if obj.get("type") == "object" and "properties" not in obj:
                    warnings.append(f"Object at path '{path}' should have a 'properties' field")
                
                # Check array requirements
                if obj.get("type") == "array" and "items" not in obj:
                    errors.append(f"Array at path '{path}' must have an 'items' field")
                
                # Recursive traversal
                traverse(value, new_path, depth + 1)
                
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                traverse(item, f"{path}[{i}]", depth + 1)

    traverse(schema)
    
    # Prepare response
    if not errors and not warnings:
        return json.dumps({
            "valid": True,
            "message": "Schema is valid and follows best practices"
        })
    else:
        return json.dumps({
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "message": "Schema validation completed with issues" if errors else "Schema is valid with warnings"
        }, indent=2)

@mcp.tool()
async def ade_extract_with_json_schema(ctx: Context, pdf_base64: str, schema: Dict[str, Any]) -> str:
    """
    Extracts structured data from a document using a JSON schema.
    
    Args:
        pdf_base64: Base64 encoded document
        schema: JSON schema defining the structure to extract
        
    Returns:
        JSON string containing extracted structured data
    """
    try:
        # Validate schema first
        validation_result = await ade_validate_json_schema(ctx, schema)
        validation_data = json.loads(validation_result)
        
        if not validation_data.get("valid", False):
            return json.dumps({
                "error": "Schema validation failed",
                "validation_errors": validation_data.get("errors", [])
            })

        config_obj = ParseConfig(extraction_schema=schema)
        
        with suppress_output():
            results = await asyncio.to_thread(
                parse, 
                base64.b64decode(pdf_base64), 
                config=config_obj
            )
        
        if not results:
            return json.dumps({"error": "No results returned from parsing"})
        
        result = results[0]
        response = {
            "extraction_error": result.extraction_error,
            "extracted_data": result.extraction,
            "field_details": {
                field: {
                    "confidence": meta.confidence,
                    "raw_text": meta.raw_text,
                    "chunk_references": meta.chunk_references
                } for field, meta in result.extraction_metadata.items() if meta
            } if result.extraction_metadata else {},
            "schema_validation": validation_data.get("warnings", [])
        }
        return json.dumps(response, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"JSON schema extraction error: {e}")
        return json.dumps({"error": f"JSON schema extraction failed: {str(e)}"})

@mcp.tool()
async def ade_get_server_info(ctx: Context) -> str:
    """
    Returns information about the ADE MCP server configuration and capabilities.
    
    Returns:
        JSON string containing server information
    """
    info = {
        "server": "ADE MCP Server",
        "version": "1.0.0",
        "capabilities": [
            "raw_extraction",
            "path_extraction",
            "url_extraction",
            "batch_processing",
            "pydantic_models",
            "json_schema",
            "schema_validation"
        ],
        "configuration": {
            "batch_size": ctx.app_context.batch_size,
            "max_workers": ctx.app_context.max_workers,
            "api_key_configured": bool(os.getenv("VISION_AGENT_API_KEY"))
        },
        "supported_formats": [
            "PDF",
            "PNG",
            "JPG/JPEG",
            "TIFF",
            "BMP",
            "WEBP"
        ]
    }
    return json.dumps(info, indent=2)

def main():
    """Main entry point for the MCP ADE server."""
    try:
        load_environment_variables()
        logger.info("Starting ADE MCP Server...")
        mcp.run(transport='stdio')
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()