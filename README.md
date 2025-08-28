# MCP ADE Server

A Model Context Protocol (MCP) server for LandingAI's Agentic Document Extraction (ADE), enabling AI assistants like Claude to extract structured data from of visually complex documents.

## Features

- **Document Processing**: Extract text and structured data from PDFs, images, and other document formats
- **Multiple Input Methods**: Process files from local paths, URLs, or base64-encoded data
- **Batch Processing**: Efficiently handle multiple documents simultaneously
- **Structured Extraction**: Use Pydantic models or JSON schemas to extract specific fields
- **Schema Validation**: Built-in validation for JSON schemas to ensure compatibility
- **Error Handling**: Robust error handling with automatic retries and detailed error messages
- **Logging**: Comprehensive logging for debugging and monitoring

## Installation

### Prerequisites

- Python 3.9-3.12
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- LandingAI Vision Agent API key ([Get one here](https://cloud.landing.ai/api-keys))

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/avaxia8/mcp-ade-server.git
cd mcp-ade-server
```

2. Install dependencies using uv:
```bash
uv venv
uv pip install -e .
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

3. Set up your environment:
```bash
cp .env.example .env
# Edit .env and add your VISION_AGENT_API_KEY
```

4. Configure Claude Desktop:

Add to your Claude Desktop configuration file:

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ade-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-ade-server",
        "run",
        "mcp-ade-server"
      ],
      "env": {
        "VISION_AGENT_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## Usage

Once configured, the ADE server provides the following tools in Claude:

### Basic Extraction

```python
# Extract raw text chunks from a document
ade_extract_from_path(path="/path/to/document.pdf")

# Extract from URL
ade_extract_from_url(url="https://example.com/document.pdf")

# Process multiple documents
ade_extract_batch(file_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"])
```

### Structured Extraction with Pydantic

```python
# Define a Pydantic model for extraction
pydantic_code = """
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    total_amount: float = Field(description="Total amount")
    due_date: str = Field(description="Payment due date")
"""

# Extract structured data
ade_extract_with_pydantic(pdf_base64=encoded_data, pydantic_model_code=pydantic_code)
```

### Structured Extraction with JSON Schema

```python
# Define extraction schema
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "due_date": {"type": "string", "format": "date"}
    }
}

# Validate schema first
ade_validate_json_schema(schema=schema)

# Extract data
ade_extract_with_json_schema(pdf_base64=encoded_data, schema=schema)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `ade_extract_raw_chunks` | Extract raw text chunks and metadata from base64-encoded documents |
| `ade_extract_from_path` | Extract from local file paths |
| `ade_extract_from_url` | Download and extract from URLs |
| `ade_extract_batch` | Process multiple documents in batch |
| `ade_extract_with_pydantic` | Extract structured data using Pydantic models |
| `ade_extract_with_json_schema` | Extract using JSON schemas |
| `ade_validate_json_schema` | Validate JSON schemas for compatibility |
| `ade_get_server_info` | Get server configuration and capabilities |

## Configuration

Environment variables (set in `.env` or system environment):

- `VISION_AGENT_API_KEY` (required): Your LandingAI API key
- `BATCH_SIZE` (optional, default: 10): Number of documents to process in parallel
- `MAX_WORKERS` (optional, default: 5): Maximum worker threads for processing

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `VISION_AGENT_API_KEY` is set in your `.env` file or environment
   - Check that the `.env` file is in the project root directory

2. **Import Errors**
   - Make sure all dependencies are installed: `uv pip install -e .`
   - Verify Python version compatibility (3.9-3.12)

3. **MCP Connection Issues**
   - Verify the path in Claude Desktop configuration is absolute
   - Restart Claude Desktop after configuration changes
   - Check logs for detailed error messages

4. **Document Processing Failures**
   - Ensure documents are in supported formats (PDF, PNG, JPG, TIFF, BMP, WEBP)
   - Check file permissions and accessibility
   - Verify URL accessibility for remote documents

### JSON Schema Guidelines

When using JSON schemas for extraction:

- Top-level type must be "object"
- Avoid prohibited keywords: `allOf`, `not`, `dependentRequired`, etc.
- Keep schema depth under 5 levels for optimal performance
- Arrays must include an "items" field
- Objects should have a "properties" field

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Project Structure

```
mcp-ade-server/
├── mcp_ade_server.py    # Main MCP server implementation
├── pyproject.toml        # Project configuration
├── .env.example          # Example environment configuration
├── README.md            # Documentation
├── LICENSE              # License file
├── examples/            # Usage examples
├── tests/              # Unit tests
└── docker/             # Docker configuration
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- Built on [LandingAI's Agentic Document Extraction](https://github.com/landing-ai/agentic-doc)
- Uses the [Model Context Protocol](https://modelcontextprotocol.io/)
- Designed for [Claude Desktop](https://claude.ai/download)

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/avaxia8/mcp-ade-server/issues)
- Check the [ADE documentation](https://github.com/landing-ai/agentic-doc)
- Visit [LandingAI support](https://landing.ai/support)

## Author

Created with ❤️ for the developer community to make document extraction accessible and reliable.
