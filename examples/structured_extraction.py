"""
Example: Structured Data Extraction

This example demonstrates how to extract structured data using Pydantic models
and JSON schemas with the ADE MCP server.
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field

# Example 1: Invoice extraction with Pydantic
class Invoice(BaseModel):
    """Invoice data model for extraction."""
    invoice_number: str = Field(description="Invoice or receipt number")
    vendor_name: str = Field(description="Name of the vendor or company")
    total_amount: float = Field(description="Total amount to be paid")
    due_date: Optional[str] = Field(description="Payment due date", default=None)
    line_items: Optional[List[str]] = Field(description="List of items purchased", default=None)

# Example 2: Resume extraction with Pydantic
class Resume(BaseModel):
    """Resume data model for extraction."""
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(description="Email address", default=None)
    phone: Optional[str] = Field(description="Phone number", default=None)
    education: List[str] = Field(description="Educational qualifications")
    experience: List[str] = Field(description="Work experience entries")
    skills: Optional[List[str]] = Field(description="Technical and soft skills", default=None)

# Example 3: JSON Schema for form extraction
FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "applicant_info": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "date_of_birth": {"type": "string", "format": "date"},
                "ssn": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip_code": {"type": "string"}
                    }
                }
            }
        },
        "employment_info": {
            "type": "object",
            "properties": {
                "employer": {"type": "string"},
                "position": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "salary": {"type": "number"}
            }
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relationship": {"type": "string"},
                    "phone": {"type": "string"}
                }
            }
        }
    }
}

def example_pydantic_extraction():
    """Example of using Pydantic models for extraction."""
    
    print("Pydantic Model Extraction Examples")
    print("=" * 50)
    
    # Invoice extraction
    invoice_code = '''
class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice or receipt number")
    vendor_name: str = Field(description="Name of the vendor or company")
    total_amount: float = Field(description="Total amount to be paid")
    due_date: Optional[str] = Field(description="Payment due date", default=None)
    line_items: Optional[List[str]] = Field(description="List of items purchased", default=None)
'''
    
    print("Invoice Extraction Model:")
    print(invoice_code)
    print()
    
    # In Claude, you would use:
    # result = await ade_extract_with_pydantic(
    #     pdf_base64=encoded_invoice,
    #     pydantic_model_code=invoice_code
    # )
    
    # Resume extraction
    resume_code = '''
class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(description="Email address", default=None)
    phone: Optional[str] = Field(description="Phone number", default=None)
    education: List[str] = Field(description="Educational qualifications")
    experience: List[str] = Field(description="Work experience entries")
    skills: Optional[List[str]] = Field(description="Technical and soft skills", default=None)
'''
    
    print("Resume Extraction Model:")
    print(resume_code)
    print()
    
    # In Claude, you would use:
    # result = await ade_extract_with_pydantic(
    #     pdf_base64=encoded_resume,
    #     pydantic_model_code=resume_code
    # )

def example_json_schema_extraction():
    """Example of using JSON schemas for extraction."""
    
    print("JSON Schema Extraction Example")
    print("=" * 50)
    
    # First validate the schema
    print("Validating schema...")
    
    # In Claude, you would use:
    # validation = await ade_validate_json_schema(schema=FORM_SCHEMA)
    
    print(f"Schema structure:\n{json.dumps(FORM_SCHEMA, indent=2)}")
    print()
    
    # Then extract data
    print("Extracting data with validated schema...")
    
    # In Claude, you would use:
    # result = await ade_extract_with_json_schema(
    #     pdf_base64=encoded_form,
    #     schema=FORM_SCHEMA
    # )

def example_custom_extraction():
    """Example of custom field extraction."""
    
    print("Custom Field Extraction Example")
    print("=" * 50)
    
    # Medical record extraction
    medical_record_code = '''
class MedicalRecord(BaseModel):
    patient_name: str = Field(description="Full name of the patient")
    date_of_birth: str = Field(description="Patient's date of birth")
    medical_record_number: str = Field(description="Unique medical record identifier")
    diagnoses: List[str] = Field(description="List of diagnoses")
    medications: List[Dict[str, str]] = Field(
        description="List of medications with name and dosage",
        default=[]
    )
    allergies: Optional[List[str]] = Field(
        description="Known allergies",
        default=None
    )
    vital_signs: Optional[Dict[str, Any]] = Field(
        description="Vital signs measurements",
        default=None
    )
'''
    
    print("Medical Record Extraction Model:")
    print(medical_record_code)
    print()
    
    # Contract extraction
    contract_schema = {
        "type": "object",
        "properties": {
            "contract_details": {
                "type": "object",
                "properties": {
                    "contract_number": {"type": "string"},
                    "effective_date": {"type": "string", "format": "date"},
                    "expiration_date": {"type": "string", "format": "date"},
                    "contract_value": {"type": "number"}
                }
            },
            "parties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "party_name": {"type": "string"},
                        "party_role": {"type": "string"},
                        "signature_date": {"type": "string", "format": "date"}
                    }
                }
            },
            "terms_and_conditions": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    print("Contract Extraction Schema:")
    print(json.dumps(contract_schema, indent=2))

if __name__ == "__main__":
    print("ADE Structured Extraction Examples")
    print("=" * 70)
    print()
    
    example_pydantic_extraction()
    print("\n" + "-" * 70 + "\n")
    
    example_json_schema_extraction()
    print("\n" + "-" * 70 + "\n")
    
    example_custom_extraction()