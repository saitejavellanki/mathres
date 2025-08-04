from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
import re


class JSONStructureInput(BaseModel):
    """Input schema for JSON Structure Validator Tool."""
    raw_output: str = Field(..., description="The raw output from the agent that needs to be converted to structured JSON")
    output_type: str = Field(..., description="Type of output expected: 'restructure' or 'marking'")


class JSONStructureValidatorTool(BaseTool):
    name: str = "JSON Structure Validator"
    description: str = (
        "Validates and converts agent outputs into properly structured JSON format. "
        "Use this tool to ensure the final output is valid JSON that matches the expected schema. "
        "Supports both 'restructure' (question-answer pairs) and 'marking' (marks allocation) output types."
    )
    args_schema: Type[BaseModel] = JSONStructureInput

    def _run(self, raw_output: str, output_type: str) -> str:
        """
        Converts raw agent output to structured JSON format.
        
        Args:
            raw_output: The raw text output from the agent
            output_type: Either 'restructure' or 'marking'
            
        Returns:
            Valid JSON string in the expected format
        """
        try:
            # Clean the raw output
            cleaned_output = self._clean_raw_output(raw_output)
            
            if output_type.lower() == 'restructure':
                return self._format_restructure_output(cleaned_output)
            elif output_type.lower() == 'marking':
                return self._format_marking_output(cleaned_output)
            else:
                raise ValueError(f"Unknown output_type: {output_type}")
                
        except Exception as e:
            # Fallback: return empty structure if parsing fails
            if output_type.lower() == 'restructure':
                return json.dumps([{
                    "question": "Error in processing",
                    "answer": f"Processing error: {str(e)}",
                    "diagram_or_equation": ""
                }], indent=2)
            else:
                return json.dumps([{
                    "question": "Error in processing",
                    "marks_awarded": 0
                }], indent=2)

    def _clean_raw_output(self, raw_output: str) -> str:
        """Clean and extract JSON from raw output."""
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Remove any text before the first [ or {
        json_start = max(cleaned.find('['), cleaned.find('{'))
        if json_start != -1:
            cleaned = cleaned[json_start:]
        
        # Remove any text after the last ] or }
        json_end_bracket = cleaned.rfind(']')
        json_end_brace = cleaned.rfind('}')
        json_end = max(json_end_bracket, json_end_brace)
        if json_end != -1:
            cleaned = cleaned[:json_end + 1]
        
        return cleaned.strip()

    def _format_restructure_output(self, cleaned_output: str) -> str:
        """Format output for restructure task."""
        try:
            # Try to parse as JSON first
            data = json.loads(cleaned_output)
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            # Validate and ensure correct structure
            formatted_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                formatted_item = {
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "diagram_or_equation": str(item.get("diagram_or_equation", ""))
                }
                formatted_data.append(formatted_item)
            
            return json.dumps(formatted_data, indent=2, ensure_ascii=False)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract structure from text
            return self._extract_restructure_from_text(cleaned_output)

    def _format_marking_output(self, cleaned_output: str) -> str:
        """Format output for marking task."""
        try:
            # Try to parse as JSON first
            data = json.loads(cleaned_output)
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            # Validate and ensure correct structure
            formatted_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Handle different possible key names for marks
                marks_value = 0
                for key in ['marks_awarded', 'marks', 'score', 'points']:
                    if key in item:
                        try:
                            marks_value = int(item[key])
                            break
                        except (ValueError, TypeError):
                            marks_value = 0
                
                formatted_item = {
                    "question": str(item.get("question", "")),
                    "marks_awarded": marks_value
                }
                formatted_data.append(formatted_item)
            
            return json.dumps(formatted_data, indent=2, ensure_ascii=False)
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing failed: {e}")  # Debug print
            # If JSON parsing fails, try to extract structure from text
            return self._extract_marking_from_text(cleaned_output)

    def _extract_restructure_from_text(self, text: str) -> str:
        """Extract restructure format from plain text when JSON parsing fails."""
        # This is a fallback method to extract question-answer pairs from text
        lines = text.split('\n')
        result = []
        current_item = {"question": "", "answer": "", "diagram_or_equation": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('"question"') or line.lower().startswith('question'):
                if current_item["question"]:
                    result.append(current_item.copy())
                    current_item = {"question": "", "answer": "", "diagram_or_equation": ""}
                current_item["question"] = self._extract_value(line)
            elif line.startswith('"answer"') or line.lower().startswith('answer'):
                current_item["answer"] = self._extract_value(line)
            elif line.startswith('"diagram_or_equation"') or 'diagram' in line.lower():
                current_item["diagram_or_equation"] = self._extract_value(line)
        
        if current_item["question"]:
            result.append(current_item)
        
        return json.dumps(result, indent=2, ensure_ascii=False)

    def _extract_marking_from_text(self, text: str) -> str:
        """Extract marking format from plain text when JSON parsing fails."""
        lines = text.split('\n')
        result = []
        current_item = {"question": "", "marks_awarded": 0}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('"question"') or line.lower().startswith('question'):
                if current_item["question"]:
                    result.append(current_item.copy())
                    current_item = {"question": "", "marks_awarded": 0}
                current_item["question"] = self._extract_value(line)
            elif any(keyword in line.lower() for keyword in ['marks_awarded', 'marks', 'score', 'points']):
                try:
                    current_item["marks_awarded"] = int(self._extract_numeric_value(line))
                except ValueError:
                    current_item["marks_awarded"] = 0
        
        if current_item["question"]:
            result.append(current_item)
        
        return json.dumps(result, indent=2, ensure_ascii=False)

    def _extract_value(self, line: str) -> str:
        """Extract value from a key-value line."""
        if ':' in line:
            return line.split(':', 1)[1].strip().strip('"').strip("'").rstrip(',')
        return line.strip().strip('"').strip("'").rstrip(',')

    def _extract_numeric_value(self, line: str) -> str:
        """Extract numeric value from a line."""
        import re
        numbers = re.findall(r'\d+', line)
        return numbers[0] if numbers else "0"