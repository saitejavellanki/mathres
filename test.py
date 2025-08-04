#!/usr/bin/env python3
"""
Fixed QA extractor that handles all possible data formats
"""

import requests
import json
import re
from typing import List, Dict, Any

class FixedQAExtractor:
    """
    A robust QA extractor that handles various data formats
    """
    
    def __init__(self, base_url="https://transback.transpoze.ai"):
        self.base_url = base_url.rstrip('/')
    
    def fetch_result(self, result_id: int) -> Dict[str, Any]:
        """Fetch result data from API"""
        try:
            response = requests.get(f"{self.base_url}/results/?result_id={result_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching result {result_id}: {e}")
            return {}
    
    def extract_json_from_text(self, text: str) -> List[Dict]:
        """
        Extract JSON data from text using multiple methods
        """
        if not isinstance(text, str):
            return []
        
        # Method 1: Direct JSON parsing
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code blocks
        json_pattern = r'```json\s*\n(.*?)\n\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                continue
        
        # Method 3: Look for JSON arrays in the text
        array_pattern = r'\[\s*\{.*?\}\s*\]'
        array_matches = re.findall(array_pattern, text, re.DOTALL)
        
        for match in array_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
        
        # Method 4: Look for individual JSON objects
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        object_matches = re.findall(object_pattern, text)
        
        objects = []
        for match in object_matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        if objects:
            return objects
        
        return []
    
    def extract_qa_pairs(self, result_id: int) -> List[Dict]:
        """
        Extract QA pairs from a result
        """
        result_data = self.fetch_result(result_id)
        if not result_data:
            return []
        
        # Extract metadata
        metadata = {
            'result_id': result_data.get('result_id'),
            'script_id': result_data.get('script_id'),
            'student_name': result_data.get('student_name'),
            'student_roll_number': result_data.get('student_roll_number'),
            'subject_name': result_data.get('subject_name'),
            'class_name': result_data.get('class_name'),
            'created_at': result_data.get('created_at'),
            'updated_at': result_data.get('updated_at')
        }
        
        qa_pairs = []
        
        # Get restructuredtext
        restructuredtext = result_data.get('restructuredtext', {})
        qa_entries = restructuredtext.get('qa_pairs', [])
        
        print(f"Found {len(qa_entries)} QA entries in restructuredtext")
        
        for entry_idx, qa_entry in enumerate(qa_entries):
            print(f"\nProcessing QA entry {entry_idx + 1}...")
            
            if not isinstance(qa_entry, dict):
                print(f"  Skipping non-dict entry")
                continue
            
            # Check if there's an answer field
            if 'answer' not in qa_entry:
                print(f"  No 'answer' field found")
                continue
            
            answer_content = qa_entry['answer']
            print(f"  Answer type: {type(answer_content)}")
            print(f"  Answer preview: {str(answer_content)[:100]}...")
            
            # Extract JSON data from the answer
            json_data = self.extract_json_from_text(answer_content)
            print(f"  Extracted {len(json_data)} JSON objects")
            
            if json_data:
                # Process each JSON object as a QA pair
                for idx, item in enumerate(json_data):
                    if isinstance(item, dict):
                        qa_pair = {
                            **metadata,
                            'entry_index': entry_idx,
                            'qa_pair_index': idx,
                            'question': item.get('question', ''),
                            'answer': item.get('answer', ''),
                            'diagram_or_equation': item.get('diagram_or_equation', ''),
                            'original_question': qa_entry.get('question', '')
                        }
                        
                        # Only add if it has a question
                        if qa_pair['question'].strip():
                            qa_pairs.append(qa_pair)
                            print(f"    Added QA pair: {qa_pair['question'][:50]}...")
            else:
                # No JSON found, treat as plain text
                print(f"  No JSON found, treating as plain text")
                qa_pair = {
                    **metadata,
                    'entry_index': entry_idx,
                    'qa_pair_index': 0,
                    'question': qa_entry.get('question', 'Unknown'),
                    'answer': str(answer_content)[:1000],  # Limit length
                    'diagram_or_equation': '',
                    'original_question': qa_entry.get('question', '')
                }
                qa_pairs.append(qa_pair)
        
        print(f"\nTotal extracted QA pairs: {len(qa_pairs)}")
        return qa_pairs
    
    def display_qa_pairs(self, qa_pairs: List[Dict]):
        """Display QA pairs in a readable format"""
        if not qa_pairs:
            print("No QA pairs to display")
            return
        
        print(f"\n{'='*80}")
        print(f"EXTRACTED QA PAIRS ({len(qa_pairs)} total)")
        print(f"{'='*80}")
        
        # Show metadata
        if qa_pairs:
            first_pair = qa_pairs[0]
            print(f"Student: {first_pair['student_name']} (Roll: {first_pair['student_roll_number']})")
            print(f"Subject: {first_pair['subject_name']}")
            print(f"Class: {first_pair['class_name']}")
            print(f"Result ID: {first_pair['result_id']}")
        
        # Show each QA pair
        for i, pair in enumerate(qa_pairs):
            print(f"\n{'-'*60}")
            print(f"QA PAIR {i+1}")
            print(f"{'-'*60}")
            print(f"QUESTION: {pair['question']}")
            print(f"\nANSWER: {pair['answer'][:300]}{'...' if len(pair['answer']) > 300 else ''}")
            
            if pair['diagram_or_equation']:
                print(f"\nDIAGRAM/EQUATION: {pair['diagram_or_equation']}")
    
    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str):
        """Save QA pairs to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(qa_pairs)} QA pairs to {filename}")
        except Exception as e:
            print(f"❌ Error saving to {filename}: {e}")

def main():
    """Main function to test the extractor"""
    print("🚀 TESTING FIXED QA EXTRACTOR")
    print("="*80)
    
    extractor = FixedQAExtractor()
    
    # Test with result ID 75
    result_id = 75
    print(f"Extracting QA pairs from result ID {result_id}...")
    
    qa_pairs = extractor.extract_qa_pairs(result_id)
    
    # Display results
    extractor.display_qa_pairs(qa_pairs)
    
    # Save to file
    if qa_pairs:
        extractor.save_qa_pairs(qa_pairs, f'fixed_qa_pairs_result_{result_id}.json')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total QA pairs extracted: {len(qa_pairs)}")
    
    if qa_pairs:
        # Count by question type
        question_types = {}
        for pair in qa_pairs:
            question = pair['question'].lower()
            if 'prove' in question:
                key = 'Proof Questions'
            elif 'explain' in question:
                key = 'Explanation Questions'
            elif 'find' in question:
                key = 'Find Questions'
            else:
                key = 'Other Questions'
            
            question_types[key] = question_types.get(key, 0) + 1
        
        print(f"\nQuestion Types:")
        for qtype, count in question_types.items():
            print(f"  {qtype}: {count}")

if __name__ == "__main__":
    main()