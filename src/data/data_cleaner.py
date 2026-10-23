import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class UCSBDataCleaner:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "cleaned"
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'cleaning_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Content quality thresholds
        self.min_content_length = 50  # Minimum characters for meaningful content
        self.max_content_length = 50000  # Maximum to avoid extremely long entries
        
        # Patterns to identify gibberish or unwanted content
        self.gibberish_patterns = [
            r'^[^a-zA-Z0-9\s]{10,}$',  # Only special characters
            r'^[\s\n\r]+$',  # Only whitespace
            r'^[0-9\s\-\.\,]+$',  # Only numbers and basic punctuation
            r'^(null|undefined|None|NaN)$',  # Null values
            r'^[^\w\s]*$',  # No word characters
        ]
        
        # Common unwanted phrases/content
        self.unwanted_phrases = [
            'javascript is required',
            'please enable javascript',
            'this page requires javascript',
            'loading...',
            'please wait',
            'error loading',
            'page not found',
            'access denied',
            'unauthorized',
            'captcha',
            'security check',
            'bot detection',
            'skip to main content',
            'academic calendar',
            'schedule of classes',
            'previous catalogs',
            'search . . .',
            'university of california santa barbara',
            'courses programs departments',
            'academics and policies',
            'introduction to our campus',
            'download as pdf',
            'menu toggle',
            'college of engineering',
            'engineering sciences'
        ]
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'cleaned_departments': 0,
            'cleaned_programs': 0,
            'cleaned_courses': 0,
            'removed_empty': 0,
            'removed_gibberish': 0,
            'removed_short': 0,
            'removed_duplicates': 0
        }

    def load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load and parse JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            self.logger.warning(f"Failed to load {file_path}: {e}")
            return None

    def is_gibberish(self, text: str) -> bool:
        """Check if text appears to be gibberish or unwanted content"""
        if not text or not isinstance(text, str):
            return True
            
        text_lower = text.lower().strip()
        
        # Check for unwanted phrases
        for phrase in self.unwanted_phrases:
            if phrase in text_lower:
                return True
                
        # Check for gibberish patterns
        for pattern in self.gibberish_patterns:
            if re.match(pattern, text_lower):
                return True
                
        # Check for reasonable content length
        if len(text.strip()) < self.min_content_length:
            return True
            
        # Check for reasonable word count
        words = text.split()
        if len(words) < 5:  
            return True
            
        # Check for reasonable character variety
        unique_chars = set(text.lower())
        if len(unique_chars) < 10:  
            return True
            
        return False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text or not isinstance(text, str):
            return ""
            
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove UCSB-specific navigation and header elements
        text = re.sub(r'Skip to Main Content.*?University of California Santa Barbara', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Academic Calendar.*?Previous Catalogs', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Search \. \. \.*?University of California Santa Barbara', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Courses.*?Departments.*?Academics and Policies', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Introduction to Our Campus.*?Student Life', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'Home\s*/\s*Courses\s*/\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Download as PDF', '', text, flags=re.IGNORECASE)
        
        # Remove navigation breadcrumbs and repeated headers
        text = re.sub(r'University of California Santa Barbara\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'College of Engineering\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Engineering Sciences\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'General\s*Full Course Title', 'Course Title:', text, flags=re.IGNORECASE)
        text = re.sub(r'Instructor Name\(s\)', 'Instructor:', text, flags=re.IGNORECASE)
        text = re.sub(r'Course Description', 'Description:', text, flags=re.IGNORECASE)
        
        # Remove common web artifacts
        text = re.sub(r'Skip to main content', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Menu\s*Toggle', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Search\s*Submit', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Print\s*this\s*page', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1{2,}', r'\1', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s*([.!?:;,])\s*', r'\1 ', text)
        
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def clean_department_data(self, dept_data: Dict) -> Optional[Dict]:
        """Clean individual department data"""
        if not dept_data or not isinstance(dept_data, dict):
            return None
            
        cleaned = {
            'name': dept_data.get('name', '').strip(),
            'code': dept_data.get('code', '').strip(),
            'url': dept_data.get('url', '').strip(),
            'overview': self.clean_text(dept_data.get('overview', '')),
            'programs': [],
            'courses': []
        }
        
        # Filter out if essential fields are missing or gibberish
        if not cleaned['name'] or not cleaned['code']:
            self.stats['removed_empty'] += 1
            return None
            
        if self.is_gibberish(cleaned['overview']):
            self.logger.warning(f"Removing department {cleaned['name']} due to gibberish overview")
            self.stats['removed_gibberish'] += 1
            return None
            
        # Clean programs
        if 'programs' in dept_data and isinstance(dept_data['programs'], list):
            for program in dept_data['programs']:
                cleaned_program = self.clean_program_data(program)
                if cleaned_program:
                    cleaned['programs'].append(cleaned_program)
                    
        # Clean courses
        if 'courses' in dept_data and isinstance(dept_data['courses'], list):
            for course in dept_data['courses']:
                cleaned_course = self.clean_course_data(course)
                if cleaned_course:
                    cleaned['courses'].append(cleaned_course)
                    
        return cleaned

    def clean_program_data(self, program_data: Dict) -> Optional[Dict]:
        """Clean individual program data"""
        if not program_data or not isinstance(program_data, dict):
            return None
            
        cleaned = {
            'name': program_data.get('name', '').strip(),
            'code': program_data.get('code', '').strip(),
            'url': program_data.get('url', '').strip(),
            'title': self.clean_text(program_data.get('title', '')),
            'content': self.clean_text(program_data.get('content', '')),
            'description': self.clean_text(program_data.get('description', '')),
            'requirements': self.clean_text(program_data.get('requirements', ''))
        }
        
        # Filter out if essential fields are missing
        if not cleaned['name'] or not cleaned['code']:
            self.stats['removed_empty'] += 1
            return None
            
        # Check if main content is gibberish
        main_content = cleaned['content'] or cleaned['description']
        if self.is_gibberish(main_content):
            self.logger.warning(f"Removing program {cleaned['name']} due to gibberish content")
            self.stats['removed_gibberish'] += 1
            return None
            
        return cleaned

    def clean_course_data(self, course_data: Dict) -> Optional[Dict]:
        """Clean individual course data"""
        if not course_data or not isinstance(course_data, dict):
            return None
            
        # Clean the content first
        raw_content = course_data.get('content', '')
        cleaned_content = self.clean_text(raw_content)
        
        # Extract structured information from cleaned content
        description = self.extract_course_description(cleaned_content)
        
        cleaned = {
            'name': course_data.get('name', '').strip(),
            'code': course_data.get('code', '').strip(),
            'url': course_data.get('url', '').strip(),
            'title': self.extract_course_title(cleaned_content, course_data.get('title', '')),
            'content': cleaned_content,
            'description': description,
            'prerequisites': self.extract_prerequisites(cleaned_content),
            'units': self.extract_units(cleaned_content),
            'instructor': self.extract_instructor(cleaned_content)
        }
        
        # Filter out if essential fields are missing
        if not cleaned['name'] or not cleaned['code']:
            self.stats['removed_empty'] += 1
            return None
            
        # Check if main content is gibberish
        main_content = cleaned['description'] or cleaned['content']
        if self.is_gibberish(main_content):
            self.logger.warning(f"Removing course {cleaned['name']} due to gibberish content")
            self.stats['removed_gibberish'] += 1
            return None
            
        return cleaned
    
    def extract_course_title(self, content: str, fallback_title: str) -> str:
        """Extract course title from content"""
        # Look for "Course Title:" or "Full Course Title"
        title_match = re.search(r'Course Title:\s*(.+?)(?:Instructor:|Description:|$)', content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try to extract from the fallback title if it's not just "University of California Santa Barbara"
        if fallback_title and "University of California Santa Barbara" not in fallback_title:
            return fallback_title.strip()
        
        # Look for course name pattern (e.g., "ENGR W 3 - Introduction to Programming")
        course_match = re.search(r'[A-Z]+\s*\w*\s*\d+\s*-\s*(.+?)(?:\s|$)', content)
        if course_match:
            return course_match.group(1).strip()
        
        return ""
    
    def extract_course_description(self, content: str) -> str:
        """Extract course description from content"""
        # Look for "Description:" or "Course Description"
        desc_match = re.search(r'Description:\s*(.+?)(?:Prerequisites:|Units:|Instructor:|$)', content, re.IGNORECASE | re.DOTALL)
        if desc_match:
            return desc_match.group(1).strip()
        
        # If no explicit description marker, try to find descriptive text after course info
        # Look for text that starts with a capital letter and has reasonable length
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 20 and line[0].isupper() and not any(marker in line.lower() for marker in ['course title', 'instructor', 'prerequisite', 'unit']):
                # This might be the description
                desc_lines = [line]
                # Get following lines that seem to be part of the description
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not any(marker in next_line.lower() for marker in ['course title', 'instructor', 'prerequisite', 'unit']):
                        desc_lines.append(next_line)
                    else:
                        break
                return ' '.join(desc_lines)
        
        return ""
    
    def extract_prerequisites(self, content: str) -> str:
        """Extract prerequisites from content"""
        prereq_match = re.search(r'Prerequisites?:\s*(.+?)(?:Units:|Instructor:|$)', content, re.IGNORECASE | re.DOTALL)
        if prereq_match:
            return prereq_match.group(1).strip()
        return ""
    
    def extract_units(self, content: str) -> str:
        """Extract units from content"""
        units_match = re.search(r'Units?:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
        if units_match:
            return units_match.group(1).strip()
        return ""
    
    def extract_instructor(self, content: str) -> str:
        """Extract instructor from content"""
        instructor_match = re.search(r'Instructor:\s*(.+?)(?:Description:|Prerequisites:|Units:|$)', content, re.IGNORECASE)
        if instructor_match:
            return instructor_match.group(1).strip()
        return ""

    def remove_duplicates(self, items: List[Dict], key_field: str = 'code') -> List[Dict]:
        """Remove duplicate items based on key field"""
        seen = set()
        unique_items = []
        
        for item in items:
            key = item.get(key_field, '')
            if key and key not in seen:
                seen.add(key)
                unique_items.append(item)
            else:
                self.stats['removed_duplicates'] += 1
                
        return unique_items

    def process_individual_files(self) -> Dict:
        """Process individual department files from scraper"""
        individual_dir = self.data_dir / "individual_departments"
        
        if not individual_dir.exists():
            self.logger.warning(f"Individual departments directory not found: {individual_dir}")
            return {'departments': [], 'programs': [], 'courses': []}
            
        all_departments = []
        all_programs = []
        all_courses = []
        
        # Process each individual department file
        for file_path in individual_dir.glob("*.json"):
            self.stats['total_files'] += 1
            self.logger.info(f"Processing {file_path.name}")
            
            data = self.load_json_file(file_path)
            if not data:
                continue
                
            cleaned_dept = self.clean_department_data(data)
            if cleaned_dept:
                all_departments.append(cleaned_dept)
                all_programs.extend(cleaned_dept['programs'])
                all_courses.extend(cleaned_dept['courses'])
                self.stats['processed_files'] += 1
                
        return {
            'departments': all_departments,
            'programs': all_programs,
            'courses': all_courses
        }

    def process_consolidated_files(self) -> Dict:
        """Process consolidated JSON files if they exist"""
        consolidated_data = {'departments': [], 'programs': [], 'courses': []}
        
        # Try to load consolidated files
        files_to_check = [
            'ucsb_engineering_catalog.json',
            'ucsb_engineering_departments.json',
            'ucsb_engineering_programs.json',
            'ucsb_engineering_courses.json'
        ]
        
        for filename in files_to_check:
            file_path = self.data_dir / filename
            if file_path.exists():
                self.logger.info(f"Processing consolidated file: {filename}")
                data = self.load_json_file(file_path)
                
                if filename == 'ucsb_engineering_catalog.json' and data:
                    # Main catalog file
                    if 'departments' in data:
                        for dept in data['departments']:
                            cleaned_dept = self.clean_department_data(dept)
                            if cleaned_dept:
                                consolidated_data['departments'].append(cleaned_dept)
                    if 'programs' in data:
                        for program in data['programs']:
                            cleaned_program = self.clean_program_data(program)
                            if cleaned_program:
                                consolidated_data['programs'].append(cleaned_program)
                    if 'courses' in data:
                        for course in data['courses']:
                            cleaned_course = self.clean_course_data(course)
                            if cleaned_course:
                                consolidated_data['courses'].append(cleaned_course)
                                
                elif filename == 'ucsb_engineering_departments.json' and data:
                    for dept in data:
                        cleaned_dept = self.clean_department_data(dept)
                        if cleaned_dept:
                            consolidated_data['departments'].append(cleaned_dept)
                            
                elif filename == 'ucsb_engineering_programs.json' and data:
                    for program in data:
                        cleaned_program = self.clean_program_data(program)
                        if cleaned_program:
                            consolidated_data['programs'].append(cleaned_program)
                            
                elif filename == 'ucsb_engineering_courses.json' and data:
                    for course in data:
                        cleaned_course = self.clean_course_data(course)
                        if cleaned_course:
                            consolidated_data['courses'].append(cleaned_course)
                            
        return consolidated_data

    def clean_all_data(self) -> Dict:
        """Main method to clean all data"""
        self.logger.info("Starting data cleaning process...")
        
        # Try individual files first, then consolidated
        cleaned_data = self.process_individual_files()
        
        if not any(cleaned_data.values()):
            self.logger.info("No individual files found, trying consolidated files...")
            cleaned_data = self.process_consolidated_files()
            
        # Remove duplicates
        cleaned_data['departments'] = self.remove_duplicates(cleaned_data['departments'], 'code')
        cleaned_data['programs'] = self.remove_duplicates(cleaned_data['programs'], 'code')
        cleaned_data['courses'] = self.remove_duplicates(cleaned_data['courses'], 'code')
        
        # Update statistics
        self.stats['cleaned_departments'] = len(cleaned_data['departments'])
        self.stats['cleaned_programs'] = len(cleaned_data['programs'])
        self.stats['cleaned_courses'] = len(cleaned_data['courses'])
        
        return cleaned_data

    def save_cleaned_data(self, cleaned_data: Dict):
        """Save cleaned data to output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete cleaned dataset
        output_files = [
            ('ucsb_engineering_cleaned.json', cleaned_data),
            ('ucsb_engineering_departments_cleaned.json', cleaned_data['departments']),
            ('ucsb_engineering_programs_cleaned.json', cleaned_data['programs']),
            ('ucsb_engineering_courses_cleaned.json', cleaned_data['courses'])
        ]
        
        for filename, data in output_files:
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved cleaned data to {file_path}")
            
        # Save statistics
        stats_file = self.output_dir / f'cleaning_stats_{timestamp}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
            
        self.logger.info(f"Saved cleaning statistics to {stats_file}")

    def generate_quality_report(self, cleaned_data: Dict):
        """Generate a quality report for the cleaned data"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.stats,
            'quality_metrics': {
                'departments': self.analyze_quality(cleaned_data['departments']),
                'programs': self.analyze_quality(cleaned_data['programs']),
                'courses': self.analyze_quality(cleaned_data['courses'])
            }
        }
        
        report_file = self.output_dir / 'quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Generated quality report: {report_file}")
        return report

    def analyze_quality(self, items: List[Dict]) -> Dict:
        """Analyze quality metrics for a list of items"""
        if not items:
            return {'count': 0, 'avg_content_length': 0, 'empty_content': 0}
            
        total_length = 0
        empty_content = 0
        
        for item in items:
            content = item.get('content', '') or item.get('overview', '') or item.get('description', '')
            if not content or len(content.strip()) < self.min_content_length:
                empty_content += 1
            else:
                total_length += len(content)
                
        return {
            'count': len(items),
            'avg_content_length': total_length / len(items) if items else 0,
            'empty_content': empty_content,
            'quality_score': (len(items) - empty_content) / len(items) * 100 if items else 0
        }

    def preview_cleaning(self, sample_file: str = None, num_samples: int = 3):
        """Preview cleaning results for debugging"""
        if sample_file:
            file_path = self.data_dir / "individual_departments" / sample_file
            if file_path.exists():
                data = self.load_json_file(file_path)
                if data and 'courses' in data:
                    for i, course in enumerate(data['courses'][:num_samples]):
                        print(f"\n{'='*50}")
                        print(f"SAMPLE {i+1}: {course.get('name', 'Unknown')}")
                        print(f"{'='*50}")
                        print(f"ORIGINAL CONTENT:")
                        print(course.get('content', '')[:500] + "..." if len(course.get('content', '')) > 500 else course.get('content', ''))
                        
                        cleaned = self.clean_course_data(course)
                        if cleaned:
                            print(f"\nCLEANED RESULT:")
                            print(f"Title: {cleaned['title']}")
                            print(f"Description: {cleaned['description'][:200]}..." if len(cleaned['description']) > 200 else cleaned['description'])
                            print(f"Instructor: {cleaned['instructor']}")
                            print(f"Units: {cleaned['units']}")
                            print(f"Prerequisites: {cleaned['prerequisites']}")
                        else:
                            print("\nCLEANED RESULT: [FILTERED OUT]")
        else:
            print("Please provide a sample file name from individual_departments folder")

    def run(self):
        """Run the complete data cleaning process"""
        self.logger.info("=" * 50)
        self.logger.info("UCSB Data Cleaning Process Started")
        self.logger.info("=" * 50)
        
        # Clean all data
        cleaned_data = self.clean_all_data()
        
        # Save cleaned data
        self.save_cleaned_data(cleaned_data)
        
        # Generate quality report
        report = self.generate_quality_report(cleaned_data)
        
        # Print final statistics
        self.logger.info("\n" + "=" * 50)
        self.logger.info("CLEANING COMPLETE - FINAL STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        self.logger.info(f"Cleaned departments: {self.stats['cleaned_departments']}")
        self.logger.info(f"Cleaned programs: {self.stats['cleaned_programs']}")
        self.logger.info(f"Cleaned courses: {self.stats['cleaned_courses']}")
        self.logger.info(f"Removed empty entries: {self.stats['removed_empty']}")
        self.logger.info(f"Removed gibberish entries: {self.stats['removed_gibberish']}")
        self.logger.info(f"Removed duplicates: {self.stats['removed_duplicates']}")
        self.logger.info(f"Overall quality scores:")
        self.logger.info(f"  - Departments: {report['quality_metrics']['departments']['quality_score']:.1f}%")
        self.logger.info(f"  - Programs: {report['quality_metrics']['programs']['quality_score']:.1f}%")
        self.logger.info(f"  - Courses: {report['quality_metrics']['courses']['quality_score']:.1f}%")
        
        return cleaned_data, report


if __name__ == "__main__":
    # Usage example
    cleaner = UCSBDataCleaner(data_dir="data")
    
    # Full cleaning process
    cleaned_data, report = cleaner.run()
    
    print("\nData cleaning completed!")
    print(f"Check the 'data/cleaned/' directory for results")
    print(f"Quality report saved for review")