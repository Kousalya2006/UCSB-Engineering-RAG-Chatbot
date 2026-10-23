
import json
import os
from typing import List, Dict, Any
from src.config.settings import Config


class UCSBDataProcessor:
    """Process UCSB Engineering data into documents for embedding."""

    def __init__(self, input_file: str = None):
        self.config = Config()

        # Read the existing catalog JSON file
        self.input_file = input_file or os.path.join(
            self.config.DATA_PATH,
            "ucsb_engineering_catalog.json"
        )

        self.output_file = os.path.join(
            self.config.DATA_PATH,
            "processed_documents.json"
        )

    def process_data(self) -> List[Dict[str, Any]]:
        """Transform hierarchical UCSB data into flat documents."""

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}"
            )

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Your catalog JSON contains a 'departments' key
        departments = (
            data["departments"]
            if isinstance(data, dict)
            else data
        )

        documents = []

        for dept in departments:
            dept_name = dept.get("name", "Unknown Department")
            dept_code = dept.get("code", "UNKNOWN")
            dept_overview = dept.get("overview", "")

            # 1. Department document
            documents.append({
                "doc_id": f"dept_{dept_code}",
                "content": (
                    f"Department: {dept_name} ({dept_code})\n\n"
                    f"Overview: {dept_overview}"
                ),
                "metadata": {
                    "type": "department",
                    "department": dept_name,
                    "department_code": dept_code,
                    "source": "UCSB College of Engineering"
                }
            })

            # 2. Program documents
            for i, program in enumerate(dept.get("programs", [])):
                program_name = program.get("name", "Unknown Program")
                program_code = program.get("code", "")
                description = program.get("description", "")
                url = program.get("url", "")

                documents.append({
                    "doc_id": f"prog_{dept_code}_{i}",
                    "content": (
                        f"Program: {program_name} ({program_code})\n"
                        f"Department: {dept_name}\n"
                        f"URL: {url}\n\n"
                        f"Description: {description}"
                    ),
                    "metadata": {
                        "type": "program",
                        "department": dept_name,
                        "department_code": dept_code,
                        "program": program_name,
                        "program_code": program_code,
                        "url": url,
                        "source": "UCSB College of Engineering"
                    }
                })

            # 3. Course documents
            for i, course in enumerate(dept.get("courses", [])):
                course_name = course.get("name", "")
                course_title = course.get("title", "")
                units = course.get("units", "")
                grading = course.get("grading", "")
                description = course.get("description", "")
                prerequisites = course.get("prerequisites", "")

                content = (
                    f"Course: {course_name} - {course_title}\n"
                    f"Department: {dept_name}\n"
                    f"Units: {units}\n"
                    f"Grading: {grading}\n\n"
                    f"Description: {description}\n"
                )

                if prerequisites:
                    content += f"\nPrerequisites: {prerequisites}"

                documents.append({
                    "doc_id": f"course_{dept_code}_{i}",
                    "content": content,
                    "metadata": {
                        "type": "course",
                        "department": dept_name,
                        "department_code": dept_code,
                        "course_name": course_name,
                        "course_title": course_title,
                        "units": units,
                        "grading": grading,
                        "source": "UCSB College of Engineering"
                    }
                })

        print(f"\nProcessed {len(documents)} documents.")

        for doc_type in ["department", "program", "course"]:
            count = sum(
                1 for doc in documents
                if doc["metadata"]["type"] == doc_type
            )
            print(f"- {doc_type.capitalize()} documents: {count}")

        return documents

    def save_processed_data(self, documents: List[Dict[str, Any]]):
        """Save processed documents to JSON."""

        os.makedirs(self.config.DATA_PATH, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"\nSaved processed documents to: {self.output_file}")

    def preview_documents(
        self, documents: List[Dict[str, Any]], n: int = 3
    ):
        """Preview the first few documents."""

        print(f"\nPreview of first {min(n, len(documents))} documents:")

        for i, doc in enumerate(documents[:n], start=1):
            print(f"\n--- Document {i} ---")
            print(f"ID: {doc['doc_id']}")
            print(f"Type: {doc['metadata']['type']}")
            print(f"Content: {doc['content'][:200]}...")

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""

        if not os.path.exists(self.output_file):
            return {"status": "not_processed", "documents": 0}

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                documents = json.load(f)

            stats = {
                "status": "processed",
                "total_documents": len(documents),
                "types": {},
                "departments": {}
            }

            for doc in documents:
                doc_type = doc["metadata"]["type"]
                department = doc["metadata"].get(
                    "department", "Unknown"
                )

                stats["types"][doc_type] = (
                    stats["types"].get(doc_type, 0) + 1
                )
                stats["departments"][department] = (
                    stats["departments"].get(department, 0) + 1
                )

            return stats

        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    processor = UCSBDataProcessor()

    try:
        print(f"Reading data from: {processor.input_file}")

        documents = processor.process_data()
        processor.preview_documents(documents)
        processor.save_processed_data(documents)

        print("\nProcessing complete!")
        print("Next step: Check src/core/embeddings.py before running it.")

    except FileNotFoundError as e:
        print(f"\nFile error: {e}")

    except (KeyError, TypeError, ValueError) as e:
        print(f"\nData format error: {e}")
        print("The catalog structure needs to be checked.")

    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
