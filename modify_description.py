#!/usr/bin/env python3
"""
Image Description Modifier Script

This script modifies image descriptions using GPT-4o-mini based on user prompts.
It loads descriptions from a JSON file, finds the matching image, and updates
the description according to the provided prompt.

Usage:
    python modify_description.py <image_filename> "<prompt>"

Example:
    python modify_description.py "IMG-20250613-WA0012.jpg" "this image depicts sunny weather"
"""

import json
import sys
import argparse
import os
from openai import OpenAI
from typing import Dict, List, Optional

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv("/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/.env")
except ImportError:
    print(
        "Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files."
    )
    pass


class ImageDescriptionModifier:
    def __init__(self, api_key: str, input_file: str = "/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/data/stanford-40-actions/gpt4/action_40_classes/name_your_experiment/step1_result.jsonl"):
        """
        Initialize the modifier with OpenAI API key and input file.

        Args:
            api_key: OpenAI API key
            input_file: Path to the file containing image descriptions
        """
        self.client = OpenAI(api_key=api_key)
        self.input_file = input_file
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        """Load image descriptions from the input file."""
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Parse each line as a separate JSON object
            data = []
            for line in content.split("\n"):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line: {line[:50]}...")
                        continue

            return data
        except FileNotFoundError:
            print(f"Error: File '{self.input_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def find_image_description(self, image_filename: str) -> Optional[Dict]:
        """
        Find the description for a specific image file.

        Args:
            image_filename: Name of the image file to find

        Returns:
            Dictionary containing the image data or None if not found
        """
        for item in self.data:
            if item.get("image_file") == image_filename:
                return item
        return None

    def modify_description(self, original_description: str, prompt: str) -> str:
        """
        Modify the image description using GPT-4o-mini based on the prompt.

        Args:
            original_description: Original image description
            prompt: User prompt for modification

        Returns:
            Modified description
        """
        system_message = """You are an expert at modifying image descriptions based on user prompts. 
        Your task is to take an existing image description and modify it according to the user's instructions, 
        while maintaining the same style, detail level, and structure as the original description.
        
        Rules:
        1. Keep the same writing style and tone as the original
        2. Maintain similar length and detail level
        3. Only change what the prompt specifically requests
        4. Preserve all other details that aren't contradicted by the prompt
        5. Make the changes feel natural and integrated into the description

        You might receive a category with a boolean value such as 'sunny: True'. 
        In this case it means that the image is that category (in this case sunny), and the description should be modified as such."""

        user_message = f"""Original description: "{original_description}"

User prompt: "{prompt}"

Please modify the original description according to the user prompt. Make sure the changes integrate naturally with the existing description while preserving its style and detail level."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return original_description

    def save_modified_data(self, output_file: str = None):
        """Save the modified data back to file."""
        if output_file is None:
            output_file = self.input_file

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Modified data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def process_image(
        self, image_filename: str, prompt: str, save_changes: bool = True
    ) -> bool:
        """
        Process a single image description modification.

        Args:
            image_filename: Name of the image file
            prompt: Modification prompt
            save_changes: Whether to save changes to file

        Returns:
            True if successful, False otherwise
        """
        # Find the image description
        image_data = self.find_image_description(image_filename)
        if not image_data:
            print(f"Error: Image '{image_filename}' not found in the data.")
            return False

        original_description = image_data.get("text", "")
        if not original_description:
            print(f"Error: No description found for image '{image_filename}'.")
            return False

        print(f"Processing image: {image_filename}")
        print(f"Original description: {original_description[:100]}...")
        print(f"Prompt: {prompt}")
        print("\nModifying description...")

        # Modify the description
        modified_description = self.modify_description(original_description, prompt)

        # Update the data
        for item in self.data:
            if item.get("image_file") == image_filename:
                item["text"] = modified_description
                break

        print(f"\nModified description: {modified_description}")

        # Save changes if requested
        if save_changes:
            self.save_modified_data()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Modify image descriptions using GPT-4o-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python modify_description.py IMG-20250613-WA0012.jpg "this image depicts sunny weather"
    python modify_description.py IMG-20250613-WA0005.jpg "there are only 5 people in the boat"
        """,
    )

    parser.add_argument("image_filename", help="Name of the image file to modify")
    parser.add_argument("prompt", help="Prompt describing the desired changes")
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--input-file",
        default="/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/data/stanford-40-actions/gpt4/action_40_classes/name_your_experiment/step1_result.jsonl",
        help="Input file containing image descriptions",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save changes to file"
    )

    args = parser.parse_args()

    # Get API key from argument, environment, or .env file
    import os

    from dotenv import load_dotenv
    load_dotenv("/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/.env")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. other one")
        print("Options:")
        print("  1. Set API_KEY in .env file")
        print("  2. Set OPENAI_API_KEY environment variable")
        print("  3. Use --api-key argument")
        sys.exit(1)

    # Create modifier and process the image
    modifier = ImageDescriptionModifier(api_key, args.input_file)
    success = modifier.process_image(
        args.image_filename, args.prompt, save_changes=not args.no_save
    )

    if success:
        print("\nDescription modified successfully!")
    else:
        print("\nFailed to modify description.")
        sys.exit(1)


if __name__ == "__main__":
    main()
