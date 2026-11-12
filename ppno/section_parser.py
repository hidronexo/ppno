"""Module for reading and writing sections of header-based text files.

This module provides the SectionParser class to handle files with sections
delimited by [HEADER] labels, similar to Hidronexo .hnx files.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union


class SectionParser:
    """Parse header-delimited text files.

    Handle files where sections are marked with [SECTION_NAME] and ended by
    another section or an [END] tag. Comments preceded by ';' are ignored.

    Attributes:
        file_path (Path): Path to the file to be parsed.
    """

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the parser with a file path.

        Args:
            file_path (Union[str, Path]): Path to the input file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.file_path}")

    def _get_lines(self) -> List[str]:
        """Read lines with encoding fallback (UTF-8, UTF-16, CP1252)."""
        encodings = ['utf-8', 'utf-16', 'cp1252']
        for enc in encodings:
            try:
                with self.file_path.open('r', encoding=enc) as f:
                    return f.readlines()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # If all fail, try one last time with 'latin-1' which never raises DecodeError
        with self.file_path.open('r', encoding='latin-1') as f:
            return f.readlines()

    def read_section(self, section_name: str) -> List[Tuple[int, Tuple[str, ...]]]:
        """Read a specific section from the file.

        Extract lines belonging to `[section_name]`, splitting each line into a tuple
        of strings and removing comments, preserving original line numbers.

        Args:
            section_name (str): Name of the section to read (case-insensitive).

        Returns:
            List[Tuple[int, Tuple[str, ...]]]: A list of tuples containing the line 
                number and the split content tuple.
        """
        extracted_data: List[Tuple[int, Tuple[str, ...]]] = []
        is_inside_target_section = False
        target_name = section_name.strip().upper()

        for i, line in enumerate(self._get_lines(), 1):
                clean_line = line.split(';')[0].strip()
                if not clean_line:
                    continue

                if clean_line.startswith('[') and clean_line.endswith(']'):
                    current_name = clean_line.strip('[] ').upper()
                    if is_inside_target_section:
                        break
                    if current_name == target_name:
                        is_inside_target_section = True
                    continue

                if is_inside_target_section:
                    extracted_data.append((i, tuple(clean_line.split())))

        return extracted_data

    def read(self) -> Dict[str, List[Tuple[int, str]]]:
        """Read all sections from the file into a dictionary.

        Returns:
            Dict[str, List[Tuple[int, str]]]: A dictionary where keys are section 
                names (in uppercase) and values are lists of (line_number, raw_line_content) tuples.
        """
        all_sections: Dict[str, List[Tuple[int, str]]] = {}
        current_section_name = None

        for i, line in enumerate(self._get_lines(), 1):
                clean_line = line.split(';')[0].strip()
                if not clean_line:
                    continue

                if clean_line.startswith('[') and clean_line.endswith(']'):
                    current_section_name = clean_line.strip('[] ').upper()
                    if current_section_name == 'END':
                        break
                    all_sections[current_section_name] = []
                elif current_section_name:
                    all_sections[current_section_name].append((i, clean_line))

        return all_sections

    def read_rows(self) -> List[Tuple[int, str]]:
        """Read non-empty data rows from a plain text file.

        Comments preceded by ';' are ignored. This is intended for auxiliary
        table files without section headers, such as pipe catalogs.
        """
        rows: List[Tuple[int, str]] = []
        for i, line in enumerate(self._get_lines(), 1):
            clean_line = line.split(';')[0].strip()
            if clean_line:
                rows.append((i, clean_line))
        return rows

    @staticmethod
    def line_to_tuple(line: str) -> Tuple[str, ...]:
        """Converts a space-separated string into a tuple of stripped words.

        Args:
            line: The string to be converted.

        Returns:
            A tuple of strings.
        """
        import re
        return tuple(word.strip() for word in re.split(r'[\s\t,]+', line) if word.strip())

    @staticmethod
    def tuple_to_line(data_tuple: Tuple[Union[str, float, int], ...], separator: str = '    ') -> str:
        """Joins a tuple into a single string separated by the given separator.

        Args:
            data_tuple: Tuple of data to be joined.
            separator: String used to separate elements. Defaults to 4 spaces.

        Returns:
            A joined string.
        """
        return separator.join(map(str, data_tuple))
