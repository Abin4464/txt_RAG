from typing import List

class TXTLoader:
    """Extracts raw text from TXT files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> str:
        """Load and extract text from TXT file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            print(f"✓ Loaded text file ({len(text)} characters)")
            return text
        
        except UnicodeDecodeError:
            # Try different encoding if UTF-8 fails
            try:
                with open(self.file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                print(f"✓ Loaded text file with latin-1 encoding ({len(text)} characters)")
                return text
            except Exception as e:
                print(f"✗ Error loading TXT: {str(e)}")
                return ""
        
        except Exception as e:
            print(f"✗ Error loading TXT: {str(e)}")
            return ""
