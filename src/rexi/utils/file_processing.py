"""
File processing utilities for REXI.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO
import aiofiles

class FileProcessor:
    """File processing utilities."""
    
    # Supported file types and their processors
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.md': 'markdown',
        '.json': 'json',
        '.csv': 'csv',
        '.docx': 'docx',
        '.doc': 'docx',
        '.rtf': 'text',
        '.epub': 'epub',
        '.mobi': 'mobi',
        '.html': 'html',
        '.htm': 'html',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml'
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'pdf': 50 * 1024 * 1024,  # 50MB
        'text': 10 * 1024 * 1024,  # 10MB
        'markdown': 10 * 1024 * 1024,  # 10MB
        'json': 10 * 1024 * 1024,  # 10MB
        'csv': 10 * 1024 * 1024,  # 10MB
        'docx': 20 * 1024 * 1024,  # 20MB
        'epub': 20 * 1024 * 1024,  # 20MB
        'mobi': 20 * 1024 * 1024,  # 20MB
        'html': 10 * 1024 * 1024,  # 10MB
        'xml': 10 * 1024 * 1024,  # 10MB
        'yaml': 5 * 1024 * 1024,  # 5MB
    }
    
    def __init__(self, upload_dir: str = "data/uploads"):
        """Initialize file processor."""
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported."""
        extension = Path(filename).suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS
    
    def get_file_type(self, filename: str) -> Optional[str]:
        """Get file type from filename."""
        extension = Path(filename).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(extension)
    
    def get_max_file_size(self, filename: str) -> int:
        """Get maximum allowed size for file type."""
        file_type = self.get_file_type(filename)
        return self.MAX_FILE_SIZES.get(file_type, 10 * 1024 * 1024)  # Default 10MB
    
    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """Validate file against constraints."""
        if not self.is_supported_file(filename):
            return False, f"Unsupported file type: {Path(filename).suffix}"
        
        max_size = self.get_max_file_size(filename)
        if file_size > max_size:
            return False, f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        
        return True, "File is valid"
    
    async def save_uploaded_file(self, file: BinaryIO, filename: str) -> str:
        """Save uploaded file and return file path."""
        try:
            # Generate unique filename
            file_extension = Path(filename).suffix
            file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
            unique_filename = f"{file_hash}_{filename}"
            
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            return str(file_path)
            
        except Exception as e:
            raise Exception(f"Failed to save file: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get file information."""
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = path.stat()
            
            return {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "extension": path.suffix.lower(),
                "type": self.get_file_type(path.name),
                "mime_type": mimetypes.guess_type(str(path))[0],
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_readable": os.access(path, os.R_OK)
            }
            
        except Exception as e:
            raise Exception(f"Failed to get file info: {e}")
    
    def clean_filename(self, filename: str) -> str:
        """Clean filename for safe storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        cleaned = filename
        
        for char in unsafe_chars:
            cleaned = cleaned.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        cleaned = cleaned.strip(' .')
        
        # Ensure filename is not empty
        if not cleaned:
            cleaned = "unnamed_file"
        
        return cleaned
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            raise Exception(f"Failed to calculate file hash: {e}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to delete file: {e}")
    
    def list_files(self, pattern: str = "*") -> List[Dict[str, any]]:
        """List files in upload directory."""
        try:
            files = []
            for file_path in self.upload_dir.glob(pattern):
                if file_path.is_file():
                    files.append(self.get_file_info(str(file_path)))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x["modified"], reverse=True)
            
            return files
            
        except Exception as e:
            raise Exception(f"Failed to list files: {e}")
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics."""
        try:
            files = self.list_files()
            
            total_size = sum(f["size"] for f in files)
            file_count = len(files)
            
            # Count by type
            type_counts = {}
            for file_info in files:
                file_type = file_info["type"] or "unknown"
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            return {
                "total_files": file_count,
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": type_counts,
                "upload_dir": str(self.upload_dir)
            }
            
        except Exception as e:
            raise Exception(f"Failed to get storage stats: {e}")
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """Clean up files older than specified days."""
        try:
            import time
            
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            raise Exception(f"Failed to cleanup old files: {e}")
    
    def batch_validate_files(self, file_list: List[Tuple[str, int]]) -> Dict[str, List[str]]:
        """Validate multiple files."""
        results = {
            "valid": [],
            "invalid": []
        }
        
        for filename, file_size in file_list:
            is_valid, message = self.validate_file(filename, file_size)
            
            if is_valid:
                results["valid"].append(filename)
            else:
                results["invalid"].append(f"{filename}: {message}")
        
        return results
