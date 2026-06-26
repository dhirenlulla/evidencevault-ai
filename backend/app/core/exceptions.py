class FileUploadError(Exception):
    """ 
    Base exception for all document-upload failures.
    
    Catching this parent exception would catch any og the more
    specific upload exceptions declared below.
    """
    pass

class InvalidFileError(FileUploadError):
    """ 
    Raised when the uploaded file is empty or has an invalid filename.
    """
    pass


class UnsupportedFileTypeError(FileUploadError):
    """ 
    Raised when the uploaded file is not accepted pdf.
    """
    pass


class FileTooLargeError(FileUploadError):
    """ 
    Raised when the uploaded file exceeds the configured size limit.
    """

class FileStorageError(FileUploadError):
    """ 
    Raised when a validated file cannot be written to local storage.
    """
    pass

class PDFProcessingError(Exception):
    """ 
    Base exception for PDF parsing and text-processing failures.
    """
    pass

class PDFPathError(PDFProcessingError):
    """ 
    Raised when a supplied PDF path is missing, invalid,
    or does not point to a PDF file.
    """
    pass

class PDFExtractionError(PDFProcessingError):
    """ 
    Raised when PyMuPDF cannot open or extract text from
    a PDF document.
    """
    
    pass



"""
exception hierarchy:

Exception
├── FileUploadError
│   ├── InvalidFileError
│   ├── UnsupportedFileTypeError
│   ├── FileTooLargeError
│   └── FileStorageError
│
└── PDFProcessingError
    ├── PDFPathError
    └── PDFExtractionError

    
"""