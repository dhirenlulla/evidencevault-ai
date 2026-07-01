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


class PDFEncryptedError(PDFExtractionError):
    """ 
    Raised when a PDF requires a password before extraction.
    """
    pass


class PDFMalformedError(PDFExtractionError):
    """ 
    Raised when PyMuPDF cannot open or parse a PDF.
    """
    pass


class DocumentProcessingWorkflowError(Exception):
    """ 
    Base exception for document-processing workflow failures.
    """
    
    pass

class DocumentNotFoundError(DocumentProcessingWorkflowError):
    """ 
    Raised when no PostgreSQL document exists for a supplied UUID.
    """
    pass

class DocumentAlreadyProcessingError(DocumentProcessingWorkflowError):
    """ 
    Raised when a document is already marked as processing
    """
    pass

class DocumentStoragePathError(DocumentProcessingWorkflowError):
    """
    Raised when a document has a missing, unsafe or invalid local
    storage path.
    """
    pass

class TextChunkingError(Exception):
    """ 
    Base exception for text-chunking failures.
    """
    pass

class InvalidChunkingConfigurationError(TextChunkingError):
    """ 
    Raised when chunk size, overlap, or minimum-content
    settings are invalid.
    """
    pass


class DocumentNotReadyForChunkingError(DocumentProcessingWorkflowError):
    """ 
    Raised when a document has not completed text extraction.
    """
    pass


class NoChunksGeneratedError(DocumentProcessingWorkflowError):
    """ 
    Raised when an extracted document produces no usable chunks.
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
├── PDFProcessingError
|    ├── PDFPathError
|    └── PDFExtractionError
|        ├── PDFEncryptedError
|        └── PDFMalformedError
|   
├── DocumentProcessingWorkflowError
|   ├── DocumentNotFoundError
|   ├── DocumentAlreadyProcessingError
|   ├── DocumentStoragePathError
|   ├── DocumentNotReadyForChunkingError
|   └── NoChunkGeneratedError
|
└── TextChunkError
    └── InvalidChunkingConfigurationError
    
    
"""